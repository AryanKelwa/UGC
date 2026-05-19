"""
gemini.py
======================
All Google Gemini REST API interactions.

Covers:
  - Low-level POST helper with multi-key rotation & rate-limit retry
  - Response text extractor
  - analyze_image()       — vision analysis of the product photo
  - generate_demo_prompt() — character + usage description
  - generate_image_prompt() — UGC image prompt (JSON-mode)
  - generate_video_prompts() — per-scene video prompts (JSON-mode)
"""

from __future__ import annotations

import json
import logging

import requests

from app.core.config import GEMINI_API_BASE, GEMINI_API_KEYS, GEMINI_MODEL
from app.prompts.prompts import (
    ANALYZE_IMAGE_PROMPT,
    DEMO_PROMPT_SYSTEM,
    IMAGE_PROMPT_SYSTEM,
    VIDEO_PROMPTS_SYSTEM,
)

import base64

log = logging.getLogger("pipeline.gemini")


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if the exception signals a 429 / 503 rate-limit or resource-exhausted."""
    if isinstance(exc, requests.HTTPError):
        return exc.response is not None and exc.response.status_code in (429, 503)
    err = str(exc).lower()
    return "429" in err or "503" in err or "rate limit" in err or "resource exhausted" in err


def _gemini_post(endpoint: str, payload: dict) -> dict:
    """
    Low-level Gemini REST call.  Returns parsed JSON.
    Rotates through GEMINI_API_KEYS on rate-limit errors.
    """
    last_err: Exception | None = None
    base_url = f"{GEMINI_API_BASE}/models/{GEMINI_MODEL}:{endpoint}"

    for idx, api_key in enumerate(GEMINI_API_KEYS):
        if not api_key:
            continue
        url = f"{base_url}?key={api_key}"
        try:
            resp = requests.post(url, json=payload, timeout=1120)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            last_err = e
            if _is_rate_limit_error(e) and idx < len(GEMINI_API_KEYS) - 1:
                log.warning(
                    "Gemini rate limit (key %d), retrying with next key: %s",
                    idx + 1, e,
                )
                continue
            raise
        except Exception as e:
            last_err = e
            if _is_rate_limit_error(e) and idx < len(GEMINI_API_KEYS) - 1:
                log.warning(
                    "Gemini request failed (key %d), retrying with next key: %s",
                    idx + 1, e,
                )
                continue
            raise

    if last_err:
        raise last_err
    raise RuntimeError("No valid Gemini API keys configured.")


def _gemini_text(data: dict) -> str:
    """
    Extract candidates[0].content.parts[0].text from a Gemini response.
    Raises RuntimeError with a clear message on blocked / empty responses.
    """
    candidates = data.get("candidates") or []
    if not candidates:
        fb = data.get("promptFeedback", {})
        raise RuntimeError(
            f"Gemini returned no candidates. promptFeedback: {fb}"
        ) from None
    c0 = candidates[0]
    if "content" not in c0:
        reason = c0.get("finishReason", "unknown")
        safety = c0.get("safetyRatings", [])
        raise RuntimeError(
            f"Gemini blocked or returned no content. finishReason={reason!r}, "
            f"safetyRatings={safety}. Try relaxing safety settings or rephrasing the prompt."
        ) from None
    parts = c0.get("content", {}).get("parts") or []
    if not parts or "text" not in parts[0]:
        raise RuntimeError(
            f"Gemini returned empty parts. finishReason={c0.get('finishReason')!r}"
        ) from None
    return parts[0]["text"]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_image(image_bytes: bytes) -> tuple[str, int, int]:
    """
    Equivalent to n8n node: 'Analyze image'
    Model: gemini-2.5-flash  |  Operation: image analyze
    Returns tuple of (candidates[0].content.parts[0].text, input_tokens, output_tokens)
    """
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": ANALYZE_IMAGE_PROMPT},
                    {"inlineData": {"mimeType": "image/jpeg", "data": image_b64}},
                ]
            }
        ]
    }
    data = _gemini_post("generateContent", payload)
    text = _gemini_text(data)
    usage = data.get("usageMetadata", {})
    in_tok = usage.get("promptTokenCount", 0)
    out_tok = usage.get("candidatesTokenCount", 0)
    log.info("Gemini image analysis complete. Tokens: IN=%d, OUT=%d", in_tok, out_tok)
    return text, in_tok, out_tok


def generate_demo_prompt(image_analysis_text: str) -> tuple[str, int, int]:
    """
    Equivalent to n8n node: 'Demo prompt'
    Model: gemini-2.5-flash
    Returns tuple of (candidates[0].content.parts[0].text, input_tokens, output_tokens)
    """
    payload = {
        "system_instruction": {"parts": [{"text": DEMO_PROMPT_SYSTEM}]},
        "contents": [{"parts": [{"text": image_analysis_text}]}],
    }
    data = _gemini_post("generateContent", payload)
    text = _gemini_text(data)
    usage = data.get("usageMetadata", {})
    in_tok = usage.get("promptTokenCount", 0)
    out_tok = usage.get("candidatesTokenCount", 0)
    log.info("Demo prompt generation complete. Tokens: IN=%d, OUT=%d", in_tok, out_tok)
    return text, in_tok, out_tok


def generate_image_prompt(
    demo_prompt_text: str,
    image_analysis_text: str,
    aspect_ratio: str = "9:16",
) -> dict:
    """
    Equivalent to n8n nodes: 'Image Prompt' (agent) + 'Output data' (structured parser)
    Uses Gemini instead of OpenRouter.  Returns { image_prompt, aspect_ratio_image }.

    aspect_ratio is provided by the caller (from the /generate request) and
    overrides whatever the AI chooses to ensure determinism.
    """
    user_message = (
        f"Your task: Create 1 image prompt as guided by your system guidelines.\n\n"
        f"Make sure that the reference image is depicted as ACCURATELY as possible in the resulting images, "
        f"especially all text.\n\n"
        f"These are the user's instructions: {demo_prompt_text}\n\n"
        f"Description of the reference image: {image_analysis_text}\n\n"
        f"The user's preferred aspect ratio: {aspect_ratio}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": IMAGE_PROMPT_SYSTEM}]},
        "contents": [{"parts": [{"text": user_message}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "image_prompt":       {"type": "STRING"},
                    "aspect_ratio_image": {"type": "STRING"},
                },
                "required": ["image_prompt", "aspect_ratio_image"],
            },
        },
    }
    data   = _gemini_post("generateContent", payload)
    output = json.loads(_gemini_text(data))
    usage  = data.get("usageMetadata", {})
    output["input_tokens"]  = usage.get("promptTokenCount", 0)
    output["output_tokens"] = usage.get("candidatesTokenCount", 0)
    # Override AI's aspect_ratio with the caller-provided value to ensure determinism
    output["aspect_ratio_image"] = aspect_ratio
    log.info(
        "Image prompt generated: aspect_ratio=%s, Tokens: IN=%d, OUT=%d",
        output.get("aspect_ratio_image"), output["input_tokens"], output["output_tokens"]
    )
    return output


def generate_video_prompts(
    image_prompt_field: str,
    image_analysis_text: str,
    demo_prompt_text: str,
    aspect_ratio: str = "9:16",
    scene_count: int = 3,
    language: str = "Hindi",
) -> dict:
    """
    Equivalent to n8n nodes: 'Video Prompts' (agent) + 'Scenes' (structured parser)
    Uses Gemini instead of OpenRouter.
    Returns { model, aspect_ratio, scenes: [{video_prompt}, …] }

    aspect_ratio and scene_count are provided by the caller so the AI does not
    need to infer them.
    """
    language_instruction = f"IMPORTANT GUIDELINE: The spoken dialogue MUST be in {language} language.\n\n" if language and language.lower() == 'hindi' else ""
    user_message = (
        f"Your task: Create video prompts as guided by your system guidelines.\n\n"
        f"Make sure that the reference image is depicted as ACCURATELY as possible in the resulting images, "
        f"especially all text.\n\n"
        f"For each of the scenes, make sure the dialogue runs continuously with smoother transitions that makes "
        f"sense. And always have the character just talk about the product and its benefits based on what you "
        f"understand about the brand, and then a product demo. So if it's a skincare item or a beauty-related "
        f"product, talk about its benefits and test it out on yourself, apply it on the face and demonstrate "
        f"the difference it makes; if it is a tech product, talk about it; and so on.\n\n"
        f"If the character will mention the brand name, only do so in the FIRST scene.\n\n"
        f"Cut to the part where the character tries out the product on themselves. Make sure the product is "
        f"correctly used. The character demonstrates this on camera.\n\n"
        f"These are the user's instructions: \n{image_prompt_field}\n\n"
        f"Count of videos to create: {scene_count}. Each video will be 8 seconds long. "
        f"Generate exactly {scene_count} scenes.\n\n"
        f"{language_instruction}"
        f"Description of the reference image/s. Just use this to understand who the product or character is, "
        f"don't use it as basis for the dialogue: {image_analysis_text}\n\n"
        f"The user's preferred aspect ratio: {aspect_ratio}\n"
        f"The user's preferred model: veo3_fast"
    )
    payload = {
        "system_instruction": {"parts": [{"text": VIDEO_PROMPTS_SYSTEM}]},
        "contents": [{"parts": [{"text": user_message}]}],
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "model":        {"type": "STRING"},
                    "aspect_ratio": {"type": "STRING"},
                    "scenes": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {"video_prompt": {"type": "STRING"}},
                        },
                    },
                },
                "required": ["model", "aspect_ratio", "scenes"],
            },
        },
    }
    data   = _gemini_post("generateContent", payload)
    output = json.loads(_gemini_text(data))
    usage  = data.get("usageMetadata", {})
    output["input_tokens"]  = usage.get("promptTokenCount", 0)
    output["output_tokens"] = usage.get("candidatesTokenCount", 0)
    # Override AI values with caller-provided values to ensure determinism
    output["aspect_ratio"] = aspect_ratio
    output["model"] = output.get("model", "veo3_fast") or "veo3_fast"
    log.info(
        "Video prompts generated: model=%s, aspect_ratio=%s, scenes=%d (expected %d), Tokens: IN=%d, OUT=%d",
        output.get("model"), aspect_ratio, len(output.get("scenes", [])), scene_count,
        output["input_tokens"], output["output_tokens"]
    )
    actual_count = len(output.get("scenes", []))
    if actual_count != scene_count:
        log.warning(
            "Scene count mismatch: requested %d, got %d. Using %d scenes as generated.",
            scene_count, actual_count, actual_count,
        )
    return output


def sanitize_video_prompt(original_prompt: str, error_message: str) -> tuple[str, int, int]:
    """
    Rewrite a video prompt that was rejected by KIE AI (policy violation,
    content filter, or similar prompt-related error) into a safe equivalent.

    Uses Gemini to understand the rejection reason and produce a clean prompt
    that preserves the original creative intent as closely as possible.

    Args:
        original_prompt: The video prompt that was rejected by KIE AI.
        error_message:   The error/rejection message returned by KIE AI.

    Returns:
        tuple[str, int, int]: A sanitized, policy-compliant video prompt string, input tokens, output tokens.
    """
    system_instruction = (
        "You are an expert video prompt writer for AI video generation tools. "
        "Your task is to rewrite a video prompt that was rejected by the video "
        "generation API due to a policy violation or content filter. "
        "You must preserve the original creative intent, product demonstration, "
        "character description, and scene composition as closely as possible. "
        "Remove or rephrase only the specific elements that likely caused the rejection. "
        "Return ONLY the rewritten prompt text — no explanations, no preamble, no quotes."
    )
    user_message = (
        f"The following video prompt was rejected by the video generation API:\n\n"
        f"ORIGINAL PROMPT:\n{original_prompt}\n\n"
        f"REJECTION / ERROR MESSAGE FROM API:\n{error_message}\n\n"
        f"Please rewrite the prompt to be policy-compliant while keeping the core "
        f"scene, character actions, product demonstration, and visual style intact. "
        f"Return only the rewritten prompt."
    )
    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"parts": [{"text": user_message}]}],
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ],
    }
    data = _gemini_post("generateContent", payload)
    sanitized = _gemini_text(data).strip()
    usage = data.get("usageMetadata", {})
    in_tok = usage.get("promptTokenCount", 0)
    out_tok = usage.get("candidatesTokenCount", 0)
    log.info(
        "Prompt sanitized via Gemini. Original length=%d, new length=%d, Tokens: IN=%d, OUT=%d",
        len(original_prompt), len(sanitized), in_tok, out_tok,
    )
    return sanitized, in_tok, out_tok
