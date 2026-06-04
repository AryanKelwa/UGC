"""
src/inference/generate.py
=========================
Performs local model inference using a fine-tuned checkpoint.
Supports Unsloth FastLanguageModel (preferred) and standard HuggingFace.
Given a raw lead qualification dialogue, runs text generation to return
predicted metrics as JSON.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.utils.logger import setup_logger
from src.utils.config_loader import load_yaml_config
from src.utils.helpers import get_project_root
from src.data.formatting import SYSTEM_PROMPT

log = setup_logger("pipeline.inference.generate")

# ---------------------------------------------------------------------------
# Detect Unsloth
# ---------------------------------------------------------------------------
_USE_UNSLOTH = False
try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    _USE_UNSLOTH = True
except ImportError:
    pass


def load_finetuned_model(
    model_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    training_config_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """
    Load a fine-tuned model and tokenizer for inference.

    Returns (model, tokenizer) tuple ready for generation.
    """
    root = get_project_root()

    if model_config_path is None:
        model_config_path = root / "configs" / "model" / "llama3_8b.yaml"
    if training_config_path is None:
        training_config_path = root / "configs" / "training" / "sft.yaml"

    m_cfg = load_yaml_config(model_config_path)
    t_cfg = load_yaml_config(training_config_path)

    if model_path is None:
        model_path = root / "outputs" / "best_model"
    model_path = Path(model_path)

    max_seq_length = m_cfg.model.get("max_seq_length", 2048)

    if _USE_UNSLOTH:
        log.info("Loading fine-tuned model via Unsloth from %s …", model_path)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=max_seq_length,
            load_in_4bit=m_cfg.model.get("load_in_4bit", True),
        )
        FastLanguageModel.for_inference(model)

        chat_template = t_cfg.training.get("chat_template", "llama-3")
        tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
    else:
        log.info("Loading fine-tuned model via HuggingFace from %s …", model_path)
        from src.models.load_model import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(str(model_path))
        model.eval()

    log.info("Model loaded and ready for inference.")
    return model, tokenizer


def generate_qualification(
    model,
    tokenizer,
    conversation_text: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    do_sample: bool = True,
) -> str:
    """
    Run generation on a conversation dialogue to get lead qualification
    JSON output.

    Args:
        model: The loaded model (Unsloth or HuggingFace).
        tokenizer: The associated tokenizer.
        conversation_text: Raw conversation text from a lead.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (low for consistent JSON).
        do_sample: Whether to use sampling; set False for greedy.

    Returns:
        The generated response string (should be valid JSON).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Lead conversation:\n{conversation_text}"},
    ]

    # Tokenize using the chat template
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    log.info("Generating qualification response …")
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode only the newly generated tokens
    generated = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response


def run_inference_cli(
    conversation_text: str | None = None,
    model_path: str | Path | None = None,
) -> str:
    """CLI entry-point: load model, run inference on a sample, print result."""
    root = get_project_root()
    gen_cfg = load_yaml_config(root / "configs" / "inference" / "generation.yaml")

    model, tokenizer = load_finetuned_model(model_path=model_path)

    if conversation_text is None:
        conversation_text = (
            "User: Hi, I'm looking to buy a home in Austin, TX. "
            "Budget is around $600k.\n"
            "Agent: Great! Are you pre-approved and what's your timeline?\n"
            "User: Yes, pre-approved. We want to close within 3 months."
        )

    result = generate_qualification(
        model,
        tokenizer,
        conversation_text,
        max_new_tokens=gen_cfg.generation.get("max_new_tokens", 512),
        temperature=gen_cfg.generation.get("temperature", 0.1),
        do_sample=gen_cfg.generation.get("do_sample", True),
    )

    print("\n" + "━" * 40)
    print("Generated Qualification:")
    print("━" * 40)
    print(result)
    print("━" * 40)
    return result
