"""
generate_training_data.py
=========================
Entry-point: generates synthetic LLM fine-tuning data via Google Gemini.

All settings are controlled through  config/config.py — no CLI flags needed.

Run:
    python generate_training_data.py

Output:
    training-data/batch_001.json
    training-data/batch_002.json
    … (one file per batch)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

# ── resolve project root so imports always work regardless of CWD ─────────────
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from config.config import (          # noqa: E402  (after sys.path patch)
    BETWEEN_BATCH_SLEEP,
    ENV_FILE,
    GEMINI_API_BASE,
    GEMINI_MODEL,
    GENERATION_CONFIG,
    MAX_RETRIES_PER_BATCH,
    MAX_WORKERS,
    OUTPUT_DIR,
    RATE_LIMIT_CODES,
    REQUEST_TIMEOUT,
    RETRY_SLEEP_SECONDS,
    START_BATCH,
    TOTAL_BATCHES,
)
from prompt import prompt as PROMPT_TEMPLATE  # noqa: E402

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("datagen")


# ─────────────────────────────────────────────────────────────────────────────
# API-key loader
# ─────────────────────────────────────────────────────────────────────────────

def load_api_keys() -> list[str]:
    """
    Load all Gemini API keys from the .env file defined in config.

    Keys discovered (in order):
        GOOGLE_GEMINI_API_KEY
        GEMINI_API_KEY_1  …  GEMINI_API_KEY_N   (stops at first gap)

    Inline comments (e.g. "AIza…  # priya") are stripped.
    Returns a deduplicated list; exits if none are found.
    """
    load_dotenv(dotenv_path=ENV_FILE, override=True)

    keys: list[str] = []
    seen: set[str]  = set()

    def _add(raw: str) -> None:
        val = raw.split("#")[0].strip()   # strip inline comments
        if val and val not in seen:
            keys.append(val)
            seen.add(val)

    _add(os.getenv("GOOGLE_GEMINI_API_KEY", ""))

    idx = 1
    while True:
        raw = os.getenv(f"GEMINI_API_KEY_{idx}", "")
        if not raw.strip():
            break
        _add(raw)
        idx += 1

    if not keys:
        log.error(
            "No Gemini API keys found in %s. "
            "Set GOOGLE_GEMINI_API_KEY or GEMINI_API_KEY_1 … GEMINI_API_KEY_N.",
            ENV_FILE,
        )
        sys.exit(1)

    log.info("Loaded %d API key(s) from %s", len(keys), ENV_FILE.name)
    return keys


# ─────────────────────────────────────────────────────────────────────────────
# Gemini REST helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_rate_limit(exc: Exception) -> bool:
    if isinstance(exc, requests.HTTPError):
        return exc.response is not None and exc.response.status_code in RATE_LIMIT_CODES
    msg = str(exc).lower()
    return any(t in msg for t in ("429", "503", "rate limit", "resource exhausted", "quota"))


def _extract_text(data: dict) -> str:
    """Pull text from a Gemini generateContent response dict."""
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"No candidates returned. promptFeedback={data.get('promptFeedback')}")

    c0 = candidates[0]
    if "content" not in c0:
        raise RuntimeError(
            f"Gemini blocked / no content. finishReason={c0.get('finishReason')!r}"
        )

    parts = c0["content"].get("parts") or []
    if not parts or "text" not in parts[0]:
        raise RuntimeError(f"Empty parts. finishReason={c0.get('finishReason')!r}")

    return parts[0]["text"]


def gemini_generate(api_keys: list[str], prompt: str) -> str:
    """
    POST to Gemini generateContent.  Rotates through api_keys on rate-limit
    errors.  Raises RuntimeError when retries are exhausted.
    """
    url_base  = f"{GEMINI_API_BASE}/models/{GEMINI_MODEL}:generateContent"
    payload   = {
        "contents":         [{"parts": [{"text": prompt}]}],
        "generationConfig": GENERATION_CONFIG,
    }

    key_count = len(api_keys)
    key_idx   = 0
    attempt   = 0

    while attempt < MAX_RETRIES_PER_BATCH:
        api_key = api_keys[key_idx % key_count]
        url     = f"{url_base}?key={api_key}"

        try:
            resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return _extract_text(resp.json())

        except requests.HTTPError as exc:
            if _is_rate_limit(exc):
                log.warning(
                    "Rate-limit on key #%d (attempt %d/%d) — rotating…",
                    key_idx % key_count + 1, attempt + 1, MAX_RETRIES_PER_BATCH,
                )
                key_idx += 1
                attempt += 1
                if key_idx % key_count == 0:
                    log.info("All keys exhausted — sleeping %ds…", RETRY_SLEEP_SECONDS)
                    time.sleep(RETRY_SLEEP_SECONDS)
                continue
            log.error("HTTP %s: %s", exc.response.status_code, exc.response.text[:400])
            raise

        except Exception as exc:
            if _is_rate_limit(exc):
                log.warning(
                    "Network/rate error (attempt %d/%d): %s — rotating key…",
                    attempt + 1, MAX_RETRIES_PER_BATCH, exc,
                )
                key_idx += 1
                attempt += 1
                if key_idx % key_count == 0:
                    log.info("All keys exhausted — sleeping %ds…", RETRY_SLEEP_SECONDS)
                    time.sleep(RETRY_SLEEP_SECONDS)
                continue
            log.error("Unexpected error: %s", exc)
            raise

    raise RuntimeError(
        f"Batch failed: {MAX_RETRIES_PER_BATCH} attempts exhausted across "
        f"{key_count} key(s)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSONL parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_jsonl(raw: str) -> list[dict]:
    """
    Parse model output as JSONL.

    Strips markdown code fences if the model wrapped the output.
    Bad lines are logged and skipped; the rest of the batch is preserved.
    """
    raw = re.sub(r"```[a-z]*\n?", "", raw, flags=re.IGNORECASE).strip()

    records: list[dict] = []
    for lineno, line in enumerate(raw.splitlines(), 1):
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            log.warning("  Line %d — JSON error (%s): %.80s…", lineno, exc, line)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Single-batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(batch_num: int, api_keys: list[str]) -> int:
    """
    Generate one batch and save it to OUTPUT_DIR/batch_NNN.json.
    Returns the number of examples saved (0 on failure).
    Skips silently if the output file already exists.
    """
    out_file = OUTPUT_DIR / f"batch_{batch_num:03d}.json"

    if out_file.exists():
        existing = json.loads(out_file.read_text(encoding="utf-8"))
        log.info(
            "Batch %03d — skipping (exists, %d examples)", batch_num, len(existing)
        )
        return len(existing)

    # Inject batch number into the prompt template
    prompt = PROMPT_TEMPLATE.replace("[1-34]", str(batch_num))

    log.info("Batch %03d — calling Gemini (%s)…", batch_num, GEMINI_MODEL)
    t0      = time.time()
    raw     = gemini_generate(api_keys, prompt)
    elapsed = time.time() - t0

    records = parse_jsonl(raw)

    if not records:
        log.error(
            "Batch %03d — no valid records parsed (%.1fs). "
            "Saving raw output for inspection.",
            batch_num, elapsed,
        )
        raw_file = OUTPUT_DIR / f"batch_{batch_num:03d}_RAW.txt"
        raw_file.write_text(raw, encoding="utf-8")
        return 0

    out_file.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info(
        "Batch %03d — %d examples → %s  (%.1fs)",
        batch_num, len(records), out_file.name, elapsed,
    )
    return len(records)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    api_keys = load_api_keys()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    end_batch   = START_BATCH + TOTAL_BATCHES - 1
    batch_nums  = list(range(START_BATCH, end_batch + 1))
    workers     = min(MAX_WORKERS, len(batch_nums))

    log.info("=" * 60)
    log.info("Training data generator")
    log.info("  Model      : %s", GEMINI_MODEL)
    log.info("  Batches    : %d → %d  (%d total)", START_BATCH, end_batch, TOTAL_BATCHES)
    log.info("  Output     : %s", OUTPUT_DIR.resolve())
    log.info("  API keys   : %d", len(api_keys))
    log.info("  Workers    : %d (parallel)", workers)
    log.info("=" * 60)

    total_examples = 0
    failed_batches: list[int] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all batches, with a small stagger to avoid an initial burst
        future_to_batch: dict = {}
        for i, batch_num in enumerate(batch_nums):
            future = executor.submit(run_batch, batch_num, api_keys)
            future_to_batch[future] = batch_num
            if i < len(batch_nums) - 1:          # no sleep after last submission
                time.sleep(BETWEEN_BATCH_SLEEP)

        # Collect results as each future completes
        for future in as_completed(future_to_batch):
            batch_num = future_to_batch[future]
            try:
                n = future.result()
                total_examples += n
                if n == 0:
                    failed_batches.append(batch_num)
            except Exception as exc:
                log.error("Batch %03d — exception: %s", batch_num, exc)
                failed_batches.append(batch_num)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Done.")
    log.info("  Batches attempted : %d", TOTAL_BATCHES)
    log.info("  Examples saved    : %d", total_examples)
    log.info("  Failed batches    : %s", sorted(failed_batches) or "none")
    log.info("=" * 60)

    if failed_batches:
        log.warning(
            "To retry failed batches, set START_BATCH in config/config.py "
            "and re-run the script."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
