"""
src/data/ingestion.py
=====================
Handles synthetic data generation using Google Gemini REST API.
Loads API keys, rotates them on rate-limit exceptions, and executes thread-pool multi-threaded generation.
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

from src.utils.config_loader import load_yaml_config
from src.utils.helpers import get_project_root
from src.data.formatting import LEAD_QUALIFICATION_PROMPT

log = logging.getLogger("pipeline.data.ingestion")


def load_api_keys(env_file_path: Path) -> list[str]:
    """
    Load all Gemini API keys from the environment file.
    Deduplicates keys and strips comments.
    """
    load_dotenv(dotenv_path=env_file_path, override=True)

    keys: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        if not raw:
            return
        val = raw.split("#")[0].strip()  # Remove comments
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
            "No Gemini API keys found in %s. Configure GOOGLE_GEMINI_API_KEY "
            "or GEMINI_API_KEY_1...GEMINI_API_KEY_N.",
            env_file_path,
        )
        sys.exit(1)

    log.debug("Successfully loaded %d API key(s) from .env", len(keys))
    return keys


class GeminiClient:
    """Standard Gemini REST client with automatic key rotation and rate-limit retry capacity."""
    def __init__(
        self,
        api_keys: list[str],
        base_url: str,
        model_name: str,
        generation_config: dict,
        rate_limit_codes: list[int],
        max_retries: int,
        retry_sleep: int,
        timeout: int
    ) -> None:
        self.api_keys = api_keys
        self.base_url = base_url
        self.model_name = model_name
        self.generation_config = generation_config
        self.rate_limit_codes = set(rate_limit_codes)
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.timeout = timeout
        self._key_count = len(self.api_keys)
        self._current_idx = 0

    def _is_rate_limit(self, exc: Exception) -> bool:
        if isinstance(exc, requests.HTTPError):
            return exc.response is not None and exc.response.status_code in self.rate_limit_codes
        msg = str(exc).lower()
        return any(t in msg for t in ("429", "503", "rate limit", "resource exhausted", "quota"))

    def _extract_text(self, data: dict) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"No candidates returned. promptFeedback={data.get('promptFeedback')}")

        c0 = candidates[0]
        if "content" not in c0:
            raise RuntimeError(f"Blocked by filters/empty content. Reason={c0.get('finishReason')!r}")

        parts = c0["content"].get("parts") or []
        if not parts or "text" not in parts[0]:
            raise RuntimeError("Empty response parts.")

        return parts[0]["text"]

    def generate(self, prompt: str) -> str:
        url_endpoint = f"{self.base_url}/models/{self.model_name}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": self.generation_config,
        }

        attempt = 0
        while attempt < self.max_retries:
            api_key = self.api_keys[self._current_idx % self._key_count]
            url = f"{url_endpoint}?key={api_key}"

            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return self._extract_text(resp.json())

            except requests.HTTPError as exc:
                if self._is_rate_limit(exc):
                    log.warning(
                        "Rate limit on key #%d (attempt %d/%d) — rotating...",
                        (self._current_idx % self._key_count) + 1, attempt + 1, self.max_retries
                    )
                    self._current_idx += 1
                    attempt += 1
                    if self._current_idx % self._key_count == 0:
                        log.info("All keys tried. Sleeping %ds before retry...", self.retry_sleep)
                        time.sleep(self.retry_sleep)
                    continue
                log.error("HTTP %s: %s", exc.response.status_code, exc.response.text[:300])
                raise

            except Exception as exc:
                if self._is_rate_limit(exc):
                    log.warning(
                        "Connection limit on key #%d (attempt %d/%d): %s — rotating...",
                        (self._current_idx % self._key_count) + 1, attempt + 1, self.max_retries, exc
                    )
                    self._current_idx += 1
                    attempt += 1
                    if self._current_idx % self._key_count == 0:
                        log.info("All keys tried. Sleeping %ds before retry...", self.retry_sleep)
                        time.sleep(self.retry_sleep)
                    continue
                log.error("Unhandled network error: %s", exc)
                raise

        raise RuntimeError(f"Exhausted all retry attempts ({self.max_retries}) across {self._key_count} keys.")


def parse_jsonl(raw: str) -> list[dict]:
    """Parse raw JSONL text strings into dictionary records."""
    raw = re.sub(r"```[a-z]*\n?", "", raw, flags=re.IGNORECASE).strip()
    records: list[dict] = []
    for lineno, line in enumerate(raw.splitlines(), 1):
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            log.warning("Line %d — JSON decoding error (%s): %.80s...", lineno, exc, line)
    return records


def run_batch(batch_num: int, client: GeminiClient, raw_dir: Path) -> int:
    """Generate a single batch and save to raw_dir."""
    out_file = raw_dir / f"batch_{batch_num:03d}.json"
    if out_file.exists():
        try:
            existing = json.loads(out_file.read_text(encoding="utf-8"))
            log.info("Batch %03d — skipping (exists, %d examples)", batch_num, len(existing))
            return len(existing)
        except Exception:
            log.warning("Batch %03d — file corrupted, recreating...", batch_num)

    # Prepare prompt
    prompt = LEAD_QUALIFICATION_PROMPT.replace("[1-34]", str(batch_num))

    log.info("Batch %03d — requesting generation from Gemini (%s)...", batch_num, client.model_name)
    t0 = time.time()
    try:
        raw = client.generate(prompt)
    except Exception as exc:
        log.error("Batch %03d — generation failed: %s", batch_num, exc)
        return 0
    elapsed = time.time() - t0

    records = parse_jsonl(raw)
    if not records:
        log.error("Batch %03d — no valid records generated (%.1fs). Saving raw output.", batch_num, elapsed)
        raw_file = raw_dir / f"batch_{batch_num:03d}_RAW.txt"
        raw_file.write_text(raw, encoding="utf-8")
        return 0

    out_file.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Batch %03d — saved %d examples in %s (%.1fs)", batch_num, len(records), out_file.name, elapsed)
    return len(records)


def run_generation(
    config_path: str | Path | None = None,
    start_batch: int | None = None,
    total_batches: int | None = None
) -> bool:
    """Run data ingestion / generation pipeline."""
    root = get_project_root()
    if config_path is None:
        config_path = root / "configs" / "dataset" / "real_estate.yaml"

    cfg = load_yaml_config(config_path)

    # Dynamic settings override
    s_batch = start_batch if start_batch is not None else cfg.generation.start_batch
    t_batches = total_batches if total_batches is not None else cfg.generation.total_batches
    
    # Paths resolution
    env_path = root / cfg.paths.env_file
    raw_dir = root / cfg.paths.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Load API Keys
    api_keys = load_api_keys(env_path)

    # Create Gemini Client
    client = GeminiClient(
        api_keys=api_keys,
        base_url=cfg.api.base_url,
        model_name=cfg.api.model_name,
        generation_config=dict(cfg.api.generation_config),
        rate_limit_codes=list(cfg.generation.rate_limit_codes),
        max_retries=cfg.generation.max_retries_per_batch,
        retry_sleep=cfg.generation.retry_sleep_seconds,
        timeout=cfg.generation.request_timeout,
    )

    end_batch = s_batch + t_batches - 1
    batch_nums = list(range(s_batch, end_batch + 1))
    workers = min(cfg.generation.max_workers, len(batch_nums))

    log.info("=" * 60)
    log.info("Starting Ingestion Generation Pipeline")
    log.info("  Model        : %s", client.model_name)
    log.info("  Batch Range  : %d -> %d (%d total)", s_batch, end_batch, t_batches)
    log.info("  Raw Out Dir  : %s", raw_dir.resolve())
    log.info("  Parallelism  : %d threads", workers)
    log.info("=" * 60)

    total_examples = 0
    failed_batches: list[int] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_batch = {}
        for i, batch_num in enumerate(batch_nums):
            future = executor.submit(run_batch, batch_num, client, raw_dir)
            future_to_batch[future] = batch_num
            if i < len(batch_nums) - 1:
                time.sleep(cfg.generation.between_batch_sleep)

        for future in as_completed(future_to_batch):
            batch_num = future_to_batch[future]
            try:
                n = future.result()
                total_examples += n
                if n == 0:
                    failed_batches.append(batch_num)
            except Exception as exc:
                log.error("Batch %03d — exception raised: %s", batch_num, exc)
                failed_batches.append(batch_num)

    log.info("=" * 60)
    log.info("Ingestion Generation Complete.")
    log.info("  Attempted Batches : %d", len(batch_nums))
    log.info("  Successful Recs   : %d", total_examples)
    log.info("  Failed Batches    : %s", sorted(failed_batches) or "none")
    log.info("=" * 60)

    return len(failed_batches) == 0
