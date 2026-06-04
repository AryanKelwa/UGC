"""
src/evaluation/evaluate.py
===========================
Model evaluation module.
Computes token-level cross-entropy loss and perplexity on the
validation set, with prompt masking so that only the assistant's
response tokens contribute to the loss.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.utils.config_loader import load_yaml_config
from src.utils.logger import setup_logger
from src.utils.helpers import get_project_root
from src.utils.gpu_monitor import log_gpu_memory

log = setup_logger("pipeline.evaluation")

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


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def run_evaluation(
    model_path: str | Path | None = None,
    val_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    training_config_path: str | Path | None = None,
    dataset_config_path: str | Path | None = None,
    batch_size: int = 2,
) -> dict[str, Any]:
    """
    Evaluate a fine-tuned model on the validation set.

    Returns a dict with keys: avg_loss, perplexity, total_tokens, quality_tier.
    """
    root = get_project_root()

    # ── Resolve paths from configs ─────────────────────────────
    if model_config_path is None:
        model_config_path = root / "configs" / "model" / "llama3_8b.yaml"
    if training_config_path is None:
        training_config_path = root / "configs" / "training" / "sft.yaml"
    if dataset_config_path is None:
        dataset_config_path = root / "configs" / "dataset" / "real_estate.yaml"

    m_cfg = load_yaml_config(model_config_path)
    t_cfg = load_yaml_config(training_config_path)
    d_cfg = load_yaml_config(dataset_config_path)

    if model_path is None:
        model_path = root / "outputs" / "best_model"
    model_path = Path(model_path)

    if val_path is None:
        val_path = root / d_cfg.paths.processed_dir / "val.jsonl"
    val_path = Path(val_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. Run training first."
        )
    if not val_path.exists():
        raise FileNotFoundError(
            f"Validation file not found at {val_path}. Run dataset preparation first."
        )

    max_seq_length = m_cfg.model.get("max_seq_length", 2048)

    # ── Load model ─────────────────────────────────────────────
    log.info("Loading model from %s …", model_path)
    log_gpu_memory()

    if _USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=max_seq_length,
            load_in_4bit=m_cfg.model.get("load_in_4bit", True),
        )
        FastLanguageModel.for_inference(model)
        chat_template = t_cfg.training.get("chat_template", "llama-3")
        tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
    else:
        from src.models.load_model import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(str(model_path))
        model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    log_gpu_memory()

    # ── Load & tokenize validation samples ─────────────────────
    val_samples = _load_jsonl(val_path)
    log.info("Loaded %d evaluation samples from %s", len(val_samples), val_path.name)

    tokenized = [
        _tokenize_with_labels(s, tokenizer, max_seq_length)
        for s in val_samples
    ]
    log.info("Tokenized %d samples", len(tokenized))

    # ── Build DataLoader ───────────────────────────────────────
    pad_id = tokenizer.pad_token_id
    val_loader = DataLoader(
        tokenized,
        batch_size=batch_size,
        collate_fn=lambda batch: _collate_fn(batch, pad_id),
    )

    # ── Compute loss ───────────────────────────────────────────
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    log.info("Running evaluation over %d batches …", len(val_loader))

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            non_masked = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * non_masked
            total_tokens += non_masked

            log.debug(
                "  Batch %d/%d | Loss: %.4f | Tokens: %d",
                i + 1, len(val_loader), outputs.loss.item(), non_masked,
            )

    # ── Final metrics ──────────────────────────────────────────
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    if avg_loss < 0.5:
        quality_tier = "excellent"
        quality_msg = "🟢 Excellent — model reliably generates correct JSON output"
    elif avg_loss < 1.5:
        quality_tier = "good"
        quality_msg = "🟡 Good — minor formatting inconsistencies possible"
    else:
        quality_tier = "needs_improvement"
        quality_msg = "🔴 High loss — consider more fine-tuning data"

    log.info("━" * 40)
    log.info("  ✅ Eval Loss   : %.4f", avg_loss)
    log.info("  ✅ Perplexity  : %.4f", perplexity)
    log.info("  📊 Total Tokens: %d", total_tokens)
    log.info("  %s", quality_msg)
    log.info("━" * 40)

    return {
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
        "quality_tier": quality_tier,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_jsonl(file_path: Path) -> list[dict]:
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _tokenize_with_labels(
    sample: dict,
    tokenizer,
    max_seq_length: int = 2048,
) -> dict[str, torch.Tensor]:
    """
    Tokenize a conversation sample and create labels that mask the prompt
    tokens with -100 so loss is computed only on the assistant response.
    """
    messages = sample["conversations"]

    # Full conversation (prompt + assistant response)
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).squeeze(0)

    # Prompt only (system + user, without the last assistant turn)
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).squeeze(0)

    prompt_len = prompt_ids.shape[0]

    # Mask prompt tokens
    labels = full_ids.clone()
    labels[:prompt_len] = -100

    # Truncate
    full_ids = full_ids[:max_seq_length]
    labels = labels[:max_seq_length]

    return {"input_ids": full_ids, "labels": labels}


def _collate_fn(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Pad variable-length samples in a batch to the same length."""
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids_padded = []
    labels_padded = []
    attention_masks = []

    for b in batch:
        pad_len = max_len - b["input_ids"].shape[0]

        input_ids_padded.append(
            torch.cat([b["input_ids"], torch.full((pad_len,), pad_token_id)])
        )
        labels_padded.append(
            torch.cat([b["labels"], torch.full((pad_len,), -100)])
        )
        attention_masks.append(
            torch.cat([
                torch.ones(b["input_ids"].shape[0]),
                torch.zeros(pad_len),
            ])
        )

    return {
        "input_ids": torch.stack(input_ids_padded).long(),
        "labels": torch.stack(labels_padded).long(),
        "attention_mask": torch.stack(attention_masks).long(),
    }


if __name__ == "__main__":
    metrics = run_evaluation()
    print(metrics)
