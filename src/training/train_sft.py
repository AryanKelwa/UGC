"""
src/training/train_sft.py
=========================
Supervised Fine-Tuning (SFT) training script.
Uses Unsloth FastLanguageModel for efficient 4-bit QLoRA training
with TRL's SFTTrainer. Falls back to standard HuggingFace
transformers + PEFT if Unsloth is not available.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

from src.utils.config_loader import load_yaml_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.utils.gpu_monitor import log_gpu_memory
from src.utils.helpers import get_project_root

log = setup_logger("pipeline.training.sft")

# ---------------------------------------------------------------------------
# Detect Unsloth availability
# ---------------------------------------------------------------------------
_USE_UNSLOTH = False
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    _USE_UNSLOTH = True
    log.info("Unsloth detected — using accelerated training path.")
except ImportError:
    log.warning(
        "Unsloth is NOT installed. Falling back to standard "
        "HuggingFace transformers + PEFT training path."
    )


def _load_model_unsloth(m_cfg, p_cfg):
    """Load model + tokenizer via Unsloth and attach PEFT adapters."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m_cfg.model.name_or_path,
        max_seq_length=m_cfg.model.get("max_seq_length", 2048),
        load_in_4bit=m_cfg.model.get("load_in_4bit", True),
        dtype=m_cfg.model.get("dtype", None),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=p_cfg.peft.get("r", 16),
        lora_alpha=p_cfg.peft.get("lora_alpha", 16),
        lora_dropout=p_cfg.peft.get("lora_dropout", 0),
        target_modules=list(p_cfg.peft.get("target_modules", [])),
        use_rslora=p_cfg.peft.get("use_rslora", True),
        use_gradient_checkpointing=p_cfg.peft.get(
            "use_gradient_checkpointing", "unsloth"
        ),
    )

    log.info("Unsloth model loaded. Trainable parameters:")
    model.print_trainable_parameters()

    return model, tokenizer


def _load_model_standard(m_cfg, p_cfg):
    """Load model + tokenizer via standard HuggingFace + PEFT."""
    from src.models.load_model import load_model_and_tokenizer
    from src.models.quantization import get_bnb_quantization_config
    from src.models.peft_utils import get_lora_wrapped_model

    model_name = m_cfg.model.get("hf_fallback", m_cfg.model.name_or_path)
    quant_config = None
    if m_cfg.model.get("load_in_4bit", False):
        quant_config = get_bnb_quantization_config({
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": m_cfg.model.get("torch_dtype", "bfloat16"),
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        })

    model, tokenizer = load_model_and_tokenizer(
        model_name,
        quantization_config=quant_config,
        torch_dtype=m_cfg.model.get("torch_dtype", "bfloat16"),
    )

    peft_dct = dict(p_cfg.peft) if p_cfg.peft else {}
    model = get_lora_wrapped_model(model, peft_dct)

    log.info("Standard HF model loaded. Trainable parameters:")
    model.print_trainable_parameters()

    return model, tokenizer


def _prepare_datasets(tokenizer, t_cfg, train_path: Path, val_path: Path):
    """Load JSONL datasets and apply chat-template formatting + EOS token."""
    from datasets import load_dataset

    chat_template = t_cfg.training.get("chat_template", "llama-3")

    if _USE_UNSLOTH:
        tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
    # For standard HF path the tokenizer already has a chat template from the
    # model card; we rely on apply_chat_template below.

    eos_token = tokenizer.eos_token

    def _format_and_eos(example):
        text = tokenizer.apply_chat_template(
            example["conversations"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text + eos_token}

    log.info("Loading train dataset from %s ...", train_path)
    train_ds = load_dataset("json", data_files=str(train_path), split="train")
    train_ds = train_ds.map(_format_and_eos)

    eval_ds = None
    if val_path.exists():
        log.info("Loading validation dataset from %s ...", val_path)
        eval_ds = load_dataset("json", data_files=str(val_path), split="train")
        eval_ds = eval_ds.map(_format_and_eos)
    else:
        log.warning("Validation dataset not found at %s — skipping evaluation.", val_path)

    return tokenizer, train_ds, eval_ds


def run_sft(
    model_config_path: str | Path,
    training_config_path: str | Path,
    dataset_config_path: str | Path,
) -> None:
    """Orchestrate SFT training using Unsloth (preferred) or standard HF."""
    from trl import SFTTrainer, SFTConfig
    from transformers import EarlyStoppingCallback

    root = get_project_root()

    # ── Load configs ───────────────────────────────────────────
    m_cfg = load_yaml_config(model_config_path)
    t_cfg = load_yaml_config(training_config_path)
    d_cfg = load_yaml_config(dataset_config_path)

    lora_cfg_path = root / "configs" / "training" / "lora.yaml"
    p_cfg = load_yaml_config(lora_cfg_path)

    t = t_cfg.training  # shorthand
    set_seed(t.get("seed", 0))

    log.info("=" * 60)
    log.info("Starting SFT Training Pipeline")
    log.info("  Backend    : %s", "Unsloth" if _USE_UNSLOTH else "HuggingFace + PEFT")
    log.info("  Base Model : %s", m_cfg.model.name_or_path)
    log.info("  Epochs     : %d", t.num_train_epochs)
    log.info("  Batch Size : %d (Accumulation: %d)",
             t.per_device_train_batch_size, t.gradient_accumulation_steps)
    log.info("  LR         : %s", t.learning_rate)
    log.info("=" * 60)

    # ── Load model ─────────────────────────────────────────────
    log_gpu_memory()
    if _USE_UNSLOTH:
        model, tokenizer = _load_model_unsloth(m_cfg, p_cfg)
    else:
        model, tokenizer = _load_model_standard(m_cfg, p_cfg)
    log_gpu_memory()

    # ── Prepare datasets ───────────────────────────────────────
    processed_dir = root / d_cfg.paths.processed_dir
    train_path = processed_dir / "train.jsonl"
    val_path = processed_dir / "val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing training dataset: {train_path}. "
            "Run 'python main.py prepare' first."
        )

    tokenizer, train_dataset, eval_dataset = _prepare_datasets(
        tokenizer, t_cfg, train_path, val_path
    )

    # ── Output directory ───────────────────────────────────────
    output_dir = root / "outputs" / "training_runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine precision ────────────────────────────────────
    use_bf16 = t.get("bf16", True)
    use_fp16 = t.get("fp16", False)
    if _USE_UNSLOTH:
        use_bf16 = is_bfloat16_supported()
        use_fp16 = not use_bf16

    # ── Build SFTConfig ────────────────────────────────────────
    sft_args = SFTConfig(
        # Dataset
        dataset_text_field="text",
        max_seq_length=t.get("max_seq_length", 2048),
        packing=t.get("packing", True),
        dataset_num_proc=t.get("dataset_num_proc", 2),
        # Training
        learning_rate=float(t.learning_rate),
        lr_scheduler_type=t.lr_scheduler_type,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        num_train_epochs=t.num_train_epochs,
        # Precision
        fp16=use_fp16,
        bf16=use_bf16,
        # Optimizer
        optim=t.get("optim", "adamw_8bit"),
        weight_decay=t.get("weight_decay", 0.01),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        max_grad_norm=t.get("max_grad_norm", 1.0),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        # Logging
        logging_steps=t.get("logging_steps", 1),
        report_to=t.get("report_to", "tensorboard"),
        # Evaluation
        eval_strategy=t.get("eval_strategy", "steps") if eval_dataset else "no",
        eval_steps=t.get("eval_steps", 50) if eval_dataset else None,
        # Checkpointing
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 50),
        save_total_limit=t.get("save_total_limit", 3),
        load_best_model_at_end=t.get("load_best_model_at_end", True) if eval_dataset else False,
        metric_for_best_model=t.get("metric_for_best_model", "eval_loss") if eval_dataset else None,
        greater_is_better=t.get("greater_is_better", False) if eval_dataset else None,
        # Output
        output_dir=str(output_dir),
        # Seed
        seed=t.get("seed", 0),
    )

    # ── Build callbacks ────────────────────────────────────────
    callbacks = []
    patience = t.get("early_stopping_patience", None)
    if patience and eval_dataset:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
        log.info("Early stopping enabled (patience=%d).", patience)

    # ── Initialise trainer ─────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_args,
        callbacks=callbacks,
    )

    log.info("SFTTrainer initialised — starting training loop …")
    log_gpu_memory()

    # ── Train ──────────────────────────────────────────────────
    trainer.train()

    # ── Save best model ────────────────────────────────────────
    best_model_dir = root / "outputs" / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    log.info("Best model + tokenizer saved to %s", best_model_dir)
    log_gpu_memory()
    log.info("=" * 60)
    log.info("SFT Training Completed Successfully!")
    log.info("=" * 60)


if __name__ == "__main__":
    root = get_project_root()
    run_sft(
        model_config_path=root / "configs" / "model" / "llama3_8b.yaml",
        training_config_path=root / "configs" / "training" / "sft.yaml",
        dataset_config_path=root / "configs" / "dataset" / "real_estate.yaml",
    )
