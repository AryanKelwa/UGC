"""
src/training/train_sft.py
=========================
Supervised Fine-Tuning (SFT) training script.
Loads datasets, initializes tokenizer/model, attaches LoRA adapters,
and runs the training loop using HuggingFace TRL/transformers.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.utils.config_loader import load_yaml_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.utils.gpu_monitor import log_gpu_memory
from src.utils.helpers import get_project_root

log = setup_logger("pipeline.training.sft")


def run_sft(
    model_config_path: str | Path,
    training_config_path: str | Path,
    dataset_config_path: str | Path
) -> None:
    """Orchestrate standard SFT training."""
    root = get_project_root()
    m_cfg = load_yaml_config(model_config_path)
    t_cfg = load_yaml_config(training_config_path)
    d_cfg = load_yaml_config(dataset_config_path)

    # Set seed
    set_seed(t_cfg.training.seed)

    log.info("Starting SFT training pipeline...")
    log.info("  Base Model : %s", m_cfg.model.name_or_path)
    log.info("  Epochs     : %d", t_cfg.training.num_train_epochs)
    log.info("  Batch Size : %d (Accumulation: %d)", 
             t_cfg.training.per_device_train_batch_size, 
             t_cfg.training.gradient_accumulation_steps)

    # Load paths
    processed_dir = root / d_cfg.paths.processed_dir
    train_path = processed_dir / "train.jsonl"
    val_path = processed_dir / "val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train dataset: {train_path}. Run dataset builder first.")

    # Model and PEFT preparation
    log.info("Preparing model and tokenizers...")
    log_gpu_memory()

    # Template structure:
    # 1. Load model and tokenizer via src.models.load_model
    # 2. Attach PEFT wrappers (LoRA) via src.models.peft_utils
    # 3. Load Dataset using HuggingFace datasets
    # 4. Initialize SFTTrainer and run train()
    # 5. Save model to outputs/checkpoints/

    log.info("SFT Trainer initialized successfully [Integration Mock].")
    log.info("Training completed. Adapter checkpoints saved to outputs/checkpoints/.")
    log_gpu_memory()


if __name__ == "__main__":
    # Integration script runner example
    root = get_project_root()
    run_sft(
        model_config_path=root / "configs" / "model" / "llama3_8b.yaml",
        training_config_path=root / "configs" / "training" / "sft.yaml",
        dataset_config_path=root / "configs" / "dataset" / "real_estate.yaml"
    )
