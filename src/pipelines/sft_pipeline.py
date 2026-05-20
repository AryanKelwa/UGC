"""
src/pipelines/sft_pipeline.py
=============================
Sequentially runs data ingestion, anonymization, dataset splitting,
and triggers the Supervised Fine-Tuning (SFT) script.
"""

import logging
from pathlib import Path

from src.utils.logger import setup_logger
from src.utils.helpers import get_project_root
from src.data.ingestion import run_generation
from src.data.anonymization import run_cleaning
from src.data.dataset_builder import run_preparation
from src.training.train_sft import run_sft

log = setup_logger("pipeline.pipelines.sft")

def execute_end_to_end_sft_pipeline(
    model_name: str = "llama3_8b",
    training_type: str = "sft",
    dataset_name: str = "real_estate"
) -> None:
    """Run full synthetic data generation through to SFT fine-tuning adapter save."""
    root = get_project_root()
    
    log.info("=" * 60)
    log.info("Starting End-to-End SFT Pipeline Execution")
    log.info("=" * 60)
    
    # 1. Ingestion / Data Generation
    log.info("[1/4] Generating synthetic dialogue data from Gemini...")
    dataset_config = root / "configs" / "dataset" / f"{dataset_name}.yaml"
    if not run_generation(config_path=dataset_config):
        raise RuntimeError("Data generation failed.")
        
    # 2. PII Scrubbing
    log.info("[2/4] Masking personal identifiers (PII) in generated dialogue...")
    if not run_cleaning(config_path=dataset_config):
        raise RuntimeError("PII anonymization failed.")
        
    # 3. Merging and Dataset Splits
    log.info("[3/4] Formatting datasets, shuffling, and splitting train/val/test...")
    if not run_preparation(config_path=dataset_config):
        raise RuntimeError("Dataset building failed.")
        
    # 4. Trigger SFT Training
    log.info("[4/4] Triggering training loop for supervised fine-tuning...")
    model_config = root / "configs" / "model" / f"{model_name}.yaml"
    training_config = root / "configs" / "training" / f"{training_type}.yaml"
    
    run_sft(
        model_config_path=model_config,
        training_config_path=training_config,
        dataset_config_path=dataset_config
    )
    
    log.info("=" * 60)
    log.info("End-to-End SFT Pipeline Completed Successfully!")
    log.info("=" * 60)

if __name__ == "__main__":
    execute_end_to_end_sft_pipeline()
