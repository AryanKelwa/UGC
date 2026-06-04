"""
src/pipelines/sft_pipeline.py
=============================
Sequentially runs data ingestion, anonymization, dataset splitting,
SFT training, model evaluation, and model export.
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
    """Run full synthetic data generation through to SFT fine-tuning, evaluation, and export."""
    root = get_project_root()
    
    log.info("=" * 60)
    log.info("Starting End-to-End SFT Pipeline Execution")
    log.info("=" * 60)
    
    # 1. Ingestion / Data Generation
    log.info("[1/6] Generating synthetic dialogue data from Gemini...")
    dataset_config = root / "configs" / "dataset" / f"{dataset_name}.yaml"
    if not run_generation(config_path=dataset_config):
        raise RuntimeError("Data generation failed.")
        
    # 2. PII Scrubbing
    log.info("[2/6] Masking personal identifiers (PII) in generated dialogue...")
    if not run_cleaning(config_path=dataset_config):
        raise RuntimeError("PII anonymization failed.")
        
    # 3. Merging and Dataset Splits
    log.info("[3/6] Formatting datasets, shuffling, and splitting train/val/test...")
    if not run_preparation(config_path=dataset_config):
        raise RuntimeError("Dataset building failed.")
        
    # 4. Trigger SFT Training
    log.info("[4/6] Triggering training loop for supervised fine-tuning...")
    model_config = root / "configs" / "model" / f"{model_name}.yaml"
    training_config = root / "configs" / "training" / f"{training_type}.yaml"
    
    run_sft(
        model_config_path=model_config,
        training_config_path=training_config,
        dataset_config_path=dataset_config
    )

    # 5. Evaluate Best Model
    log.info("[5/6] Evaluating fine-tuned model on validation set...")
    try:
        from src.evaluation.evaluate import run_evaluation
        metrics = run_evaluation(
            model_config_path=model_config,
            training_config_path=training_config,
            dataset_config_path=dataset_config,
        )
        log.info(
            "Evaluation complete — Loss: %.4f | Perplexity: %.4f | Quality: %s",
            metrics["avg_loss"], metrics["perplexity"], metrics["quality_tier"],
        )
    except Exception as exc:
        log.warning("Evaluation step failed (non-fatal): %s", exc)

    # 6. Export Model (merge LoRA + package for SageMaker)
    log.info("[6/6] Exporting model (merge LoRA adapters + SageMaker packaging)...")
    try:
        from src.deployment.export import merge_and_save_16bit, package_for_sagemaker
        merged_dir = merge_and_save_16bit()
        package_for_sagemaker(model_dir=merged_dir)
        log.info("Model exported and packaged for SageMaker deployment.")
    except Exception as exc:
        log.warning("Export step failed (non-fatal): %s", exc)
    
    log.info("=" * 60)
    log.info("End-to-End SFT Pipeline Completed Successfully!")
    log.info("=" * 60)

if __name__ == "__main__":
    execute_end_to_end_sft_pipeline()
