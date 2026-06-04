"""
src/deployment/export.py
=========================
Model export utilities for post-training deployment:
  - Merge LoRA adapters into base model and save as 16-bit weights
  - Export to GGUF quantised formats
  - Push merged / GGUF models to HuggingFace Hub
  - Package model as model.tar.gz for AWS SageMaker deployment
"""

from __future__ import annotations

import os
import shutil
import tarfile
import logging
from pathlib import Path
from typing import Any

from src.utils.logger import setup_logger
from src.utils.config_loader import load_yaml_config
from src.utils.helpers import get_project_root

log = setup_logger("pipeline.deployment.export")

# ---------------------------------------------------------------------------
# Detect Unsloth
# ---------------------------------------------------------------------------
_USE_UNSLOTH = False
try:
    from unsloth import FastLanguageModel
    _USE_UNSLOTH = True
except ImportError:
    pass


def _load_model(model_path: Path, max_seq_length: int = 2048):
    """Load model for export (Unsloth preferred)."""
    if _USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
        return model, tokenizer
    else:
        from src.models.load_model import load_model_and_tokenizer
        return load_model_and_tokenizer(str(model_path))


def merge_and_save_16bit(
    model_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Merge LoRA adapters into the base model and save as 16-bit weights.

    Returns the output directory path.
    """
    root = get_project_root()
    if model_path is None:
        model_path = root / "outputs" / "best_model"
    if output_dir is None:
        output_dir = root / "outputs" / "merged_16bit"

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Merging LoRA adapters → 16-bit weights …")
    log.info("  Source : %s", model_path)
    log.info("  Output : %s", output_dir)

    if _USE_UNSLOTH:
        model, tokenizer = _load_model(model_path)
        model.save_pretrained_merged(
            str(output_dir), tokenizer, save_method="merged_16bit"
        )
    else:
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        import torch

        model = AutoPeftModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype=torch.float16, device_map="auto"
        )
        merged = model.merge_and_unload()
        merged.save_pretrained(str(output_dir))

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        tokenizer.save_pretrained(str(output_dir))

    log.info("✅ Merged 16-bit model saved to %s", output_dir)
    return output_dir


def export_gguf(
    model_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    quant_methods: list[str] | None = None,
) -> Path:
    """
    Export model to GGUF format using Unsloth.
    Requires Unsloth — raises RuntimeError if unavailable.

    Returns the output directory containing GGUF files.
    """
    if not _USE_UNSLOTH:
        raise RuntimeError(
            "GGUF export requires Unsloth. Install with: "
            "pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'"
        )

    root = get_project_root()
    if model_path is None:
        model_path = root / "outputs" / "best_model"
    if output_dir is None:
        output_dir = root / "outputs" / "gguf"
    if quant_methods is None:
        quant_methods = ["q4_k_m", "q8_0"]

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _load_model(model_path)

    log.info("Exporting GGUF quantisations: %s", quant_methods)
    for quant in quant_methods:
        log.info("  → Exporting %s …", quant)
        model.save_pretrained_gguf(str(output_dir), tokenizer, quant)

    log.info("✅ GGUF exports saved to %s", output_dir)
    return output_dir


def push_to_hub(
    model_path: str | Path | None = None,
    repo_id: str | None = None,
    push_gguf: bool = False,
    gguf_quant_methods: list[str] | None = None,
) -> None:
    """
    Push merged model (and optionally GGUF) to HuggingFace Hub.

    Reads HF_TOKEN and HF_REPO_ID from environment if not provided.
    """
    if not _USE_UNSLOTH:
        raise RuntimeError("Hub push currently requires Unsloth.")

    root = get_project_root()
    if model_path is None:
        model_path = root / "outputs" / "best_model"
    if repo_id is None:
        repo_id = os.environ.get("HF_REPO_ID")
    if not repo_id:
        raise ValueError(
            "HuggingFace repo ID not set. Pass repo_id= or set HF_REPO_ID env var."
        )

    hf_token = os.environ.get("HF_TOKEN")
    model_path = Path(model_path)

    model, tokenizer = _load_model(model_path)

    log.info("Pushing merged 16-bit model to %s …", repo_id)
    model.push_to_hub_merged(repo_id, tokenizer, save_method="merged_16bit", token=hf_token)

    if push_gguf:
        if gguf_quant_methods is None:
            gguf_quant_methods = ["q4_k_m", "q8_0"]
        gguf_repo = f"{repo_id}-GGUF"
        log.info("Pushing GGUF models to %s …", gguf_repo)
        for quant in gguf_quant_methods:
            model.push_to_hub_gguf(gguf_repo, tokenizer, quant, token=hf_token)

    log.info("✅ Model pushed to HuggingFace Hub.")


def package_for_sagemaker(
    model_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """
    Create a model.tar.gz archive from model files for SageMaker deployment.

    The tarball contains the model weights, config, and tokenizer files
    at the root level (SageMaker convention).

    Returns the path to the created tarball.
    """
    root = get_project_root()
    if model_dir is None:
        model_dir = root / "outputs" / "merged_16bit"
    if output_path is None:
        output_path = root / "outputs" / "sagemaker" / "model.tar.gz"

    model_dir = Path(model_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. "
            "Run 'merge_and_save_16bit()' first."
        )

    log.info("Packaging model for SageMaker …")
    log.info("  Source : %s", model_dir)
    log.info("  Output : %s", output_path)

    with tarfile.open(str(output_path), "w:gz") as tar:
        for item in model_dir.iterdir():
            tar.add(str(item), arcname=item.name)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info("✅ model.tar.gz created (%.1f MB) at %s", size_mb, output_path)
    return output_path


def upload_to_s3(
    local_path: str | Path,
    s3_bucket: str | None = None,
    s3_key: str | None = None,
) -> str:
    """
    Upload a file (e.g. model.tar.gz) to S3.

    Returns the S3 URI (s3://bucket/key).
    """
    import boto3

    if s3_bucket is None:
        s3_bucket = os.environ.get("AWS_S3_BUCKET")
    if not s3_bucket:
        raise ValueError("S3 bucket not configured. Set AWS_S3_BUCKET env var.")

    local_path = Path(local_path)
    if s3_key is None:
        s3_key = f"llama-finetune/{local_path.name}"

    log.info("Uploading %s → s3://%s/%s …", local_path.name, s3_bucket, s3_key)

    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), s3_bucket, s3_key)

    s3_uri = f"s3://{s3_bucket}/{s3_key}"
    log.info("✅ Upload complete: %s", s3_uri)
    return s3_uri


def run_export(
    format: str = "merged_16bit",
    model_path: str | Path | None = None,
    push_hub: bool = False,
    package_sm: bool = False,
    upload_s3: bool = False,
) -> dict[str, Any]:
    """
    CLI entry-point for the export pipeline.

    Args:
        format: "merged_16bit" | "gguf" | "all"
        model_path: Path to the fine-tuned model checkpoint.
        push_hub: Whether to push to HuggingFace Hub.
        package_sm: Whether to create model.tar.gz for SageMaker.
        upload_s3: Whether to upload to S3 after packaging.

    Returns a summary dict.
    """
    results: dict[str, Any] = {}

    if format in ("merged_16bit", "all"):
        merged_dir = merge_and_save_16bit(model_path=model_path)
        results["merged_dir"] = str(merged_dir)

    if format in ("gguf", "all"):
        gguf_dir = export_gguf(model_path=model_path)
        results["gguf_dir"] = str(gguf_dir)

    if push_hub:
        push_to_hub(model_path=model_path)
        results["pushed_to_hub"] = True

    if package_sm or upload_s3:
        merged_dir = results.get("merged_dir")
        tarball = package_for_sagemaker(model_dir=merged_dir)
        results["sagemaker_tarball"] = str(tarball)

        if upload_s3:
            s3_uri = upload_to_s3(tarball)
            results["s3_uri"] = s3_uri

    log.info("Export pipeline completed: %s", results)
    return results
