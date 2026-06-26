"""
scripts/register_model.py
==========================
Registers a trained model into the SageMaker Model Registry as a
ModelPackage with status "PendingManualApproval".

Reads eval metrics from S3 (written by the evaluate stage) and
packages the model artifact into the "llm-finetuning-registry"
Model Package Group.

Sends an SNS notification so the reviewer knows a model is waiting.

Usage:
    python scripts/register_model.py \
        --run-id run_20260622_143000 \
        --s3-bucket llm-finetune-711427856574-ap-southeast-2 \
        --sns-topic-arn arn:aws:sns:ap-southeast-2:711427856574:llm-model-approval-notifications
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError


REGION = os.environ.get("AWS_REGION", "ap-southeast-2")
ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID", "")
MODEL_PACKAGE_GROUP = "llm-finetuning-registry"

# HuggingFace DLC image for inference (ap-southeast-2, PyTorch 2.1, TF 4.43)
HF_DLC_IMAGE = (
    "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/"
    "huggingface-pytorch-inference:2.1.0-transformers4.43.0-gpu-py310-cu121-ubuntu22.04"
)


def ensure_model_package_group(sm_client) -> None:
    """Create the Model Package Group if it doesn't exist yet."""
    try:
        sm_client.create_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP,
            ModelPackageGroupDescription=(
                "LLaMA 3.1 8B fine-tuned models for real-estate lead qualification."
            ),
            Tags=[{"Key": "Project", "Value": "llm-finetuning"}],
        )
        print(f"[register_model] Created Model Package Group: {MODEL_PACKAGE_GROUP}")
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ValidationException":
            print(f"[register_model] Model Package Group already exists: {MODEL_PACKAGE_GROUP}")
        else:
            raise


def load_eval_metrics(s3_client, bucket: str, run_id: str) -> dict:
    """Load evaluation metrics JSON written by the evaluate stage."""
    key = f"pipeline-metadata/{run_id}/eval_metrics.json"
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        metrics = json.loads(obj["Body"].read())
        print(f"[register_model] Loaded eval metrics from s3://{bucket}/{key}")
        return metrics
    except ClientError as exc:
        print(f"[register_model] WARNING: Could not load eval metrics ({exc}). Using empty dict.")
        return {}


def load_training_metadata(s3_client, bucket: str, run_id: str) -> dict:
    """Load training job metadata JSON written by start_training_job.py."""
    key = f"pipeline-metadata/{run_id}/training_job.json"
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError:
        return {}


def register_model_package(
    sm_client,
    run_id: str,
    bucket: str,
    metrics: dict,
    training_meta: dict,
    role_arn: str,
) -> str:
    """Create a ModelPackage in PendingManualApproval state. Returns the ARN."""
    model_s3_uri = (
        training_meta.get("model_s3_prefix")
        or f"s3://{bucket}/llama-finetune/{run_id}/output/model.tar.gz"
    )

    # Flatten metrics to strings for CustomerMetadataProperties (SM requirement)
    metadata_props = {
        "run_id": run_id,
        "val_loss": str(metrics.get("val_loss", "n/a")),
        "perplexity": str(metrics.get("perplexity", "n/a")),
        "quality_tier": str(metrics.get("quality_tier", "n/a")),
        "record_count": str(metrics.get("record_count", "n/a")),
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }

    description = (
        f"LLaMA 3.1 8B SFT | run: {run_id} | "
        f"ppl={metrics.get('perplexity', '?')} | "
        f"val_loss={metrics.get('val_loss', '?')} | "
        f"tier={metrics.get('quality_tier', '?')}"
    )

    print(f"[register_model] Registering model package:")
    print(f"  Group:       {MODEL_PACKAGE_GROUP}")
    print(f"  Model S3:    {model_s3_uri}")
    print(f"  Description: {description}")

    response = sm_client.create_model_package(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelPackageDescription=description,
        InferenceSpecification={
            "Containers": [
                {
                    "Image": HF_DLC_IMAGE,
                    "ModelDataUrl": model_s3_uri,
                    "Environment": {
                        "HF_TASK": "text-generation",
                        "SM_NUM_GPUS": "1",
                        "MAX_INPUT_LENGTH": "2048",
                        "MAX_TOTAL_TOKENS": "4096",
                    },
                }
            ],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.g4dn.xlarge"],
            "SupportedAsyncInferenceInstanceTypes": ["ml.g4dn.xlarge"],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        ModelApprovalStatus="PendingManualApproval",
        CustomerMetadataProperties=metadata_props,
        Tags=[
            {"Key": "Project", "Value": "llm-finetuning"},
            {"Key": "RunId", "Value": run_id},
        ],
    )

    arn = response["ModelPackageArn"]
    print(f"[register_model] ✅ Registered: {arn}")
    return arn


def send_approval_notification(
    sns_client,
    topic_arn: str,
    run_id: str,
    model_package_arn: str,
    metrics: dict,
    bucket: str,
) -> None:
    """Send SNS email to reviewer with model details and approval link."""
    region = REGION
    console_link = (
        f"https://{region}.console.aws.amazon.com/sagemaker/home"
        f"?region={region}#/model-registry/{MODEL_PACKAGE_GROUP}"
    )

    subject = f"[ACTION REQUIRED] LLM Model {run_id[:20]} Ready for Review"

    body = f"""
=== LLM Fine-Tuning — New Model Ready for Production Approval ===

A new fine-tuned LLaMA 3.1 8B model has passed automated evaluation
and is waiting for your review and approval before going live.

──────────────────────────────────────────────────
  Run ID:          {run_id}
  Val Loss:        {metrics.get('val_loss', 'N/A')}
  Perplexity:      {metrics.get('perplexity', 'N/A')}
  Quality Tier:    {metrics.get('quality_tier', 'N/A')}
  Training Recs:   {metrics.get('record_count', 'N/A')} new samples
──────────────────────────────────────────────────

  Model Package:   {model_package_arn}
  S3 Path:         s3://{bucket}/llama-finetune/{run_id}/

HOW TO REVIEW:
  1. Open SageMaker Model Registry in AWS Console:
     {console_link}

  2. Find the latest "PendingManualApproval" package.

  3. Review the eval metrics and approve or reject.

  4. The CodePipeline HUMAN_APPROVAL stage is paused.
     Approve/Reject directly from the pipeline console, or use:

     aws codepipeline list-pipeline-executions --pipeline-name LlmFineTunePipeline --region {region}

NOTE: If no action is taken within 7 days, the pipeline will
      automatically expire and the model will NOT be deployed.

-- LLM Fine-Tuning Pipeline (automated notification)
"""

    try:
        sns_client.publish(TopicArn=topic_arn, Subject=subject, Message=body)
        print(f"[register_model] SNS notification sent to topic: {topic_arn}")
    except ClientError as exc:
        print(f"[register_model] WARNING: Failed to send SNS notification: {exc}")


def write_registration_metadata(s3_client, bucket: str, run_id: str, model_package_arn: str) -> None:
    """Persist the model package ARN to S3 for the deploy stage to read."""
    key = f"pipeline-metadata/{run_id}/model_registration.json"
    data = {
        "run_id": run_id,
        "model_package_arn": model_package_arn,
        "model_package_group": MODEL_PACKAGE_GROUP,
        "approval_status": "PendingManualApproval",
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"[register_model] Registration metadata written to s3://{bucket}/{key}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Register model in SageMaker Model Registry")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--s3-bucket", default=os.environ.get("AWS_S3_BUCKET", ""))
    parser.add_argument("--role-arn", default=os.environ.get("AWS_SAGEMAKER_ROLE_ARN", ""))
    parser.add_argument("--sns-topic-arn", default=os.environ.get("SNS_APPROVAL_TOPIC_ARN", ""),
                        help="SNS topic ARN for reviewer notification (optional)")
    parser.add_argument("--region", default=REGION)
    args = parser.parse_args()

    if not args.s3_bucket:
        print("ERROR: --s3-bucket or AWS_S3_BUCKET is required", file=sys.stderr)
        sys.exit(1)

    session = boto3.Session(region_name=args.region)
    sm_client = session.client("sagemaker")
    s3_client = session.client("s3")

    ensure_model_package_group(sm_client)

    metrics = load_eval_metrics(s3_client, args.s3_bucket, args.run_id)
    training_meta = load_training_metadata(s3_client, args.s3_bucket, args.run_id)

    model_package_arn = register_model_package(
        sm_client=sm_client,
        run_id=args.run_id,
        bucket=args.s3_bucket,
        metrics=metrics,
        training_meta=training_meta,
        role_arn=args.role_arn,
    )

    write_registration_metadata(s3_client, args.s3_bucket, args.run_id, model_package_arn)

    if args.sns_topic_arn:
        sns_client = session.client("sns")
        send_approval_notification(
            sns_client=sns_client,
            topic_arn=args.sns_topic_arn,
            run_id=args.run_id,
            model_package_arn=model_package_arn,
            metrics=metrics,
            bucket=args.s3_bucket,
        )

    print(f"\n[register_model] ✅ Model {args.run_id} registered and pending approval.")


if __name__ == "__main__":
    main()
