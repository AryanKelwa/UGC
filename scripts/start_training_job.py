"""
scripts/start_training_job.py
==============================
Launches a SageMaker Training Job using the ECR fine-tuning image
and blocks until the job reaches a terminal state.

Called by buildspec_train.yml inside the CodePipeline TRAIN stage.

Usage:
    python scripts/start_training_job.py \
        --run-id run_20260622_143000 \
        --s3-bucket llm-finetune-711427856574-ap-southeast-2 \
        --region ap-southeast-2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError


REGION = os.environ.get("AWS_REGION", "ap-southeast-2")
ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID", "")
INSTANCE_TYPE = "ml.g4dn.xlarge"
MAX_RUNTIME_SECONDS = 4 * 3600   # 4 hours hard limit
POLL_INTERVAL_SECONDS = 60


def get_account_id(session: boto3.Session) -> str:
    if ACCOUNT_ID:
        return ACCOUNT_ID
    return session.client("sts").get_caller_identity()["Account"]


def get_ecr_image_uri(account: str, region: str) -> str:
    return f"{account}.dkr.ecr.{region}.amazonaws.com/llm-finetuning:latest"


def create_training_job(
    sm_client,
    job_name: str,
    run_id: str,
    bucket: str,
    role_arn: str,
    account: str,
    region: str,
) -> str:
    """Submit the SageMaker Training Job and return its name."""

    image_uri = get_ecr_image_uri(account, region)
    dataset_s3_uri = f"s3://{bucket}/datasets/{run_id}/"
    output_s3_uri = f"s3://{bucket}/llama-finetune/{run_id}/output/"
    checkpoint_s3_uri = f"s3://{bucket}/llama-finetune/{run_id}/checkpoints/"

    print(f"[start_training_job] Submitting job: {job_name}")
    print(f"  Image:       {image_uri}")
    print(f"  Dataset:     {dataset_s3_uri}")
    print(f"  Output:      {output_s3_uri}")
    print(f"  Checkpoint:  {checkpoint_s3_uri}")
    print(f"  Instance:    {INSTANCE_TYPE}")

    sm_client.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": image_uri,
            "TrainingInputMode": "FastFile",
        },
        RoleArn=role_arn,
        InputDataConfig=[
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": dataset_s3_uri,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/jsonlines",
            }
        ],
        OutputDataConfig={
            "S3OutputPath": output_s3_uri,
        },
        CheckpointConfig={
            "S3Uri": checkpoint_s3_uri,
            "LocalPath": "/opt/ml/checkpoints",
        },
        ResourceConfig={
            "InstanceType": INSTANCE_TYPE,
            "InstanceCount": 1,
            "VolumeSizeInGB": 50,
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": MAX_RUNTIME_SECONDS,
        },
        HyperParameters={
            "model": "llama3_8b",
            "training": "sft",
        },
        Environment={
            # T4 GPU (g4dn.xlarge) does not support Flash Attention 2
            "DISABLE_FLASH_ATTN": "true",
            "TRAINING_RUN_ID": run_id,
            "SM_OUTPUT_DIR": "/opt/ml/model",
            "SM_CHECKPOINT_DIR": "/opt/ml/checkpoints",
        },
        Tags=[
            {"Key": "Project", "Value": "llm-finetuning"},
            {"Key": "RunId", "Value": run_id},
            {"Key": "ManagedBy", "Value": "codepipeline"},
        ],
    )
    return job_name


def wait_for_job(sm_client, job_name: str) -> str:
    """Poll until the job reaches a terminal state. Returns final status."""
    print(f"\n[start_training_job] Waiting for job '{job_name}' to complete...")
    terminal = {"Completed", "Failed", "Stopped"}

    while True:
        resp = sm_client.describe_training_job(TrainingJobName=job_name)
        status = resp["TrainingJobStatus"]
        secondary = resp.get("SecondaryStatus", "")
        elapsed = ""

        if resp.get("TrainingStartTime"):
            started = resp["TrainingStartTime"]
            now = datetime.now(timezone.utc)
            elapsed_min = int((now - started).total_seconds() / 60)
            elapsed = f" — {elapsed_min} min elapsed"

        print(f"  [{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
              f"{status} / {secondary}{elapsed}")

        if status in terminal:
            return status

        time.sleep(POLL_INTERVAL_SECONDS)


def write_job_metadata(job_name: str, run_id: str, final_status: str, bucket: str) -> None:
    """Write training job metadata JSON to S3 for downstream stages."""
    s3 = boto3.client("s3", region_name=REGION)
    metadata = {
        "run_id": run_id,
        "job_name": job_name,
        "status": final_status,
        "model_s3_prefix": f"s3://{bucket}/llama-finetune/{run_id}/output/",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    key = f"pipeline-metadata/{run_id}/training_job.json"
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(metadata, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"\n[start_training_job] Metadata written to s3://{bucket}/{key}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch SageMaker Training Job")
    parser.add_argument("--run-id", required=True, help="Unique run identifier (e.g. run_20260622_143000)")
    parser.add_argument("--s3-bucket", default=os.environ.get("AWS_S3_BUCKET", ""),
                        help="S3 bucket name (or set AWS_S3_BUCKET env var)")
    parser.add_argument("--role-arn", default=os.environ.get("AWS_SAGEMAKER_ROLE_ARN", ""),
                        help="SageMaker execution role ARN")
    parser.add_argument("--region", default=REGION, help="AWS region")
    args = parser.parse_args()

    if not args.s3_bucket:
        print("ERROR: --s3-bucket or AWS_S3_BUCKET env var is required", file=sys.stderr)
        sys.exit(1)
    if not args.role_arn:
        print("ERROR: --role-arn or AWS_SAGEMAKER_ROLE_ARN env var is required", file=sys.stderr)
        sys.exit(1)

    session = boto3.Session(region_name=args.region)
    sm_client = session.client("sagemaker")
    account = get_account_id(session)

    # Sanitise run_id for job name (SageMaker allows only alphanumeric + hyphens)
    job_name = f"llm-ft-{args.run_id.replace('_', '-')[:55]}"

    try:
        create_training_job(
            sm_client=sm_client,
            job_name=job_name,
            run_id=args.run_id,
            bucket=args.s3_bucket,
            role_arn=args.role_arn,
            account=account,
            region=args.region,
        )
    except ClientError as exc:
        print(f"ERROR: Failed to create training job: {exc}", file=sys.stderr)
        sys.exit(1)

    final_status = wait_for_job(sm_client, job_name)
    write_job_metadata(job_name, args.run_id, final_status, args.s3_bucket)

    if final_status != "Completed":
        print(f"\n[start_training_job] ❌ Job ended with status: {final_status}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[start_training_job] ✅ Training job completed successfully: {job_name}")


if __name__ == "__main__":
    main()
