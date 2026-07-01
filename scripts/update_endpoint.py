"""
scripts/update_endpoint.py
===========================
Blue-green deployment: updates the SageMaker Async Inference endpoint
to serve a newly approved model version with zero downtime.

Reads the model package ARN from S3 (written by register_model.py),
creates a new EndpointConfig pointing to the new model weights,
then calls update_endpoint() which AWS handles atomically.

Also updates the SageMaker Model Registry status to "Approved"
and writes deployment metadata back to S3.

Usage:
    python scripts/update_endpoint.py \
        --run-id run_20260622_143000 \
        --s3-bucket llm-finetune-711427856574-ap-southeast-2 \
        --endpoint-name llama-finetune-async \
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
ENDPOINT_NAME = "llama-finetune-async"
INSTANCE_TYPE = "ml.g4dn.xlarge"
POLL_INTERVAL_SECONDS = 30
MAX_WAIT_SECONDS = 20 * 60  # 20 minutes

# HuggingFace DLC image for async inference
HF_DLC_IMAGE = (
    "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/"
    "huggingface-pytorch-inference:2.1.0-transformers4.43.0-gpu-py310-cu121-ubuntu22.04"
)


def get_account_id(session: boto3.Session) -> str:
    if ACCOUNT_ID:
        return ACCOUNT_ID
    return session.client("sts").get_caller_identity()["Account"]


def load_registration_metadata(s3_client, bucket: str, run_id: str) -> dict:
    """Load model_registration.json written by register_model.py."""
    key = f"pipeline-metadata/{run_id}/model_registration.json"
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = json.loads(obj["Body"].read())
    print(f"[update_endpoint] Loaded registration metadata: {data['model_package_arn']}")
    return data


def get_model_s3_uri(sm_client, model_package_arn: str) -> str:
    """Resolve the model.tar.gz S3 URI from the model package."""
    resp = sm_client.describe_model_package(ModelPackageName=model_package_arn)
    containers = resp["InferenceSpecification"]["Containers"]
    return containers[0]["ModelDataUrl"]


def create_sm_model(sm_client, run_id: str, model_s3_uri: str, role_arn: str) -> str:
    """Create a SageMaker Model resource from the new weights. Returns model name."""
    model_name = f"llm-ft-{run_id.replace('_', '-')[:50]}"

    try:
        sm_client.create_model(
            ModelName=model_name,
            ExecutionRoleArn=role_arn,
            PrimaryContainer={
                "Image": HF_DLC_IMAGE,
                "ModelDataUrl": model_s3_uri,
                "Environment": {
                    "HF_TASK": "text-generation",
                    "SM_NUM_GPUS": "1",
                    "MAX_INPUT_LENGTH": "2048",
                    "MAX_TOTAL_TOKENS": "4096",
                },
            },
            Tags=[
                {"Key": "Project", "Value": "llm-finetuning"},
                {"Key": "RunId", "Value": run_id},
            ],
        )
        print(f"[update_endpoint] Created SageMaker Model: {model_name}")
    except ClientError as exc:
        if "already exists" in str(exc):
            print(f"[update_endpoint] Model already exists: {model_name}")
        else:
            raise

    return model_name


def create_endpoint_config(
    sm_client,
    run_id: str,
    model_name: str,
    bucket: str,
) -> str:
    """Create a new EndpointConfig for the updated model. Returns config name."""
    config_name = f"llm-ft-async-cfg-{run_id.replace('_', '-')[:44]}"

    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": INSTANCE_TYPE,
                "InitialVariantWeight": 1.0,
            }
        ],
        AsyncInferenceConfig={
            "OutputConfig": {
                "S3OutputPath": f"s3://{bucket}/inference-outputs/",
            },
            "ClientConfig": {
                "MaxConcurrentInvocationsPerInstance": 4,
            },
        },
        Tags=[
            {"Key": "Project", "Value": "llm-finetuning"},
            {"Key": "RunId", "Value": run_id},
        ],
    )
    print(f"[update_endpoint] Created EndpointConfig: {config_name}")
    return config_name


def endpoint_exists(sm_client, endpoint_name: str) -> bool:
    """Return True if the SageMaker endpoint already exists."""
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ValidationException":
            return False
        raise


def create_endpoint(sm_client, endpoint_name: str, config_name: str) -> None:
    """Create the endpoint for the very first time (no existing endpoint)."""
    print(f"[update_endpoint] 🆕 First-time creation of endpoint '{endpoint_name}'")
    sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
    )
    _wait_for_endpoint(sm_client, endpoint_name)


def update_endpoint(sm_client, endpoint_name: str, config_name: str) -> None:
    """Issue blue-green endpoint update and wait for InService status."""
    print(f"[update_endpoint] Updating endpoint '{endpoint_name}' → config '{config_name}'")
    sm_client.update_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
        RetainAllVariantProperties=False,
    )
    _wait_for_endpoint(sm_client, endpoint_name)


def _wait_for_endpoint(sm_client, endpoint_name: str) -> None:
    """Poll until endpoint reaches InService or a terminal failure state."""
    deadline = time.time() + MAX_WAIT_SECONDS
    while time.time() < deadline:
        resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        print(f"  [{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Endpoint status: {status}")

        if status == "InService":
            print(f"[update_endpoint] ✅ Endpoint is InService.")
            return
        if status in ("Failed", "OutOfService", "RollingBack"):
            failure_reason = resp.get("FailureReason", "unknown")
            raise RuntimeError(f"Endpoint reached terminal state: {status} — {failure_reason}")

        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(f"Endpoint did not reach InService within {MAX_WAIT_SECONDS}s")


def approve_model_package(sm_client, model_package_arn: str, run_id: str) -> None:
    """Set the ModelPackage approval status to Approved in SM Model Registry."""
    sm_client.update_model_package(
        ModelPackageName=model_package_arn,
        ModelApprovalStatus="Approved",
        ApprovalDescription=f"Approved and deployed via CodePipeline. Run: {run_id}",
    )
    print(f"[update_endpoint] Model Registry status → Approved: {model_package_arn}")


def write_deployment_metadata(s3_client, bucket: str, run_id: str, config_name: str) -> None:
    """Write deployment record to S3 for traceability."""
    key = f"pipeline-metadata/{run_id}/deployment.json"
    data = {
        "run_id": run_id,
        "endpoint_name": ENDPOINT_NAME,
        "endpoint_config_name": config_name,
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "status": "deployed",
    }
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"[update_endpoint] Deployment metadata written to s3://{bucket}/{key}")


def apply_autoscaling(endpoint_name: str, region: str) -> None:
    """
    Apply scale-to-zero auto-scaling on the endpoint variant.
    Safe to call on both first creation and subsequent updates.
    Uses ApproximateBacklogSizePerInstance: scales to 0 when idle,
    scales to 1 when a request comes in.
    """
    aas_client = boto3.client("application-autoscaling", region_name=region)
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

    # Register the scalable target (idempotent — safe to call multiple times)
    aas_client.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=0,
        MaxCapacity=1,
    )

    # Apply target-tracking policy based on async backlog size
    aas_client.put_scaling_policy(
        PolicyName=f"{endpoint_name}-scale-to-zero",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 1.0,
            "CustomizedMetricSpecification": {
                "MetricName": "ApproximateBacklogSizePerInstance",
                "Namespace": "AWS/SageMaker",
                "Dimensions": [
                    {"Name": "EndpointName", "Value": endpoint_name},
                    {"Name": "VariantName",  "Value": "AllTraffic"},
                ],
                "Statistic": "Average",
            },
            "ScaleInCooldown": 300,   # 5 min before scaling in (avoid flapping)
            "ScaleOutCooldown": 60,   # 1 min before scaling out
        },
    )
    print(f"[update_endpoint] ✅ Auto-scaling (scale-to-zero) applied to {endpoint_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Blue-green update SageMaker endpoint")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--s3-bucket", default=os.environ.get("AWS_S3_BUCKET", ""))
    parser.add_argument("--endpoint-name", default=os.environ.get("ENDPOINT_NAME", ENDPOINT_NAME))
    parser.add_argument("--role-arn", default=os.environ.get("AWS_SAGEMAKER_ROLE_ARN", ""))
    parser.add_argument("--region", default=REGION)
    args = parser.parse_args()

    if not args.s3_bucket:
        print("ERROR: --s3-bucket or AWS_S3_BUCKET is required", file=sys.stderr)
        sys.exit(1)
    if not args.role_arn:
        print("ERROR: --role-arn or AWS_SAGEMAKER_ROLE_ARN is required", file=sys.stderr)
        sys.exit(1)

    session = boto3.Session(region_name=args.region)
    sm_client = session.client("sagemaker")
    s3_client = session.client("s3")

    # 1. Load what was registered
    reg_meta = load_registration_metadata(s3_client, args.s3_bucket, args.run_id)
    model_package_arn = reg_meta["model_package_arn"]

    # 2. Resolve model S3 path
    model_s3_uri = get_model_s3_uri(sm_client, model_package_arn)
    print(f"[update_endpoint] Model weights: {model_s3_uri}")

    # 3. Create SageMaker Model resource
    model_name = create_sm_model(sm_client, args.run_id, model_s3_uri, args.role_arn)

    # 4. Create new EndpointConfig
    config_name = create_endpoint_config(sm_client, args.run_id, model_name, args.s3_bucket)

    # 5. Create (first run) or update (subsequent runs) the endpoint
    if endpoint_exists(sm_client, args.endpoint_name):
        print(f"[update_endpoint] Endpoint exists — performing blue-green update.")
        update_endpoint(sm_client, args.endpoint_name, config_name)
    else:
        print(f"[update_endpoint] Endpoint does not exist yet — creating for the first time.")
        create_endpoint(sm_client, args.endpoint_name, config_name)

    # 6. Apply/refresh scale-to-zero auto-scaling (idempotent — safe on every run)
    apply_autoscaling(args.endpoint_name, args.region)

    # 7. Mark model as Approved in SM Registry
    approve_model_package(sm_client, model_package_arn, args.run_id)

    # 8. Persist deployment record
    write_deployment_metadata(s3_client, args.s3_bucket, args.run_id, config_name)

    print(f"\n[update_endpoint] ✅ Deployment complete. Model {args.run_id} is now live.")


if __name__ == "__main__":
    main()
