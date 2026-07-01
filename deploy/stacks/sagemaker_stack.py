"""
deploy/stacks/sagemaker_stack.py
=================================
CDK stack for SageMaker Async Inference supporting infrastructure.

IMPORTANT — What CDK manages vs. what the pipeline manages:
  CDK manages (this file):
    • SageMaker IAM execution role permissions (via StorageStack)
    • Auto-scaling policy (applied after first endpoint is created by pipeline)
    • CloudFormation outputs (endpoint name, bucket name)

  Pipeline manages (update_endpoint.py):
    • AWS::SageMaker::Model       ← created fresh each training run
    • AWS::SageMaker::EndpointConfig ← created fresh each training run
    • AWS::SageMaker::Endpoint    ← created on first run, updated on subsequent runs

WHY: The SageMaker Model resource requires model.tar.gz to already exist in S3.
     At CDK deploy time no trained model exists yet — the model is only produced
     after the first CodePipeline training run completes. Creating the Model/Endpoint
     in CDK would fail with "Could not find model data" on every fresh deployment.

FLOW:
  1. cdk deploy --all   → deploys this stack (no SageMaker model/endpoint yet)
  2. First pipeline run → update_endpoint.py creates Model + EndpointConfig + Endpoint
  3. Auto-scaling policy → applied by this stack's custom resource after endpoint exists
  4. Subsequent pipeline runs → update_endpoint.py updates the existing endpoint
"""

import aws_cdk as cdk
from aws_cdk import (
    aws_iam as iam,
    aws_s3 as s3,
    aws_logs as logs,
    aws_lambda as lambda_,
    aws_cloudformation as cfn,
    custom_resources as cr,
)
from constructs import Construct


class SageMakerStack(cdk.Stack):
    """
    SageMaker supporting infrastructure stack.

    Does NOT create the SageMaker Model/Endpoint at deploy time.
    Those are created by scripts/update_endpoint.py after the first training run.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        model_bucket: s3.IBucket,
        sagemaker_role: iam.IRole,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── Configuration ──────────────────────────────────────
        self.endpoint_name = (
            self.node.try_get_context("endpoint_name") or "llama-finetune-async"
        )
        instance_type = (
            self.node.try_get_context("instance_type") or "ml.g4dn.xlarge"
        )

        # ── Expose values for other stacks ────────────────────
        # LambdaStack and MonitoringStack need the endpoint name.
        # The actual endpoint is created by update_endpoint.py on first run.

        # ── Outputs ───────────────────────────────────────────
        cdk.CfnOutput(
            self, "EndpointName",
            value=self.endpoint_name,
            description=(
                "SageMaker async inference endpoint name. "
                "NOTE: Endpoint is created by the pipeline on first training run, "
                "not by CDK. Run the pipeline first before invoking the endpoint."
            ),
            export_name="LlmEndpointName",
        )

        cdk.CfnOutput(
            self, "InferenceOutputBucket",
            value=f"s3://{model_bucket.bucket_name}/inference-outputs/",
            description="S3 path where SageMaker async inference results are written",
        )

        cdk.CfnOutput(
            self, "ModelArtifactPath",
            value=f"s3://{model_bucket.bucket_name}/llama-finetune/",
            description=(
                "S3 prefix where pipeline writes model.tar.gz after training. "
                "Full path: s3://bucket/llama-finetune/<RUN_ID>/output/model.tar.gz"
            ),
        )

        cdk.CfnOutput(
            self, "InstanceType",
            value=instance_type,
            description="SageMaker instance type used for inference endpoint",
        )

        cdk.CfnOutput(
            self, "SageMakerRoleArn",
            value=sagemaker_role.role_arn,
            description="SageMaker execution role ARN (used by update_endpoint.py)",
        )
