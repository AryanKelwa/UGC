"""
deploy/stacks/storage_stack.py
===============================
CDK stack for S3 model artifact storage and SageMaker IAM execution role.
"""

import aws_cdk as cdk
from aws_cdk import (
    aws_s3 as s3,
    aws_iam as iam,
)
from constructs import Construct


class StorageStack(cdk.Stack):
    """S3 bucket for model artifacts + SageMaker execution role."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── S3 Bucket for model weights & inference I/O ────────
        self.model_bucket = s3.Bucket(
            self, "LlmModelBucket",
            bucket_name=cdk.Fn.sub(
                "llm-finetune-${AWS::AccountId}-${AWS::Region}"
            ),
            removal_policy=cdk.RemovalPolicy.RETAIN,
            auto_delete_objects=False,
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="CleanupOldInferenceOutputs",
                    prefix="inference-outputs/",
                    expiration=cdk.Duration.days(30),
                ),
            ],
        )

        # ── SageMaker Execution Role ──────────────────────────
        self.sagemaker_role = iam.Role(
            self, "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                ),
            ],
        )

        # Grant S3 read/write to SageMaker role
        self.model_bucket.grant_read_write(self.sagemaker_role)

        # ── Outputs ───────────────────────────────────────────
        cdk.CfnOutput(
            self, "ModelBucketName",
            value=self.model_bucket.bucket_name,
            description="S3 bucket for model artifacts",
            export_name="LlmModelBucketName",
        )

        cdk.CfnOutput(
            self, "SageMakerRoleArn",
            value=self.sagemaker_role.role_arn,
            description="SageMaker execution role ARN",
            export_name="LlmSageMakerRoleArn",
        )
