"""
deploy/stacks/storage_stack.py
===============================
CDK stack for S3 model artifact storage and SageMaker IAM execution role.

IMPORTANT — Existing bucket strategy:
  This stack IMPORTS the existing S3 bucket by name rather than creating
  a new one. This preserves all existing objects and avoids naming conflicts.

  The existing bucket is: real-estate-711427856574-ap-southeast-2-an
  (set via CDK context key "existing_bucket_name" or env var AWS_S3_BUCKET)

  CDK will:
    ✅ Manage IAM permissions on the existing bucket (grant read/write to SageMaker)
    ✅ Use the bucket for model artifacts, inference I/O, and pipeline metadata
    ❌ NOT modify the bucket's settings (versioning, encryption, lifecycle rules)
    ❌ NOT delete the bucket on `cdk destroy`
"""

import os

import aws_cdk as cdk
from aws_cdk import (
    aws_s3 as s3,
    aws_iam as iam,
)
from constructs import Construct


class StorageStack(cdk.Stack):
    """Imports existing S3 bucket + creates SageMaker execution role."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── Resolve bucket name ───────────────────────────────
        # Priority: CDK context > env var > default naming pattern
        existing_bucket_name = (
            self.node.try_get_context("existing_bucket_name")
            or os.environ.get("AWS_S3_BUCKET", "")
            or f"real-estate-{self.account}-{self.region}-an"
        )

        # ── Import the EXISTING S3 bucket ─────────────────────
        # from_bucket_name() does NOT create a new bucket.
        # It creates a CDK reference so other stacks can call
        # bucket.grant_read_write(), bucket.bucket_name, etc.
        # The actual S3 bucket and all its objects are untouched.
        self.model_bucket = s3.Bucket.from_bucket_name(
            self, "ExistingLlmBucket",
            bucket_name=existing_bucket_name,
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

        # Grant SageMaker role read/write access to the existing bucket
        self.model_bucket.grant_read_write(self.sagemaker_role)

        # ── Outputs ───────────────────────────────────────────
        cdk.CfnOutput(
            self, "ModelBucketName",
            value=self.model_bucket.bucket_name,
            description="Existing S3 bucket used for model artifacts",
            export_name="LlmModelBucketName",
        )

        cdk.CfnOutput(
            self, "SageMakerRoleArn",
            value=self.sagemaker_role.role_arn,
            description="SageMaker execution role ARN",
            export_name="LlmSageMakerRoleArn",
        )
