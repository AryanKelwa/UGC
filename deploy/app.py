#!/usr/bin/env python3
"""
deploy/app.py
=============
AWS CDK application entry point.
Deploys the cost-effective SageMaker Async Inference + Lambda + EventBridge stack.
"""

import os
import aws_cdk as cdk

from stacks.storage_stack import StorageStack
from stacks.sagemaker_stack import SageMakerStack
from stacks.lambda_stack import LambdaStack

app = cdk.App()

env = cdk.Environment(
    account=os.environ.get("AWS_ACCOUNT_ID", os.environ.get("CDK_DEFAULT_ACCOUNT")),
    region=os.environ.get("AWS_REGION", os.environ.get("CDK_DEFAULT_REGION", "us-east-1")),
)

# ── Stack 1: S3 Buckets + IAM Roles ──────────────────────────
storage = StorageStack(app, "LlmStorageStack", env=env)

# ── Stack 2: SageMaker Async Endpoint ────────────────────────
sagemaker = SageMakerStack(
    app, "LlmSageMakerStack",
    model_bucket=storage.model_bucket,
    sagemaker_role=storage.sagemaker_role,
    env=env,
)
sagemaker.add_dependency(storage)

# ── Stack 3: Lambda + EventBridge Trigger ─────────────────────
lambda_stack = LambdaStack(
    app, "LlmLambdaStack",
    model_bucket=storage.model_bucket,
    sagemaker_endpoint_name=sagemaker.endpoint_name,
    env=env,
)
lambda_stack.add_dependency(sagemaker)

app.synth()
