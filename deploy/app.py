#!/usr/bin/env python3
"""
deploy/app.py
=============
AWS CDK application entry point.
Region: ap-southeast-2 (Sydney)

Stacks deployed (in dependency order):
  1. LlmStorageStack       — S3 bucket + SageMaker IAM role
  2. LlmSageMakerStack     — SageMaker Async Endpoint (scale 0→1)
  3. LlmLambdaStack        — Inference poller Lambda + EventBridge 15-min schedule
  4. LlmPipelineStack      — 7-stage CodePipeline (Build→Train→Evaluate→Approve→Deploy)
  5. LlmTriggerStack       — Fine-tune trigger Lambda (daily, 1000-record threshold)
  6. LlmMonitoringStack    — CloudWatch dashboard + $50/month budget alarm

CDK Context variables (set via cdk.json or --context flags):
  reviewer_email        — email address to receive approval/alert notifications
  github_owner          — GitHub organisation or username (for source stage)
  github_repo           — GitHub repo name (default: llm-finetuning)
  github_branch         — Branch to track (default: main)
  github_connection_arn — AWS CodeStar Connections ARN for GitHub OAuth
  budget_limit_usd      — Monthly spend limit before alarm fires (default: 50)

Usage:
  cdk deploy --all --context reviewer_email=you@example.com
  cdk deploy LlmStorageStack LlmSageMakerStack  (deploy subset)
"""

import os
import aws_cdk as cdk

from stacks.storage_stack import StorageStack
from stacks.sagemaker_stack import SageMakerStack
from stacks.lambda_stack import LambdaStack
from stacks.pipeline_stack import PipelineStack
from stacks.trigger_stack import TriggerStack
from stacks.monitoring_stack import MonitoringStack

app = cdk.App()

# ── Environment ───────────────────────────────────────────────
# Primary region is ap-southeast-2 to match the existing S3 bucket.
env = cdk.Environment(
    account=os.environ.get("AWS_ACCOUNT_ID", os.environ.get("CDK_DEFAULT_ACCOUNT")),
    region=os.environ.get("AWS_REGION", os.environ.get("CDK_DEFAULT_REGION", "ap-southeast-2")),
)

# ── Context / config ──────────────────────────────────────────
reviewer_email = (
    app.node.try_get_context("reviewer_email")
    or os.environ.get("REVIEWER_EMAIL", "")
)
github_owner = (
    app.node.try_get_context("github_owner")
    or os.environ.get("GITHUB_OWNER", "")
)
github_repo = app.node.try_get_context("github_repo") or "llm-finetuning"
github_branch = app.node.try_get_context("github_branch") or "main"
budget_limit = float(
    app.node.try_get_context("budget_limit_usd")
    or os.environ.get("BUDGET_LIMIT_USD", "50")
)

# ─────────────────────────────────────────────────────────────
# Stack 1: Storage — S3 bucket + SageMaker IAM role
# ─────────────────────────────────────────────────────────────
storage = StorageStack(app, "LlmStorageStack", env=env)

# ─────────────────────────────────────────────────────────────
# Stack 2: SageMaker Async Inference Endpoint
#   • ml.g4dn.xlarge (T4 GPU, 16 GB VRAM)
#   • Scale-to-zero auto-scaling (min=0, max=1)
# ─────────────────────────────────────────────────────────────
sagemaker = SageMakerStack(
    app, "LlmSageMakerStack",
    model_bucket=storage.model_bucket,
    sagemaker_role=storage.sagemaker_role,
    env=env,
)
sagemaker.add_dependency(storage)

# ─────────────────────────────────────────────────────────────
# Stack 3: Inference Loop
#   • Lambda polls MongoDB every 15 min → SageMaker Async
#   • Result Processor Lambda reads S3 output → updates MongoDB
# ─────────────────────────────────────────────────────────────
lambda_stack = LambdaStack(
    app, "LlmLambdaStack",
    model_bucket=storage.model_bucket,
    sagemaker_endpoint_name=sagemaker.endpoint_name,
    env=env,
)
lambda_stack.add_dependency(sagemaker)

# ─────────────────────────────────────────────────────────────
# Stack 4: Continuous Fine-Tuning Pipeline
#   • 7-stage CodePipeline with human approval gate
#   • Stages: Source → Build → DataPrep → Train → Evaluate → Approve → Deploy
# ─────────────────────────────────────────────────────────────
pipeline_stack = PipelineStack(
    app, "LlmPipelineStack",
    model_bucket=storage.model_bucket,
    sagemaker_role=storage.sagemaker_role,
    reviewer_email=reviewer_email,
    github_owner=github_owner,
    github_repo=github_repo,
    github_branch=github_branch,
    env=env,
)
pipeline_stack.add_dependency(storage)

# ─────────────────────────────────────────────────────────────
# Stack 5: Fine-Tune Trigger
#   • Lambda checks MongoDB daily at 2 AM AEST
#   • Triggers CodePipeline when ≥ 1000 new processed records exist
#   • Can also be invoked manually at any time
# ─────────────────────────────────────────────────────────────
trigger_stack = TriggerStack(
    app, "LlmTriggerStack",
    pipeline_name="LlmFineTunePipeline",
    threshold=1000,
    env=env,
)
trigger_stack.add_dependency(pipeline_stack)

# ─────────────────────────────────────────────────────────────
# Stack 6: Monitoring & Observability
#   • CloudWatch dashboard (endpoint, Lambda, training metrics)
#   • Alarms: endpoint errors, Lambda failures
#   • AWS Budget alert at 80% of $50/month
# ─────────────────────────────────────────────────────────────
monitoring_stack = MonitoringStack(
    app, "LlmMonitoringStack",
    reviewer_email=reviewer_email,
    budget_limit_usd=budget_limit,
    endpoint_name=sagemaker.endpoint_name,
    env=env,
)
monitoring_stack.add_dependency(sagemaker)
monitoring_stack.add_dependency(lambda_stack)

app.synth()

