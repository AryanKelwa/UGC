"""
deploy/stacks/trigger_stack.py
================================
CDK stack for the fine-tuning trigger system:

  1. Lambda function — checks MongoDB daily for new records (≥ 1000)
     and triggers the CodePipeline fine-tuning pipeline.
  2. EventBridge scheduled rule — fires daily at 2 AM AEST (UTC+10).
  3. IAM permissions — Lambda needs CodePipeline:StartPipelineExecution
     and Secrets Manager read access.

This Lambda can also be invoked manually at any time via:
  aws lambda invoke --function-name llm-finetune-trigger ...
"""

import os

import aws_cdk as cdk
from aws_cdk import (
    aws_lambda as lambda_,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct


class TriggerStack(cdk.Stack):
    """Fine-tune trigger Lambda + EventBridge daily schedule."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        pipeline_name: str = "LlmFineTunePipeline",
        threshold: int = 1000,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── MongoDB secret reference ──────────────────────────
        mongo_secret = secretsmanager.Secret.from_secret_name_v2(
            self, "MongoSecret",
            secret_name="llm/mongodb-url",
        )

        # ── Lambda Function ───────────────────────────────────
        trigger_fn = lambda_.Function(
            self, "FineTuneTriggerFn",
            function_name="llm-finetune-trigger",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset(
                # Dependencies (pymongo) are pre-installed into this folder via:
                #   pip install -r requirements.txt -t deploy/lambda_fn/fine_tune_trigger/
                # No Docker bundling needed — CDK just zips the folder as-is.
                "lambda_fn/fine_tune_trigger",
            ),
            timeout=cdk.Duration.minutes(5),
            memory_size=256,
            description="Checks MongoDB for 1000+ new records and triggers fine-tuning CodePipeline",
            environment={
                "THRESHOLD": str(threshold),
                "CODEPIPELINE_NAME": pipeline_name,
                "MONGODB_SECRET_NAME": "llm/mongodb-url",
                "MONGODB_DB_NAME": "realestate_data",
                "MONGODB_COLLECTION": "leads",
                "TRAINING_RUNS_COLLECTION": "training_runs",
            },
            tracing=lambda_.Tracing.ACTIVE,
        )

        # ── IAM Permissions ───────────────────────────────────

        # Read MongoDB connection string from Secrets Manager
        mongo_secret.grant_read(trigger_fn)

        # Trigger the CodePipeline fine-tuning pipeline
        trigger_fn.add_to_role_policy(
            iam.PolicyStatement(
                sid="StartFineTunePipeline",
                actions=["codepipeline:StartPipelineExecution"],
                resources=[
                    f"arn:aws:codepipeline:{self.region}:{self.account}:{pipeline_name}"
                ],
            )
        )

        # ── EventBridge Schedule: daily at 2 AM AEST (UTC 16:00 prev day) ──
        # AEST = UTC+10, so 2 AM AEST = 16:00 UTC previous day
        rule = events.Rule(
            self, "DailyTriggerRule",
            rule_name="llm-finetune-daily-trigger",
            # Cron: minute=0, hour=16 (UTC) = 2 AM AEST
            schedule=events.Schedule.cron(minute="0", hour="16"),
            description="Daily check: if 1000+ new records exist, trigger LLM fine-tuning",
        )
        rule.add_target(targets.LambdaFunction(
            trigger_fn,
            retry_attempts=2,
        ))

        # ── Outputs ───────────────────────────────────────────
        cdk.CfnOutput(
            self, "TriggerFunctionName",
            value=trigger_fn.function_name,
            description="Lambda function name for manual fine-tuning trigger",
            export_name="LlmFineTuneTriggerFnName",
        )

        cdk.CfnOutput(
            self, "ManualInvokeCommand",
            value=(
                f"aws lambda invoke --function-name {trigger_fn.function_name} "
                f"--region {self.region} /tmp/trigger_response.json"
            ),
            description="CLI command to manually trigger fine-tuning check",
        )
