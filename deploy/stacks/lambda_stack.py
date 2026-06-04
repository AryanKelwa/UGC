"""
deploy/stacks/lambda_stack.py
==============================
CDK stack for the Lambda function that polls MongoDB for pending leads
and triggers SageMaker async inference. Includes an EventBridge
scheduled rule (default: every 15 minutes).
"""

import aws_cdk as cdk
from aws_cdk import (
    aws_lambda as lambda_,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_s3 as s3,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct


class LambdaStack(cdk.Stack):
    """Lambda function + EventBridge scheduled trigger for async inference."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        model_bucket: s3.IBucket,
        sagemaker_endpoint_name: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── Configuration ─────────────────────────────────────
        schedule_expression = (
            self.node.try_get_context("schedule_expression")
            or "rate(15 minutes)"
        )

        # ── MongoDB connection secret ─────────────────────────
        # Store your MongoDB connection string in Secrets Manager
        # under the key "llm/mongodb-url"
        mongo_secret = secretsmanager.Secret.from_secret_name_v2(
            self, "MongoSecret",
            secret_name="llm/mongodb-url",
        )

        # ── Lambda Function ───────────────────────────────────
        inference_fn = lambda_.Function(
            self, "InferenceTriggerFn",
            function_name="llm-inference-trigger",
            runtime=lambda_.Runtime.PYTHON_3_10,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset("lambda_fn"),
            timeout=cdk.Duration.minutes(5),
            memory_size=256,
            environment={
                "SAGEMAKER_ENDPOINT_NAME": sagemaker_endpoint_name,
                "S3_BUCKET": model_bucket.bucket_name,
                "S3_INPUT_PREFIX": "inference-inputs/",
                "MONGODB_SECRET_NAME": "llm/mongodb-url",
                "MONGODB_DB_NAME": "realestate_data",
                "MONGODB_COLLECTION": "leads",
            },
            tracing=lambda_.Tracing.ACTIVE,
        )

        # ── IAM Permissions ───────────────────────────────────

        # S3: read/write for inference input/output
        model_bucket.grant_read_write(inference_fn)

        # SageMaker: invoke async endpoint
        inference_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "sagemaker:InvokeEndpointAsync",
                    "sagemaker:InvokeEndpoint",
                ],
                resources=[
                    f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint/{sagemaker_endpoint_name}",
                ],
            )
        )

        # Secrets Manager: read MongoDB connection string
        mongo_secret.grant_read(inference_fn)

        # ── EventBridge Scheduled Rule ────────────────────────
        rule = events.Rule(
            self, "InferenceScheduleRule",
            rule_name="llm-inference-schedule",
            schedule=events.Schedule.expression(schedule_expression),
            description="Triggers Lambda to poll MongoDB and invoke SageMaker async inference",
        )
        rule.add_target(targets.LambdaFunction(inference_fn))

        # ── Outputs ───────────────────────────────────────────
        cdk.CfnOutput(
            self, "LambdaFunctionName",
            value=inference_fn.function_name,
            description="Lambda function name for inference trigger",
        )

        cdk.CfnOutput(
            self, "ScheduleRule",
            value=schedule_expression,
            description="EventBridge schedule expression",
        )
