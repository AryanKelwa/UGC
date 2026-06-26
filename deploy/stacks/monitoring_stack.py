"""
deploy/stacks/monitoring_stack.py
==================================
CDK stack for observability:

  • CloudWatch Dashboard — training job metrics, endpoint latency,
    Lambda invocation stats, pipeline stage durations
  • SNS Budget Alert — notifies when monthly AWS spend exceeds $50
  • CloudWatch Alarm — endpoint error rate alarm
"""

import aws_cdk as cdk
from aws_cdk import (
    aws_cloudwatch as cw,
    aws_cloudwatch_actions as cw_actions,
    aws_sns as sns,
    aws_sns_subscriptions as sns_subs,
    aws_budgets as budgets,
)
from constructs import Construct


class MonitoringStack(cdk.Stack):
    """CloudWatch dashboard + budget alarm for the LLM pipeline."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        reviewer_email: str = "",
        budget_limit_usd: float = 50.0,
        endpoint_name: str = "llama-finetune-async",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── Alerts SNS topic ──────────────────────────────────
        alert_topic = sns.Topic(
            self, "AlertTopic",
            topic_name="llm-pipeline-alerts",
            display_name="LLM Pipeline Alerts",
        )
        if reviewer_email:
            alert_topic.add_subscription(
                sns_subs.EmailSubscription(reviewer_email)
            )

        # ── CloudWatch Metrics ────────────────────────────────

        # Endpoint: invocation errors
        endpoint_errors = cw.Metric(
            namespace="AWS/SageMaker",
            metric_name="Invocation5XXErrors",
            dimensions_map={"EndpointName": endpoint_name, "VariantName": "AllTraffic"},
            statistic="Sum",
            period=cdk.Duration.minutes(15),
        )

        # Endpoint: model latency (p99)
        endpoint_latency = cw.Metric(
            namespace="AWS/SageMaker",
            metric_name="ModelLatency",
            dimensions_map={"EndpointName": endpoint_name, "VariantName": "AllTraffic"},
            statistic="p99",
            period=cdk.Duration.minutes(15),
        )

        # Lambda: inference trigger errors
        lambda_errors = cw.Metric(
            namespace="AWS/Lambda",
            metric_name="Errors",
            dimensions_map={"FunctionName": "llm-inference-trigger"},
            statistic="Sum",
            period=cdk.Duration.minutes(15),
        )

        # Lambda: fine-tune trigger invocations
        trigger_invocations = cw.Metric(
            namespace="AWS/Lambda",
            metric_name="Invocations",
            dimensions_map={"FunctionName": "llm-finetune-trigger"},
            statistic="Sum",
            period=cdk.Duration.days(1),
        )

        # ── CloudWatch Alarms ─────────────────────────────────

        # Alarm: endpoint error spike
        endpoint_error_alarm = cw.Alarm(
            self, "EndpointErrorAlarm",
            alarm_name="llm-endpoint-errors",
            alarm_description="SageMaker async endpoint returned 5XX errors",
            metric=endpoint_errors,
            threshold=5,
            evaluation_periods=2,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        )
        endpoint_error_alarm.add_alarm_action(cw_actions.SnsAction(alert_topic))

        # Alarm: inference Lambda failures
        lambda_error_alarm = cw.Alarm(
            self, "LambdaErrorAlarm",
            alarm_name="llm-lambda-errors",
            alarm_description="Inference trigger Lambda is failing",
            metric=lambda_errors,
            threshold=3,
            evaluation_periods=2,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        )
        lambda_error_alarm.add_alarm_action(cw_actions.SnsAction(alert_topic))

        # ── CloudWatch Dashboard ──────────────────────────────
        dashboard = cw.Dashboard(
            self, "LlmDashboard",
            dashboard_name="LlmFineTuningPipeline",
            default_interval=cdk.Duration.days(7),
        )

        dashboard.add_widgets(
            # Row 1: Endpoint health
            cw.TextWidget(
                markdown="## 🤖 SageMaker Inference Endpoint",
                width=24, height=1,
            ),
            cw.GraphWidget(
                title="Endpoint Invocations",
                left=[cw.Metric(
                    namespace="AWS/SageMaker",
                    metric_name="InvocationsProcessed",
                    dimensions_map={"EndpointName": endpoint_name, "VariantName": "AllTraffic"},
                    statistic="Sum",
                    period=cdk.Duration.hours(1),
                    color=cw.Color.GREEN,
                )],
                width=8, height=6,
            ),
            cw.GraphWidget(
                title="Model Latency (p50 / p99)",
                left=[
                    cw.Metric(
                        namespace="AWS/SageMaker",
                        metric_name="ModelLatency",
                        dimensions_map={"EndpointName": endpoint_name, "VariantName": "AllTraffic"},
                        statistic="p50",
                        period=cdk.Duration.hours(1),
                        label="p50",
                        color=cw.Color.BLUE,
                    ),
                    cw.Metric(
                        namespace="AWS/SageMaker",
                        metric_name="ModelLatency",
                        dimensions_map={"EndpointName": endpoint_name, "VariantName": "AllTraffic"},
                        statistic="p99",
                        period=cdk.Duration.hours(1),
                        label="p99",
                        color=cw.Color.ORANGE,
                    ),
                ],
                width=8, height=6,
            ),
            cw.GraphWidget(
                title="Endpoint 5XX Errors",
                left=[endpoint_errors],
                width=8, height=6,
            ),
        )

        dashboard.add_widgets(
            # Row 2: Lambda health
            cw.TextWidget(
                markdown="## ⚡ Lambda Functions",
                width=24, height=1,
            ),
            cw.GraphWidget(
                title="Inference Trigger Lambda — Invocations & Errors",
                left=[cw.Metric(
                    namespace="AWS/Lambda",
                    metric_name="Invocations",
                    dimensions_map={"FunctionName": "llm-inference-trigger"},
                    statistic="Sum",
                    period=cdk.Duration.hours(1),
                    label="Invocations",
                    color=cw.Color.GREEN,
                )],
                right=[lambda_errors],
                width=12, height=6,
            ),
            cw.GraphWidget(
                title="Fine-Tune Trigger Lambda — Daily Invocations",
                left=[trigger_invocations],
                width=12, height=6,
            ),
        )

        dashboard.add_widgets(
            # Row 3: Training jobs
            cw.TextWidget(
                markdown="## 🏋️ SageMaker Training Jobs",
                width=24, height=1,
            ),
            cw.GraphWidget(
                title="Training Job Instance CPU Utilization",
                left=[cw.Metric(
                    namespace="/aws/sagemaker/TrainingJobs",
                    metric_name="CPUUtilization",
                    statistic="Average",
                    period=cdk.Duration.minutes(5),
                )],
                width=12, height=6,
            ),
            cw.GraphWidget(
                title="Training Job GPU Utilization",
                left=[cw.Metric(
                    namespace="/aws/sagemaker/TrainingJobs",
                    metric_name="GPUUtilization",
                    statistic="Average",
                    period=cdk.Duration.minutes(5),
                )],
                width=12, height=6,
            ),
        )

        # ── AWS Budget Alert ──────────────────────────────────
        # Alerts when estimated monthly spend exceeds $50
        budgets.CfnBudget(
            self, "MonthlyCostBudget",
            budget=budgets.CfnBudget.BudgetDataProperty(
                budget_name="LlmPipelineMonthlyBudget",
                budget_type="COST",
                time_unit="MONTHLY",
                budget_limit=budgets.CfnBudget.SpendProperty(
                    amount=budget_limit_usd,
                    unit="USD",
                ),
            ),
            notifications_with_subscribers=[
                budgets.CfnBudget.NotificationWithSubscribersProperty(
                    notification=budgets.CfnBudget.NotificationProperty(
                        notification_type="ACTUAL",
                        comparison_operator="GREATER_THAN",
                        threshold=80,  # Alert at 80% of budget
                        threshold_type="PERCENTAGE",
                    ),
                    subscribers=[
                        budgets.CfnBudget.SubscriberProperty(
                            subscription_type="SNS",
                            address=alert_topic.topic_arn,
                        ),
                    ] + ([
                        budgets.CfnBudget.SubscriberProperty(
                            subscription_type="EMAIL",
                            address=reviewer_email,
                        ),
                    ] if reviewer_email else []),
                ),
            ],
        )

        # ── Outputs ───────────────────────────────────────────
        cdk.CfnOutput(
            self, "DashboardUrl",
            value=(
                f"https://{self.region}.console.aws.amazon.com/cloudwatch/home"
                f"?region={self.region}#dashboards:name=LlmFineTuningPipeline"
            ),
            description="CloudWatch dashboard URL",
        )

        cdk.CfnOutput(
            self, "AlertTopicArn",
            value=alert_topic.topic_arn,
            description="SNS topic ARN for pipeline alerts",
        )
