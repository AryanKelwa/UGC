"""
deploy/stacks/sagemaker_stack.py
=================================
CDK stack for the SageMaker Async Inference endpoint.
Uses a HuggingFace Deep Learning Container with the fine-tuned model
weights from S3. Configured with auto-scaling to zero for cost savings.
"""

import aws_cdk as cdk
from aws_cdk import (
    aws_sagemaker as sagemaker,
    aws_applicationautoscaling as appscaling,
    aws_iam as iam,
    aws_s3 as s3,
)
from constructs import Construct


class SageMakerStack(cdk.Stack):
    """SageMaker Model + Async Endpoint with scale-to-zero auto-scaling."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        model_bucket: s3.IBucket,
        sagemaker_role: iam.IRole,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── Configuration (from CDK context or defaults) ──────
        instance_type = self.node.try_get_context("instance_type") or "ml.g4dn.xlarge"
        endpoint_name = self.node.try_get_context("endpoint_name") or "llama-finetune-async"
        model_s3_key = self.node.try_get_context("model_s3_key") or "llama-finetune/model.tar.gz"

        model_data_url = f"s3://{model_bucket.bucket_name}/{model_s3_key}"

        # ── HuggingFace DLC Image URI ─────────────────────────
        # Using the HuggingFace inference container for text-generation
        image_uri = (
            f"763104351884.dkr.ecr.{self.region}.amazonaws.com/"
            f"huggingface-pytorch-inference:2.1.0-transformers4.43.0-gpu-py310-cu121-ubuntu22.04"
        )

        # ── SageMaker Model ───────────────────────────────────
        model = sagemaker.CfnModel(
            self, "LlmModel",
            execution_role_arn=sagemaker_role.role_arn,
            model_name=f"{endpoint_name}-model",
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image=image_uri,
                model_data_url=model_data_url,
                environment={
                    "HF_TASK": "text-generation",
                    "SM_NUM_GPUS": "1",
                    "MAX_INPUT_LENGTH": "2048",
                    "MAX_TOTAL_TOKENS": "4096",
                },
            ),
        )

        # ── Async Inference Config ────────────────────────────
        endpoint_config = sagemaker.CfnEndpointConfig(
            self, "LlmEndpointConfig",
            endpoint_config_name=f"{endpoint_name}-config",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    variant_name="AllTraffic",
                    model_name=model.attr_model_name,
                    initial_instance_count=1,
                    instance_type=instance_type,
                    initial_variant_weight=1.0,
                ),
            ],
            async_inference_config=sagemaker.CfnEndpointConfig.AsyncInferenceConfigProperty(
                output_config=sagemaker.CfnEndpointConfig.AsyncInferenceOutputConfigProperty(
                    s3_output_path=f"s3://{model_bucket.bucket_name}/inference-outputs/",
                ),
                client_config=sagemaker.CfnEndpointConfig.AsyncInferenceClientConfigProperty(
                    max_concurrent_invocations_per_instance=4,
                ),
            ),
        )
        endpoint_config.add_dependency(model)

        # ── SageMaker Endpoint ────────────────────────────────
        endpoint = sagemaker.CfnEndpoint(
            self, "LlmEndpoint",
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_config.attr_endpoint_config_name,
        )
        endpoint.add_dependency(endpoint_config)

        # ── Auto-Scaling to Zero ──────────────────────────────
        # Register scalable target on the variant
        scalable_target = appscaling.ScalableTarget(
            self, "LlmScalableTarget",
            service_namespace=appscaling.ServiceNamespace.SAGEMAKER,
            resource_id=f"endpoint/{endpoint_name}/variant/AllTraffic",
            scalable_dimension="sagemaker:variant:DesiredInstanceCount",
            min_capacity=0,
            max_capacity=1,
        )
        scalable_target.node.add_dependency(endpoint)

        # Scale based on backlog (ApproximateBacklogSizePerInstance)
        scalable_target.scale_to_track_metric(
            "BacklogTracking",
            target_value=2.0,
            custom_metric=appscaling.MetricAggregationType.AVERAGE,
            predefined_metric=appscaling.PredefinedMetric.SAGEMAKER_VARIANT_INVOCATIONS_PER_INSTANCE,
            scale_in_cooldown=cdk.Duration.seconds(300),
            scale_out_cooldown=cdk.Duration.seconds(60),
        )

        # ── Expose endpoint name ──────────────────────────────
        self.endpoint_name = endpoint_name

        # ── Outputs ───────────────────────────────────────────
        cdk.CfnOutput(
            self, "EndpointName",
            value=endpoint_name,
            description="SageMaker async inference endpoint name",
            export_name="LlmEndpointName",
        )

        cdk.CfnOutput(
            self, "ModelDataUrl",
            value=model_data_url,
            description="S3 URI for model.tar.gz",
        )
