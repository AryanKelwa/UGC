"""
deploy/stacks/pipeline_stack.py
================================
CDK stack for the continuous fine-tuning CodePipeline.

Stages:
  1. SOURCE       — GitHub / CodeCommit source (main branch)
  2. BUILD        — Docker build → ECR push (existing buildspec.yml)
  3. DATA_PREP    — Export MongoDB splits → S3 (buildspec_data_prep.yml)
  4. TRAIN        — SageMaker Training Job (buildspec_train.yml)
  5. EVALUATE     — Eval + Model Registry registration (buildspec_evaluate.yml)
  6. HUMAN_APPROVAL — Manual review gate (SNS email → Approve/Reject)
  7. DEPLOY       — Blue-green endpoint update (buildspec_deploy.yml)

Environment variables passed to all CodeBuild stages:
  AWS_S3_BUCKET, AWS_REGION, AWS_ACCOUNT_ID,
  AWS_SAGEMAKER_ROLE_ARN, SNS_APPROVAL_TOPIC_ARN
"""

import aws_cdk as cdk
from aws_cdk import (
    aws_codepipeline as codepipeline,
    aws_codepipeline_actions as pipeline_actions,
    aws_codebuild as codebuild,
    aws_iam as iam,
    aws_s3 as s3,
    aws_sns as sns,
    aws_sns_subscriptions as sns_subs,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct


class PipelineStack(cdk.Stack):
    """
    7-stage CodePipeline for continuous LLM fine-tuning with human approval gate.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        model_bucket: s3.IBucket,
        sagemaker_role: iam.IRole,
        reviewer_email: str,
        github_owner: str = "",
        github_repo: str = "llm-finetuning",
        github_branch: str = "main",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── SNS Topic for approval notifications ──────────────
        approval_topic = sns.Topic(
            self, "ApprovalTopic",
            topic_name="llm-model-approval-notifications",
            display_name="LLM Model Review Notifications",
        )
        if reviewer_email:
            approval_topic.add_subscription(
                sns_subs.EmailSubscription(reviewer_email)
            )

        # ── Shared CodeBuild environment variables ─────────────
        shared_env_vars = {
            "AWS_S3_BUCKET": codebuild.BuildEnvironmentVariable(
                value=model_bucket.bucket_name
            ),
            "AWS_REGION": codebuild.BuildEnvironmentVariable(
                value=self.region
            ),
            "AWS_ACCOUNT_ID": codebuild.BuildEnvironmentVariable(
                value=self.account
            ),
            "AWS_SAGEMAKER_ROLE_ARN": codebuild.BuildEnvironmentVariable(
                value=sagemaker_role.role_arn
            ),
            "SNS_APPROVAL_TOPIC_ARN": codebuild.BuildEnvironmentVariable(
                value=approval_topic.topic_arn
            ),
            "ENDPOINT_NAME": codebuild.BuildEnvironmentVariable(
                value="llama-finetune-async"
            ),
        }

        # MongoDB URL from Secrets Manager (needed by DATA_PREP and EVALUATE)
        mongo_secret = secretsmanager.Secret.from_secret_name_v2(
            self, "MongoSecret", secret_name="llm/mongodb-url"
        )
        shared_env_vars["MONGODB_URL"] = codebuild.BuildEnvironmentVariable(
            value=mongo_secret.secret_value.unsafe_unwrap(),
            type=codebuild.BuildEnvironmentVariableType.PLAINTEXT,
        )

        # ── Common CodeBuild role ─────────────────────────────
        cb_role = iam.Role(
            self, "CodeBuildRole",
            assumed_by=iam.ServicePrincipal("codebuild.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AWSCodeBuildAdminAccess"),
            ],
        )
        model_bucket.grant_read_write(cb_role)
        approval_topic.grant_publish(cb_role)
        mongo_secret.grant_read(cb_role)
        sagemaker_role.grant_pass_role(cb_role)

        # Allow CodeBuild to pull/push ECR images
        cb_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
                "ecr:PutImage",
                "ecr:InitiateLayerUpload",
                "ecr:UploadLayerPart",
                "ecr:CompleteLayerUpload",
            ],
            resources=["*"],
        ))

        # Allow CodeBuild to register/update SM Model Registry
        cb_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "sagemaker:CreateModelPackageGroup",
                "sagemaker:CreateModelPackage",
                "sagemaker:UpdateModelPackage",
                "sagemaker:DescribeModelPackage",
                "sagemaker:ListModelPackages",
                "sagemaker:CreateModel",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:UpdateEndpoint",
                "sagemaker:DescribeEndpoint",
                "sagemaker:CreateTrainingJob",
                "sagemaker:DescribeTrainingJob",
            ],
            resources=["*"],
        ))

        # ── GPU-enabled CodeBuild for TRAIN stage ─────────────
        # Standard CodeBuild doesn't have GPUs — training is delegated to SM,
        # so we just need a standard compute class to run start_training_job.py
        standard_compute = codebuild.ComputeType.SMALL

        # ── S3 bucket for CodeBuild pip cache ─────────────────
        # Stores /root/.cache/pip across builds so pip install is fast after first run.
        # Must be defined BEFORE make_cb_project() which references it.
        cache_bucket = s3.Bucket(
            self, "CbCacheBucket",
            bucket_name=cdk.Fn.sub(
                "llm-codebuild-cache-${AWS::AccountId}-${AWS::Region}"
            ),
            removal_policy=cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="ExpireOldCache",
                    expiration=cdk.Duration.days(30),  # auto-purge stale cache monthly
                    enabled=True,
                )
            ],
        )
        cache_bucket.grant_read_write(cb_role)

        # ── Helper: create a CodeBuild project ────────────────
        def make_cb_project(name: str, buildspec_file: str) -> codebuild.Project:
            return codebuild.Project(
                self, name,
                project_name=f"llm-{name.lower()}",
                role=cb_role,
                environment=codebuild.BuildEnvironment(
                    build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                    compute_type=standard_compute,
                    privileged=(name == "Build"),  # Docker-in-Docker for BUILD stage only
                ),
                build_spec=codebuild.BuildSpec.from_source_filename(buildspec_file),
                environment_variables=shared_env_vars,
                timeout=cdk.Duration.hours(6),  # Allow enough time for training job polling
                # S3 pip cache: restores /root/.cache/pip before install phase.
                # BUILD stage uses Docker layer cache instead (ECR --cache-from).
                cache=codebuild.Cache.bucket(
                    cache_bucket,
                    prefix=f"pip-cache/{name.lower()}",
                ) if name != "Build" else codebuild.Cache.no_cache(),
                logging=codebuild.LoggingOptions(
                    cloud_watch=codebuild.CloudWatchLoggingOptions(
                        log_group_name=f"/llm-pipeline/{name.lower()}",
                        prefix="build",
                    )
                ),
            )

        # ── Create CodeBuild projects for each stage ──────────
        build_project = make_cb_project("Build", "buildspec.yml")
        data_prep_project = make_cb_project("DataPrep", "buildspec_data_prep.yml")
        train_project = make_cb_project("Train", "buildspec_train.yml")
        evaluate_project = make_cb_project("Evaluate", "buildspec_evaluate.yml")
        deploy_project = make_cb_project("Deploy", "buildspec_deploy.yml")


        # ── Pipeline artifact buckets ─────────────────────────
        artifact_bucket = s3.Bucket(
            self, "PipelineArtifactBucket",
            bucket_name=cdk.Fn.sub(
                "llm-pipeline-artifacts-${AWS::AccountId}-${AWS::Region}"
            ),
            removal_policy=cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
        )

        # ── Pipeline artifacts ────────────────────────────────
        source_output = codepipeline.Artifact("SourceCode")
        build_output = codepipeline.Artifact("BuildOutput")
        data_prep_output = codepipeline.Artifact("DataPrepOutput")
        train_output = codepipeline.Artifact("TrainOutput")
        evaluate_output = codepipeline.Artifact("EvaluateOutput")

        # ── Pipeline definition ───────────────────────────────
        pipeline = codepipeline.Pipeline(
            self, "FineTunePipeline",
            pipeline_name="LlmFineTunePipeline",
            artifact_bucket=artifact_bucket,
            restart_execution_on_update=False,
        )

        # ── Stage 1: SOURCE ───────────────────────────────────
        # Using GitHub connection (you must create a CodeStar connection in AWS Console first)
        # OR use CodeCommit as an alternative
        pipeline.add_stage(
            stage_name="SOURCE",
            actions=[
                pipeline_actions.CodeStarConnectionsSourceAction(
                    action_name="GitHubSource",
                    owner=github_owner or self.node.try_get_context("github_owner") or "your-github-org",
                    repo=github_repo,
                    branch=github_branch,
                    connection_arn=self.node.try_get_context("github_connection_arn") or "",
                    output=source_output,
                    trigger_on_push=False,  # Triggered by data threshold Lambda, not git push
                ),
            ],
        )

        # ── Stage 2: BUILD (Docker → ECR) ─────────────────────
        pipeline.add_stage(
            stage_name="BUILD",
            actions=[
                pipeline_actions.CodeBuildAction(
                    action_name="DockerBuildAndPush",
                    project=build_project,
                    input=source_output,
                    outputs=[build_output],
                ),
            ],
        )

        # ── Stage 3: DATA_PREP ────────────────────────────────
        pipeline.add_stage(
            stage_name="DATA_PREP",
            actions=[
                pipeline_actions.CodeBuildAction(
                    action_name="ExportMongoSplitsToS3",
                    project=data_prep_project,
                    input=source_output,
                    outputs=[data_prep_output],
                    environment_variables={
                        # RUN_ID is exported from this stage for downstream use
                    },
                ),
            ],
        )

        # ── Stage 4: TRAIN ────────────────────────────────────
        pipeline.add_stage(
            stage_name="TRAIN",
            actions=[
                pipeline_actions.CodeBuildAction(
                    action_name="SageMakerTrainingJob",
                    project=train_project,
                    input=source_output,
                    extra_inputs=[data_prep_output],
                    outputs=[train_output],
                ),
            ],
        )

        # ── Stage 5: EVALUATE_AND_REGISTER ───────────────────
        pipeline.add_stage(
            stage_name="EVALUATE_AND_REGISTER",
            actions=[
                pipeline_actions.CodeBuildAction(
                    action_name="EvaluateAndRegister",
                    project=evaluate_project,
                    input=source_output,
                    extra_inputs=[data_prep_output, train_output],
                    outputs=[evaluate_output],
                ),
            ],
        )

        # ── Stage 6: HUMAN_APPROVAL (Pipeline PAUSES here) ───
        pipeline.add_stage(
            stage_name="HUMAN_APPROVAL",
            actions=[
                pipeline_actions.ManualApprovalAction(
                    action_name="ModelReviewApproval",
                    notification_topic=approval_topic,
                    additional_information=(
                        "A new fine-tuned LLaMA 3.1 8B model has passed automated evaluation. "
                        "Please review the metrics in SageMaker Model Registry before approving. "
                        f"Registry: https://{self.region}.console.aws.amazon.com/sagemaker/home"
                        f"?region={self.region}#/model-registry/llm-finetuning-registry"
                    ),
                    # 7 days before auto-expire
                ),
            ],
        )

        # ── Stage 7: DEPLOY ───────────────────────────────────
        pipeline.add_stage(
            stage_name="DEPLOY",
            actions=[
                pipeline_actions.CodeBuildAction(
                    action_name="BlueGreenEndpointUpdate",
                    project=deploy_project,
                    input=source_output,
                    extra_inputs=[data_prep_output, evaluate_output],
                ),
            ],
        )

        # ── Expose pipeline for other stacks ──────────────────
        self.pipeline = pipeline
        self.approval_topic = approval_topic

        # ── Outputs ───────────────────────────────────────────
        cdk.CfnOutput(
            self, "PipelineName",
            value=pipeline.pipeline_name,
            description="CodePipeline name for fine-tuning",
            export_name="LlmFineTunePipelineName",
        )

        cdk.CfnOutput(
            self, "PipelineConsoleUrl",
            value=(
                f"https://{self.region}.console.aws.amazon.com/codesuite/codepipeline/pipelines"
                f"/{pipeline.pipeline_name}/view?region={self.region}"
            ),
            description="Direct link to the fine-tuning pipeline in AWS Console",
        )

        cdk.CfnOutput(
            self, "ApprovalTopicArn",
            value=approval_topic.topic_arn,
            description="SNS topic ARN for model review notifications",
            export_name="LlmApprovalTopicArn",
        )
