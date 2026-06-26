"""
deploy/lambda_fn/fine_tune_trigger/handler.py
==============================================
AWS Lambda function that checks if enough new data has accumulated
in MongoDB to justify a fine-tuning run, and if so, triggers the
CodePipeline fine-tuning pipeline.

Trigger: EventBridge scheduled rule — runs daily at 2 AM AEST.
Can also be invoked manually at any time via Lambda console/CLI.

Logic:
  1. Connect to MongoDB (connection string from Secrets Manager)
  2. Count records with etl_status="processed" AND no training_run_id
  3. If count >= THRESHOLD (default 1000):
     a. Start CodePipeline execution
     b. Mark records as "queued_for_training" in MongoDB
     c. Create a training_run document in MongoDB
  4. Else: log the current count and exit cleanly

Environment Variables (set by CDK):
  THRESHOLD            — minimum new records to trigger (default: 1000)
  CODEPIPELINE_NAME    — "LlmFineTunePipeline"
  MONGODB_SECRET_NAME  — Secrets Manager key for MongoDB connection string
  MONGODB_DB_NAME      — "realestate_data"
  MONGODB_COLLECTION   — "leads"
  TRAINING_RUNS_COLLECTION — "training_runs"
  AWS_REGION_NAME      — ap-southeast-2 (note: Lambda provides AWS_REGION)
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── AWS clients (module-level for warm reuse) ─────────────────
secrets_client = boto3.client("secretsmanager")
codepipeline_client = boto3.client("codepipeline")

# ── Cached MongoDB client ─────────────────────────────────────
_mongo_client = None

THRESHOLD = int(os.environ.get("THRESHOLD", "1000"))
PIPELINE_NAME = os.environ.get("CODEPIPELINE_NAME", "LlmFineTunePipeline")
MONGODB_SECRET_NAME = os.environ.get("MONGODB_SECRET_NAME", "llm/mongodb-url")
DB_NAME = os.environ.get("MONGODB_DB_NAME", "realestate_data")
LEADS_COLLECTION = os.environ.get("MONGODB_COLLECTION", "leads")
RUNS_COLLECTION = os.environ.get("TRAINING_RUNS_COLLECTION", "training_runs")


def _get_mongo_db():
    """Lazily initialise and cache the MongoDB connection."""
    global _mongo_client
    if _mongo_client is None:
        try:
            from pymongo import MongoClient
        except ImportError:
            raise RuntimeError(
                "pymongo is not installed. Add it to lambda_fn/fine_tune_trigger/requirements.txt."
            )
        secret = secrets_client.get_secret_value(SecretId=MONGODB_SECRET_NAME)
        mongo_url = secret["SecretString"]
        _mongo_client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
    return _mongo_client[DB_NAME]


def _count_eligible_records(db) -> int:
    """Count processed records that have not yet been assigned to a training run."""
    return db[LEADS_COLLECTION].count_documents({
        "etl_status": "processed",
        "training_run_id": {"$exists": False},
    })


def _trigger_pipeline() -> str:
    """Start a new CodePipeline execution and return the execution ID."""
    response = codepipeline_client.start_pipeline_execution(name=PIPELINE_NAME)
    return response["pipelineExecutionId"]


def _mark_records_queued(db, run_id: str) -> int:
    """
    Mark all eligible records as queued_for_training and assign the run_id.
    Returns the number of records updated.
    """
    result = db[LEADS_COLLECTION].update_many(
        {
            "etl_status": "processed",
            "training_run_id": {"$exists": False},
        },
        {
            "$set": {
                "etl_status": "queued_for_training",
                "training_run_id": run_id,
                "queued_at": datetime.now(timezone.utc),
            }
        },
    )
    return result.modified_count


def _create_training_run_doc(db, run_id: str, record_count: int, pipeline_execution_id: str) -> None:
    """Insert a training_run document to track this fine-tuning run."""
    db[RUNS_COLLECTION].insert_one({
        "_id": run_id,
        "status": "triggered",
        "pipeline_execution_id": pipeline_execution_id,
        "record_count": record_count,
        "triggered_at": datetime.now(timezone.utc),
        "started_at": None,
        "completed_at": None,
        "sm_training_job_name": None,
        "s3_model_path": None,
        "eval_metrics": None,
        "approval_status": None,
        "deployed": False,
        "deployed_at": None,
        "endpoint_config_name": None,
    })
    logger.info("Created training_run document: %s", run_id)


def lambda_handler(event, context):
    """
    Main entry point.
    Can be triggered by EventBridge (scheduled) or invoked manually.
    """
    logger.info("Fine-tune trigger Lambda invoked. Threshold=%d, Pipeline=%s",
                THRESHOLD, PIPELINE_NAME)

    # ── 1. Connect to MongoDB ─────────────────────────────────
    try:
        db = _get_mongo_db()
    except Exception as exc:
        logger.error("MongoDB connection failed: %s", exc)
        return {"status": "error", "message": str(exc)}

    # ── 2. Count eligible records ─────────────────────────────
    try:
        count = _count_eligible_records(db)
        logger.info("Eligible records: %d / %d required", count, THRESHOLD)
    except Exception as exc:
        logger.error("Failed to count MongoDB records: %s", exc)
        return {"status": "error", "message": str(exc)}

    # ── 3. Check threshold ────────────────────────────────────
    if count < THRESHOLD:
        logger.info(
            "Not enough new records to trigger fine-tuning. "
            "Have %d, need %d. Exiting.",
            count, THRESHOLD
        )
        return {
            "status": "skipped",
            "reason": "below_threshold",
            "eligible_records": count,
            "threshold": THRESHOLD,
        }

    logger.info(
        "Threshold met: %d records >= %d. Triggering CodePipeline.",
        count, THRESHOLD
    )

    # ── 4. Generate run ID ────────────────────────────────────
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uid = uuid.uuid4().hex[:6]
    run_id = f"run_{timestamp}_{short_uid}"

    # ── 5. Trigger CodePipeline ───────────────────────────────
    try:
        pipeline_execution_id = _trigger_pipeline()
        logger.info("CodePipeline triggered. Execution ID: %s", pipeline_execution_id)
    except Exception as exc:
        logger.error("Failed to start CodePipeline: %s", exc)
        return {"status": "error", "message": f"CodePipeline trigger failed: {exc}"}

    # ── 6. Mark records as queued ─────────────────────────────
    try:
        updated = _mark_records_queued(db, run_id)
        logger.info("Marked %d records as queued_for_training with run_id=%s", updated, run_id)
    except Exception as exc:
        logger.error("Failed to update lead records: %s", exc)
        # Pipeline was already triggered — log but don't fail the Lambda
        # The next run will handle the untagged records gracefully

    # ── 7. Create training_run document ──────────────────────
    try:
        _create_training_run_doc(db, run_id, count, pipeline_execution_id)
    except Exception as exc:
        logger.error("Failed to create training_run document: %s", exc)

    result = {
        "status": "triggered",
        "run_id": run_id,
        "pipeline_execution_id": pipeline_execution_id,
        "eligible_records": count,
        "threshold": THRESHOLD,
    }
    logger.info("Fine-tune trigger completed: %s", json.dumps(result))
    return result
