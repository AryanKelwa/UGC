"""
deploy/lambda_fn/handler.py
=============================
AWS Lambda handler that:
  1. Reads the MongoDB connection string from Secrets Manager
  2. Queries MongoDB for pending lead records (etl_status = "pending_inference")
  3. Uploads each payload as JSON to S3
  4. Invokes SageMaker Async Endpoint for each payload
  5. Updates the MongoDB record with the S3 output location

Environment Variables (set by CDK):
  - SAGEMAKER_ENDPOINT_NAME
  - S3_BUCKET
  - S3_INPUT_PREFIX
  - MONGODB_SECRET_NAME
  - MONGODB_DB_NAME
  - MONGODB_COLLECTION
"""

from __future__ import annotations

import json
import os
import uuid
import logging
from datetime import datetime, timezone

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── AWS clients ───────────────────────────────────────────────
s3_client = boto3.client("s3")
sm_runtime = boto3.client("sagemaker-runtime")
secrets_client = boto3.client("secretsmanager")

# ── Cached MongoDB client (reused across warm invocations) ────
_mongo_client = None


def _get_mongo_collection():
    """Lazily initialize and cache the MongoDB connection."""
    global _mongo_client
    if _mongo_client is None:
        try:
            from pymongo import MongoClient
        except ImportError:
            raise RuntimeError(
                "pymongo is not installed in the Lambda layer. "
                "Add it to lambda_fn/requirements.txt and rebuild."
            )

        secret_name = os.environ["MONGODB_SECRET_NAME"]
        response = secrets_client.get_secret_value(SecretId=secret_name)
        mongo_url = response["SecretString"]
        _mongo_client = MongoClient(mongo_url)

    db_name = os.environ.get("MONGODB_DB_NAME", "realestate_data")
    collection_name = os.environ.get("MONGODB_COLLECTION", "leads")
    return _mongo_client[db_name][collection_name]


def _upload_payload_to_s3(payload: dict, request_id: str) -> str:
    """Upload a JSON inference payload to S3 and return the S3 URI."""
    bucket = os.environ["S3_BUCKET"]
    prefix = os.environ.get("S3_INPUT_PREFIX", "inference-inputs/")
    key = f"{prefix}{request_id}.json"

    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )

    return f"s3://{bucket}/{key}"


def lambda_handler(event, context):
    """
    Main Lambda entry point.
    Triggered by EventBridge schedule — polls MongoDB for pending leads
    and dispatches them to SageMaker async inference.
    """
    endpoint_name = os.environ["SAGEMAKER_ENDPOINT_NAME"]

    logger.info("Lambda triggered — querying MongoDB for pending records …")

    try:
        collection = _get_mongo_collection()
    except Exception as exc:
        logger.error("Failed to connect to MongoDB: %s", exc)
        return {"status": "error", "message": str(exc)}

    # ── Query for pending records ─────────────────────────────
    pending_records = list(
        collection.find(
            {"inference_status": "pending"},
            limit=50,  # process in batches of 50
        )
    )

    if not pending_records:
        logger.info("No pending records found.")
        return {"status": "no_pending_data", "processed": 0}

    logger.info("Found %d pending records.", len(pending_records))

    triggered = 0
    errors = 0

    for record in pending_records:
        record_id = str(record.get("_id", "unknown"))
        request_id = f"{record_id}-{uuid.uuid4().hex[:8]}"

        try:
            # ── Build inference payload ───────────────────────
            conversation_text = record.get("conversation", "")
            if not conversation_text:
                logger.warning("Record %s has no conversation text — skipping.", record_id)
                continue

            payload = {
                "inputs": conversation_text,
                "parameters": {
                    "max_new_tokens": 512,
                    "temperature": 0.1,
                    "do_sample": True,
                },
            }

            # ── Upload payload to S3 ─────────────────────────
            input_s3_uri = _upload_payload_to_s3(payload, request_id)

            # ── Invoke SageMaker Async ────────────────────────
            response = sm_runtime.invoke_endpoint_async(
                EndpointName=endpoint_name,
                InputLocation=input_s3_uri,
                ContentType="application/json",
                Accept="application/json",
            )

            output_location = response.get("OutputLocation", "")
            logger.info(
                "Record %s → async invocation sent. Output: %s",
                record_id, output_location,
            )

            # ── Update DB record ──────────────────────────────
            collection.update_one(
                {"_id": record["_id"]},
                {
                    "$set": {
                        "inference_status": "processing",
                        "sagemaker_output_s3": output_location,
                        "sagemaker_request_id": request_id,
                        "inference_triggered_at": datetime.now(timezone.utc),
                    }
                },
            )

            triggered += 1

        except Exception as exc:
            logger.error("Failed to process record %s: %s", record_id, exc)
            # Mark record as failed so it can be retried
            try:
                collection.update_one(
                    {"_id": record["_id"]},
                    {
                        "$set": {
                            "inference_status": "error",
                            "inference_error": str(exc),
                            "inference_error_at": datetime.now(timezone.utc),
                        }
                    },
                )
            except Exception:
                pass
            errors += 1

    summary = {
        "status": "completed",
        "total_pending": len(pending_records),
        "triggered": triggered,
        "errors": errors,
    }
    logger.info("Lambda execution summary: %s", summary)
    return summary
