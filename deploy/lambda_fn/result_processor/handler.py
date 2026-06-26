"""
deploy/lambda_fn/result_processor/handler.py
=============================================
AWS Lambda function that processes SageMaker Async Inference results.

Triggered by: S3 PutObject event on the inference-outputs/ prefix.

When SageMaker Async completes an inference request, it writes the
output JSON to s3://<bucket>/inference-outputs/<request-id>.out
This Lambda reads that file and updates the corresponding MongoDB
lead record with the inference result and final status.

Environment Variables (set by CDK):
  S3_BUCKET            — S3 bucket name
  MONGODB_SECRET_NAME  — Secrets Manager key for MongoDB URL
  MONGODB_DB_NAME      — "realestate_data"
  MONGODB_COLLECTION   — "leads"
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from urllib.parse import unquote_plus

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── AWS clients ───────────────────────────────────────────────
s3_client = boto3.client("s3")
secrets_client = boto3.client("secretsmanager")

# ── Cached MongoDB client ─────────────────────────────────────
_mongo_client = None

MONGODB_SECRET_NAME = os.environ.get("MONGODB_SECRET_NAME", "llm/mongodb-url")
DB_NAME = os.environ.get("MONGODB_DB_NAME", "realestate_data")
COLLECTION_NAME = os.environ.get("MONGODB_COLLECTION", "leads")


def _get_mongo_collection():
    """Lazily initialise and cache the MongoDB connection."""
    global _mongo_client
    if _mongo_client is None:
        try:
            from pymongo import MongoClient
        except ImportError:
            raise RuntimeError("pymongo not installed in Lambda layer.")
        secret = secrets_client.get_secret_value(SecretId=MONGODB_SECRET_NAME)
        _mongo_client = MongoClient(secret["SecretString"], serverSelectionTimeoutMS=5000)
    return _mongo_client[DB_NAME][COLLECTION_NAME]


def _read_s3_output(bucket: str, key: str) -> dict:
    """Read inference output JSON from S3."""
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read().decode("utf-8")

    # SageMaker Async output may be a JSON array or object
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        logger.warning("Output is not valid JSON — storing as raw string.")
        data = {"raw_output": body}

    return data


def _extract_request_id_from_key(key: str) -> str | None:
    """
    Derive the original request_id from the S3 output key.

    Input key format (from lambda inference handler):
      inference-outputs/<request_id>.out  or
      inference-outputs/<request_id>

    The request_id stored in MongoDB is: <mongo_id>-<uuid8>
    """
    filename = key.split("/")[-1]
    # Strip .out or .json extension if present
    request_id = filename.removesuffix(".out").removesuffix(".json")
    return request_id if request_id else None


def _update_lead_record(collection, request_id: str, inference_output: dict) -> bool:
    """
    Find the lead record by sagemaker_request_id and update it with results.
    Returns True if a record was found and updated.
    """
    # Extract the lead score / qualification from the model output
    # SageMaker HF DLC returns: [{"generated_text": "..."}]
    generated_text = ""
    if isinstance(inference_output, list) and len(inference_output) > 0:
        generated_text = inference_output[0].get("generated_text", "")
    elif isinstance(inference_output, dict):
        generated_text = inference_output.get("generated_text", str(inference_output))

    result = collection.update_one(
        {"sagemaker_request_id": request_id},
        {
            "$set": {
                "inference_status": "completed",
                "inference_result": generated_text,
                "inference_raw_output": inference_output,
                "inference_completed_at": datetime.now(timezone.utc),
            }
        },
    )
    return result.modified_count > 0


def lambda_handler(event, context):
    """
    Main entry point — triggered by S3 PutObject on inference-outputs/ prefix.
    """
    logger.info("Result processor triggered with %d S3 records.", len(event.get("Records", [])))

    processed = 0
    errors = 0
    not_found = 0

    try:
        collection = _get_mongo_collection()
    except Exception as exc:
        logger.error("MongoDB connection failed: %s", exc)
        return {"status": "error", "message": str(exc)}

    for record in event.get("Records", []):
        try:
            bucket = record["s3"]["bucket"]["name"]
            key = unquote_plus(record["s3"]["object"]["key"])

            # Skip non-output files (e.g., input payloads in inference-inputs/)
            if "inference-inputs/" in key:
                logger.debug("Skipping input file: %s", key)
                continue

            logger.info("Processing S3 output: s3://%s/%s", bucket, key)

            # ── Read inference output from S3 ─────────────────
            inference_output = _read_s3_output(bucket, key)

            # ── Derive request_id from S3 key ─────────────────
            request_id = _extract_request_id_from_key(key)
            if not request_id:
                logger.warning("Could not derive request_id from key: %s", key)
                errors += 1
                continue

            # ── Update MongoDB lead record ────────────────────
            updated = _update_lead_record(collection, request_id, inference_output)

            if updated:
                logger.info("Updated lead record for request_id: %s", request_id)
                processed += 1
            else:
                logger.warning(
                    "No lead record found for request_id: %s — "
                    "may have already been processed or the ID scheme changed.",
                    request_id,
                )
                not_found += 1

        except Exception as exc:
            logger.error("Error processing S3 record %s: %s", record.get("s3", {}).get("object", {}).get("key", "?"), exc)
            errors += 1

    summary = {
        "status": "completed",
        "processed": processed,
        "not_found": not_found,
        "errors": errors,
    }
    logger.info("Result processor summary: %s", summary)
    return summary
