"""
config/config.py
================
Central configuration for the LLM training-data generator.

Edit the values in this file to control the generation run.
No command-line flags are needed.
"""

from pathlib import Path

# ── Project root (one level up from this file) ────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

# Where the .env file lives
ENV_FILE = ROOT_DIR / ".env"

# Where generated batch_NNN.json files are saved
OUTPUT_DIR = ROOT_DIR / "training-data"

# Where PII-cleaned batch files are saved
CLEAN_OUTPUT_DIR = ROOT_DIR / "training-data-clean"

# ─────────────────────────────────────────────────────────────────────────────
# Gemini API settings
# ─────────────────────────────────────────────────────────────────────────────

# Base URL for Gemini REST API
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Model to use for generation
# Options: "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash", etc.
GEMINI_MODEL = "gemini-2.5-flash"

# Generation parameters passed to the API
GENERATION_CONFIG = {
    "temperature": 1.0,
    "maxOutputTokens": 65536,
}

# ─────────────────────────────────────────────────────────────────────────────
# Batch / generation settings
# ─────────────────────────────────────────────────────────────────────────────

# Total number of batches to generate
# Each batch asks Gemini for 30 JSONL examples  →  34 × 30 = 1 020 examples
TOTAL_BATCHES = 34

# First batch number (change to resume from a specific point)
START_BATCH = 1

# ─────────────────────────────────────────────────────────────────────────────
# Retry / rate-limit settings
# ─────────────────────────────────────────────────────────────────────────────

# Maximum total attempts (across all keys) before giving up on a single batch
MAX_RETRIES_PER_BATCH = 10

# Seconds to sleep when all keys are exhausted before trying again
RETRY_SLEEP_SECONDS = 10

# Polite pause (seconds) between successive batch *submissions* when running
# in parallel mode (keeps the initial burst from hammering the API).
BETWEEN_BATCH_SLEEP = 1

# Number of batches to generate in parallel (ThreadPoolExecutor max_workers).
# Tune this to the number of API keys you have; more workers = more concurrency
# but also more simultaneous requests per key.
MAX_WORKERS = 4

# Number of parallel workers for the PII-cleaning pipeline
# (clean_training_data.py).  Controls both batch-level and record-level
# concurrency.
PII_WORKERS = 4

# HTTP status codes considered rate-limit / retriable
RATE_LIMIT_CODES = {429, 503}

# Request timeout in seconds
REQUEST_TIMEOUT = 180
