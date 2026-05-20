"""
src/data/quality_checks.py
==========================
Performs basic sanity and quality checks on generated datasets.
"""

import json
from pathlib import Path
import logging

log = logging.getLogger("pipeline.data.quality")

def verify_jsonl_dataset(file_path: Path | str) -> bool:
    """Load and verify standard format schema compatibility of SFT JSONL split."""
    path = Path(file_path)
    if not path.exists():
        log.error("Dataset file not found: %s", path)
        return False
        
    log.info("Running quality checks on %s...", path.name)
    success = True
    total_records = 0
    malformed_records = 0
    
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            total_records += 1
            try:
                data = json.loads(line)
                if "conversations" not in data:
                    log.error("Line %d — Missing 'conversations' key.", lineno)
                    success = False
                    malformed_records += 1
                    continue
                
                convs = data["conversations"]
                if not isinstance(convs, list) or len(convs) != 3:
                    log.error("Line %d — 'conversations' is not a list of 3 items.", lineno)
                    success = False
                    malformed_records += 1
                    continue
                    
                roles = [m.get("role") for m in convs]
                if roles != ["system", "user", "assistant"]:
                    log.error("Line %d — Conversation roles mismatch: %s", lineno, roles)
                    success = False
                    malformed_records += 1
                    continue
            except json.JSONDecodeError as exc:
                log.error("Line %d — Invalid JSON formatting: %s", lineno, exc)
                success = False
                malformed_records += 1

    log.info(
        "Quality Checks Finished. Inspected: %d | Malformed: %d | Status: %s",
        total_records, malformed_records, "PASSED" if success else "FAILED"
    )
    return success
