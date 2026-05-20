"""
src/data/preprocessing.py
==========================
Data preprocessing utility functions.
Contains text normalization, token length estimation, and cleanups.
"""

import re

def normalize_whitespace(text: str) -> str:
    """Replace multiple consecutive spaces/tabs/newlines with a single space."""
    if not isinstance(text, str):
        return text
    return re.sub(r'\s+', ' ', text).strip()

def estimate_token_count(text: str) -> int:
    """Vague heuristic to estimate tokens (approx 1 token = 4 chars or 0.75 words)."""
    if not isinstance(text, str):
        return 0
    return len(text.split()) * 4 // 3
