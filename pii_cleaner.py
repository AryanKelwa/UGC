"""
pii_cleaner.py
==============
Anonymizes **personal identifiers** from conversation text while
preserving location data that the lead-qualification model needs.

Replaces:
  - Emails        → [EMAIL]
  - Phone numbers → [PHONE]  (US, India, and generic international)
  - Person names  → [NAME]   (via spaCy NER)

Intentionally left UNTOUCHED:
  - Locations / cities / states  (training signal for preferred_locality)
  - Street addresses             (training signal for property identification)
  - ZIP / PIN codes              (training signal for locality)

The spaCy model is loaded *once* at module import to avoid repeated
expensive reloads when this module is called from a concurrent pipeline.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("pii_clean")

# ── Load NLP model once (important for performance) ───────────────────────────
# spaCy's nlp() call is thread-safe for inference.
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    _HAS_SPACY = True
    log.info("spaCy model 'en_core_web_sm' loaded successfully.")
except (ImportError, OSError) as exc:
    _nlp = None
    _HAS_SPACY = False
    log.warning(
        "spaCy model not available (%s). "
        "Person-name anonymization will be skipped. "
        "Install with: python -m spacy download en_core_web_sm",
        exc,
    )

# ── Pre-compiled regex patterns ───────────────────────────────────────────────

# Email addresses
_RE_EMAIL = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
)

# Phone numbers — covers common US, Indian, and international formats:
#   (555) 123-4567  |  555-123-4567  |  555.123.4567
#   +1 555 123 4567 |  +1-555-123-4567
#   +91 98765 43210 |  9876543210
_RE_PHONE = re.compile(
    r'(?<!\d)'                        # not preceded by a digit
    r'(?:\+?\d{1,3}[\s.-]?)?'        # optional country code
    r'(?:\(?\d{3}\)?[\s.-]?)?'        # optional area code with parens
    r'\d{3}[\s.-]?\d{4}'             # core 7 digits
    r'(?!\d)',                        # not followed by a digit
)

# Indian mobile (standalone 10-digit starting with 6-9)
_RE_PHONE_IN = re.compile(
    r'(?<!\d)'
    r'(?:\+?91[\s.-]?)?'
    r'[6-9]\d{9}'
    r'(?!\d)',
)


def clean_pii_text(text: str) -> str:
    """
    Anonymize personal identifiers from a conversation text string.

    Steps applied in order:
      1. Email addresses  → [EMAIL]
      2. Phone numbers    → [PHONE]
      3. PERSON entities  → [NAME]   (spaCy NER, if available)

    Locations, addresses, and ZIP codes are intentionally preserved.

    Parameters
    ----------
    text : str
        Raw conversation text.

    Returns
    -------
    str
        Anonymized text with personal identifiers substituted.
    """
    # 1. EMAILS
    text = _RE_EMAIL.sub("[EMAIL]", text)

    # 2. PHONE NUMBERS (US + India + generic international)
    text = _RE_PHONE_IN.sub("[PHONE]", text)   # Indian first (more specific)
    text = _RE_PHONE.sub("[PHONE]", text)       # then general

    # 3. PERSON NAMES via spaCy NER
    if _HAS_SPACY and _nlp is not None:
        doc = _nlp(text)

        # Collect PERSON entity replacements
        replacements: list[tuple[str, str]] = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                replacements.append((ent.text, "[NAME]"))

        # Apply longest-match first to avoid partial substitution issues
        replacements.sort(key=lambda x: len(x[0]), reverse=True)
        for original, token in replacements:
            text = text.replace(original, token)

    return text
