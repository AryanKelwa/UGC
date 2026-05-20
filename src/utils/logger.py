"""
src/utils/logger.py
===================
Enterprise logger setup.
Provides structured, readable logging for dataset generation and training.
"""

from __future__ import annotations

import logging
import sys

def setup_logger(name: str = "pipeline", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a styled standard library logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if setup multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  [%(name)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
