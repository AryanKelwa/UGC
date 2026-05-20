"""
src/models/load_model.py
========================
Loads pre-trained HuggingFace models and tokenizers.
Supports quantization configuration and device mapping.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

log = logging.getLogger("pipeline.models.loader")

def load_model_and_tokenizer(
    model_name_or_path: str,
    quantization_config: dict | None = None,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto"
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Initialize and return model and tokenizer from HuggingFace.
    (Template integration with standard transformers pipeline).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    log.info("Loading tokenizer for %s...", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16
    
    log.info("Loading model %s on device: %s...", model_name_or_path, device_map)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    
    return model, tokenizer
