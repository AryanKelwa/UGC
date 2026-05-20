"""
src/models/quantization.py
==========================
Prepares BitsAndBytes quantizers for 4-bit/8-bit precision models.
"""

from transformers import BitsAndBytesConfig
import torch

def get_bnb_quantization_config(quant_cfg_dct: dict) -> BitsAndBytesConfig:
    """Create the BitsAndBytes config for QLoRA fine-tuning."""
    compute_dtype = torch.bfloat16 if quant_cfg_dct.get("bnb_4bit_compute_dtype") == "bfloat16" else torch.float16
    
    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg_dct.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant_cfg_dct.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_cfg_dct.get("bnb_4bit_use_double_quant", True)
    )
