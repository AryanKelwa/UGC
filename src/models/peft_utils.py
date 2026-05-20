"""
src/models/peft_utils.py
========================
Configures parameter-efficient fine-tuning (PEFT/LoRA) wrappers.
"""

from peft import LoraConfig, get_peft_model, TaskType
from transformers import PreTrainedModel

def get_lora_wrapped_model(model: PreTrainedModel, peft_config_dct: dict) -> PreTrainedModel:
    """Wrap a base HuggingFace model in LoRA adapter layers."""
    config = LoraConfig(
        r=peft_config_dct.get("r", 16),
        lora_alpha=peft_config_dct.get("lora_alpha", 32),
        target_modules=peft_config_dct.get("target_modules"),
        lora_dropout=peft_config_dct.get("lora_dropout", 0.05),
        bias=peft_config_dct.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM
    )
    return get_peft_model(model, config)
