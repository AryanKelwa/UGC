"""
src/inference/generate.py
=========================
Performs local model inference.
Given a raw lead qualification dialogue, runs text generation to return predicted metrics.
"""

from __future__ import annotations

import logging
from transformers import pipeline

from src.utils.logger import setup_logger
from src.data.formatting import SYSTEM_PROMPT

log = setup_logger("pipeline.inference.generate")

def generate_qualification(
    model,
    tokenizer,
    conversation_text: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7
) -> str:
    """Run generation on a conversation dialogue to get lead qualification JSON output."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Lead conversation:\n{conversation_text}"}
    ]
    
    # Format messages using the tokenizer's chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    log.info("Generating qualification response...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True if temperature > 0.0 else False,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Extract only the generated tokens (ignoring the prompt)
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response
