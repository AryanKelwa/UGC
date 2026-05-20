"""
src/utils/gpu_monitor.py
========================
Helper module to log GPU VRAM usage and check PyTorch CUDA status.
"""

import logging

log = logging.getLogger("pipeline.utils.gpu")

def log_gpu_memory() -> None:
    """Log current PyTorch CUDA memory allocation stats."""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                name = torch.cuda.get_device_name(i)
                log.info(
                    "GPU [%d] (%s) — Memory Allocated: %.2f GB | Reserved: %.2f GB",
                    i, name, allocated, reserved
                )
        else:
            log.debug("CUDA GPU is not available.")
    except ImportError:
        log.debug("PyTorch is not installed. GPU monitoring skipped.")
