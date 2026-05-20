"""
src/utils/seed.py
=================
Sets global random seeds for reproducibility across random, numpy, and torch.
"""

import random
import numpy as np

def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
