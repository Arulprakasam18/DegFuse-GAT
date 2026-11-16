"""
Global seed utility for reproducible experiments.

Sets all relevant random seeds and enables deterministic operations.
"""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set all random seeds for reproducibility.

    Seeds: Python random, numpy, torch (CPU + CUDA), sets cuDNN deterministic,
    sets PYTHONHASHSEED env var, and optionally enables
    torch.use_deterministic_algorithms.

    Args:
        seed: Integer seed value.
        deterministic: If True, enable fully deterministic operations
                       (may reduce performance).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older PyTorch versions don't support warn_only
            torch.use_deterministic_algorithms(True)
