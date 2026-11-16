"""
Compatibility patches for OGB and PyTorch version differences.

Call `apply_patches()` once at program startup before importing OGB datasets.
This module consolidates monkey-patches that were previously duplicated
across multiple files.
"""

import builtins
import torch


_patches_applied = False


def apply_patches() -> None:
    """
    Apply compatibility patches:
      1. Auto-answer OGB dataset prompts ("y" for downloads, "n" for updates)
      2. Fix PyTorch 2.6+ weights_only default change breaking OGB loading
    """
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    # Patch 1: Auto-answer OGB interactive prompts
    _original_input = builtins.input

    def _auto_input(prompt=""):
        p = str(prompt).lower()
        if "update" in p:
            return "n"
        return "y"

    builtins.input = _auto_input

    # Patch 2: Fix torch.load weights_only default for OGB compatibility
    _original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load
