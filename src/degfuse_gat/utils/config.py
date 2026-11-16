"""
Configuration loading utilities using OmegaConf.

Provides a single function to load a YAML config with optional CLI overrides.
"""

from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf


def load_config(
    path: str,
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """
    Load a YAML configuration file and apply optional key=value overrides.

    Args:
        path: Path to the YAML config file.
        overrides: List of "dotted.key=value" strings to override config values.

    Returns:
        Resolved OmegaConf DictConfig.

    Example:
        cfg = load_config("configs/default.yaml", ["training.epochs=10", "training.seed=123"])
    """
    cfg = OmegaConf.load(path)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg
