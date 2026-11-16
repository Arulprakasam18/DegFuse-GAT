"""
Lightweight logging configuration for DegFuse-GAT.

Uses Python's standard logging module. No wandb/mlflow/tensorboard.
"""

import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf


def setup_logging(
    run_name: str,
    output_dir: str = "results/metrics",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure logging to stdout and a log file.

    Args:
        run_name: Name of the experiment run (used for log filename).
        output_dir: Directory for log files.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{run_name}.log")

    logger = logging.getLogger("degfuse_gat")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # File handler
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_git_commit() -> str:
    """Return the current git commit SHA, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_hardware_info() -> dict:
    """Return a dict with GPU and CUDA information."""
    info = {
        "gpu": "N/A",
        "gpu_count": 0,
        "cuda_version": "N/A",
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda or "N/A"
    return info


def config_hash(cfg: DictConfig) -> str:
    """Return a short hash of the resolved config for tracking."""
    yaml_str = OmegaConf.to_yaml(cfg, resolve=True)
    return hashlib.md5(yaml_str.encode()).hexdigest()[:8]


def log_experiment_start(
    logger: logging.Logger,
    cfg: DictConfig,
    seed: int,
) -> None:
    """Log standard experiment metadata at INFO level."""
    hw = get_hardware_info()
    logger.info("=" * 60)
    logger.info("DegFuse-GAT Experiment")
    logger.info("=" * 60)
    logger.info("Config hash: %s", config_hash(cfg))
    logger.info("Seed: %d", seed)
    logger.info("Git commit: %s", get_git_commit())
    logger.info("GPU: %s (x%d)", hw["gpu"], hw["gpu_count"])
    logger.info("CUDA: %s", hw["cuda_version"])
    logger.info("PyTorch: %s", torch.__version__)
