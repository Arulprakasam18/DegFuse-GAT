"""
Train DegFuse-GAT.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --override training.seed=123 training.epochs=100
"""

import argparse
import json
import os
import sys

from omegaconf import OmegaConf

from src.degfuse_gat.utils.compat import apply_patches
from src.degfuse_gat.utils.config import load_config
from src.degfuse_gat.utils.seed import set_global_seed
from src.degfuse_gat.utils.logging import setup_logging, log_experiment_start
from src.degfuse_gat.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="DegFuse-GAT training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--override", nargs="*", default=[], help="key=value overrides")
    args = parser.parse_args()

    apply_patches()
    cfg = load_config(args.config, args.override if args.override else None)

    seed = cfg.training.seed
    set_global_seed(seed)

    run_name = cfg.logging.get("run_name", "default")
    output_dir = cfg.logging.get("output_dir", "results/metrics")
    logger = setup_logging(run_name, output_dir)
    log_experiment_start(logger, cfg, seed)

    trainer = Trainer(cfg)
    metrics = trainer.fit()

    # Save metrics
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, f"{run_name}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics saved to %s", metrics_path)

    # Save resolved config
    config_path = os.path.join(output_dir, f"{run_name}.config.yaml")
    OmegaConf.save(cfg, config_path)
    logger.info("Config saved to %s", config_path)


if __name__ == "__main__":
    main()
