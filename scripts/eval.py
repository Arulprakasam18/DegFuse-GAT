"""
Evaluate a DegFuse-GAT checkpoint.

Usage:
    python scripts/eval.py --config configs/default.yaml
    python scripts/eval.py --config configs/default.yaml --override evaluation.checkpoint_path=path/to/ckpt
"""

import argparse
import json
import logging
import os
import time

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.degfuse_gat.utils.compat import apply_patches
from src.degfuse_gat.utils.config import load_config
from src.degfuse_gat.utils.seed import set_global_seed
from src.degfuse_gat.utils.logging import setup_logging, log_experiment_start, get_hardware_info
from src.degfuse_gat.data.datasets import load_dataset
from src.degfuse_gat.models.degfuse_gat_model import DegFuseGAT
from src.degfuse_gat.models.csr_utils import edge_index_to_csr

logger = logging.getLogger("degfuse_gat")


def main() -> None:
    parser = argparse.ArgumentParser(description="DegFuse-GAT evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--override", nargs="*", default=[], help="key=value overrides")
    args = parser.parse_args()

    apply_patches()
    cfg = load_config(args.config, args.override if args.override else None)

    seed = cfg.training.seed
    set_global_seed(seed)

    run_name = cfg.logging.get("run_name", "default")
    output_dir = cfg.logging.get("output_dir", "results/metrics")
    log = setup_logging(f"{run_name}_eval", output_dir)
    log_experiment_start(log, cfg, seed)

    device = torch.device(cfg.hardware.device)
    data, num_features, num_classes = load_dataset(
        cfg.data.dataset, device, cfg.data.root
    )

    row_ptr, col_idx, _ = edge_index_to_csr(data.edge_index, data.num_nodes)

    model = DegFuseGAT(
        in_channels=num_features,
        hidden_channels=cfg.model.hidden_dim,
        out_channels=num_classes,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        dispatch_mode=cfg.model.get("dispatch_mode", "auto"),
    ).to(device)

    # Load checkpoint if specified
    ckpt_path = cfg.evaluation.get("checkpoint_path", None)
    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        log.info("Loaded checkpoint from %s", ckpt_path)
    else:
        log.info("No checkpoint loaded — evaluating randomly initialized model")

    model.eval()
    t_start = time.time()

    with torch.no_grad():
        logits = model(data.x, row_ptr, col_idx, data.edge_index)

    # Compute metrics on available masks
    metrics = {}
    for split_name in ["train_mask", "val_mask", "test_mask"]:
        mask = getattr(data, split_name, None)
        if mask is not None:
            acc = (logits[mask].argmax(1) == data.y[mask]).float().mean().item()
            metrics[f"{split_name.replace('_mask', '')}_acc"] = acc
            log.info("%s accuracy: %.4f", split_name, acc)

    duration = time.time() - t_start
    hw = get_hardware_info()

    result = {
        "run_name": f"{run_name}_eval",
        "seed": seed,
        "dataset": cfg.data.dataset,
        "hardware": hw,
        "metrics": metrics,
        "duration_sec": round(duration, 2),
    }

    os.makedirs(output_dir, exist_ok=True)
    eval_path = os.path.join(output_dir, f"{run_name}_eval.json")
    with open(eval_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info("Evaluation saved to %s", eval_path)


if __name__ == "__main__":
    main()
