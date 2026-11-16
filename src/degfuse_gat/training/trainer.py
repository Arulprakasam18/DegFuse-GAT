"""
Training loop for DegFuse-GAT node classification.
"""

import json
import logging
import os
import time
from typing import Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from src.degfuse_gat.data.datasets import load_dataset, get_split_masks
from src.degfuse_gat.models.baseline_gat import BaselineGAT
from src.degfuse_gat.models.csr_utils import edge_index_to_csr, print_csr_stats
from src.degfuse_gat.models.degfuse_gat_model import DegFuseGAT
from src.degfuse_gat.models.degree_aware_scheduler import print_bucket_stats
from src.degfuse_gat.utils.logging import get_hardware_info, config_hash

logger = logging.getLogger("degfuse_gat")


class Trainer:
    """
    Trainer for DegFuse-GAT and baseline GAT node classification.

    Args:
        cfg: OmegaConf DictConfig with all experiment parameters.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.hardware.device)

    def fit(self) -> dict:
        """
        Run the full training loop.

        Returns:
            Dict with final metrics (best_val_acc, best_test_acc, duration_sec, etc.).
        """
        cfg = self.cfg
        device = self.device

        # Load data
        data_root = cfg.data.root
        dataset_name = cfg.data.dataset
        data, num_features, num_classes = load_dataset(
            dataset_name, device, data_root
        )

        # Build split masks
        if dataset_name in ["Cora", "CiteSeer", "PubMed"]:
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
            labels = data.y
        elif dataset_name == "ogbn-arxiv":
            from ogb.nodeproppred import PygNodePropPredDataset
            ds = PygNodePropPredDataset(
                name="ogbn-arxiv", root=f"{data_root}/OGB"
            )
            split = ds.get_idx_split()
            N = data.num_nodes
            train_mask = torch.zeros(N, dtype=torch.bool, device=device)
            val_mask = torch.zeros(N, dtype=torch.bool, device=device)
            test_mask = torch.zeros(N, dtype=torch.bool, device=device)
            train_mask[split["train"]] = True
            val_mask[split["valid"]] = True
            test_mask[split["test"]] = True
            labels = data.y
        else:
            raise ValueError(f"Training not implemented for: {dataset_name}")

        # Build CSR
        row_ptr, col_idx, _ = edge_index_to_csr(data.edge_index, data.num_nodes)
        E = int(row_ptr[-1].item())
        logger.info("Dataset: %s  N=%d  E=%d", dataset_name, data.num_nodes, E)

        # Build model
        dispatch_mode = cfg.model.get("dispatch_mode", "auto")
        edge_threshold = cfg.adaptive.get("edge_threshold", 100_000)

        model = DegFuseGAT(
            in_channels=num_features,
            hidden_channels=cfg.model.hidden_dim,
            out_channels=num_classes,
            num_heads=cfg.model.num_heads,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            dispatch_mode=dispatch_mode,
            edge_threshold=edge_threshold,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

        best_val, best_test = 0.0, 0.0
        log_interval = cfg.logging.get("log_interval", 10)
        t_start = time.time()

        for epoch in range(1, cfg.training.epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(data.x, row_ptr, col_idx, data.edge_index)
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()

            if epoch % log_interval == 0 or epoch == cfg.training.epochs:
                model.eval()
                with torch.no_grad():
                    logits = model(data.x, row_ptr, col_idx, data.edge_index)
                    val_acc = (
                        (logits[val_mask].argmax(1) == labels[val_mask])
                        .float()
                        .mean()
                        .item()
                    )
                    test_acc = (
                        (logits[test_mask].argmax(1) == labels[test_mask])
                        .float()
                        .mean()
                        .item()
                    )
                if val_acc > best_val:
                    best_val, best_test = val_acc, test_acc
                logger.info(
                    "Epoch %3d | Loss %.4f | Val %.4f | Test %.4f",
                    epoch, loss.item(), val_acc, test_acc,
                )

        duration = time.time() - t_start
        hw = get_hardware_info()

        metrics = {
            "run_name": cfg.logging.get("run_name", "default"),
            "config_hash": config_hash(cfg),
            "seed": cfg.training.seed,
            "dataset": dataset_name,
            "hardware": hw,
            "metrics": {
                "final_val_acc": best_val,
                "final_test_acc": best_test,
                "final_train_loss": loss.item(),
            },
            "duration_sec": round(duration, 2),
        }

        logger.info("Best Test Accuracy: %.4f", best_test)
        logger.info("Training time: %.1f s", duration)

        return metrics
