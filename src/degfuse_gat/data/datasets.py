"""
Dataset loading utilities for DegFuse-GAT experiments.

Supports Planetoid (Cora, CiteSeer, PubMed) and OGB (ogbn-arxiv,
ogbn-products, ogbn-proteins) datasets.
"""

import torch
from omegaconf import DictConfig


def load_dataset(name: str, device: torch.device, data_root: str = "./data"):
    """
    Load a graph dataset and move it to the specified device.

    Args:
        name: Dataset name (Cora, CiteSeer, PubMed, ogbn-arxiv,
              ogbn-products, ogbn-proteins, Reddit).
        device: Target torch device.
        data_root: Root directory for dataset storage.

    Returns:
        Tuple of (data, num_features, num_classes).
    """
    if name in ["Cora", "CiteSeer", "PubMed"]:
        from torch_geometric.datasets import Planetoid
        ds   = Planetoid(root=f"{data_root}/Planetoid", name=name)
        data = ds[0].to(device)
        return data, ds.num_node_features, ds.num_classes

    elif name == "ogbn-arxiv":
        from ogb.nodeproppred import PygNodePropPredDataset
        ds   = PygNodePropPredDataset(name="ogbn-arxiv", root=f"{data_root}/OGB")
        data = ds[0].to(device)
        data.y = data.y.squeeze()
        return data, data.x.shape[1], ds.num_classes

    elif name == "ogbn-products":
        from ogb.nodeproppred import PygNodePropPredDataset
        ds   = PygNodePropPredDataset(name="ogbn-products", root=f"{data_root}/OGB")
        data = ds[0].to(device)
        data.y = data.y.squeeze()
        return data, data.x.shape[1], ds.num_classes

    elif name == "ogbn-proteins":
        from ogb.nodeproppred import PygNodePropPredDataset
        ds   = PygNodePropPredDataset(name="ogbn-proteins", root=f"{data_root}/OGB")
        data = ds[0].to(device)
        # ogbn-proteins has no node features (x=None), only edge_attr [E,8]
        nf = 8
        data.x = torch.randn(data.num_nodes, nf, device=device)
        nc = ds.num_classes if ds.num_classes else 112  # 112 binary labels
        data.y = data.y.squeeze()
        return data, nf, nc

    elif name == "Reddit":
        from torch_geometric.datasets import Reddit
        ds   = Reddit(root=f"{data_root}/Reddit")
        data = ds[0].to(device)
        return data, data.x.shape[1], ds.num_classes

    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_split_masks(name: str, data, ds, device: torch.device):
    """
    Get train/val/test masks for a dataset.

    Args:
        name: Dataset name.
        data: PyG data object.
        ds: PyG dataset object.
        device: Target torch device.

    Returns:
        Tuple of (train_mask, val_mask, test_mask, labels).
    """
    if name in ["Cora", "CiteSeer", "PubMed"]:
        return data.train_mask, data.val_mask, data.test_mask, data.y

    elif name == "ogbn-arxiv":
        split = ds.get_idx_split()
        N = data.num_nodes
        train_mask = torch.zeros(N, dtype=torch.bool, device=device)
        val_mask   = torch.zeros(N, dtype=torch.bool, device=device)
        test_mask  = torch.zeros(N, dtype=torch.bool, device=device)
        train_mask[split["train"]] = True
        val_mask[split["valid"]]   = True
        test_mask[split["test"]]   = True
        return train_mask, val_mask, test_mask, data.y

    else:
        raise ValueError(f"Split masks not implemented for: {name}")
