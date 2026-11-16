"""Paper Figure: Degree Distribution Histogram + Bucket Overlay."""
import builtins
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Auto-answer OGB prompts: "y" for downloads, "n" for updates
_original_input = builtins.input
def _auto_input(prompt=""):
    p = str(prompt).lower()
    if "update" in p:
        return "n"
    return "y"
builtins.input = _auto_input

# Fix PyTorch 2.6+ weights_only default change breaking OGB
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

DATASETS = ["Cora", "CiteSeer", "PubMed", "ogbn-arxiv"]
COLORS   = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
BUCKET_THRESHOLDS = [32, 128, 512]


def load_degrees(name):
    try:
        if name in ["Cora", "CiteSeer", "PubMed"]:
            from torch_geometric.datasets import Planetoid
            data = Planetoid(root="./data/Planetoid", name=name)[0]
        elif name == "ogbn-arxiv":
            from ogb.nodeproppred import PygNodePropPredDataset
            data = PygNodePropPredDataset(name="ogbn-arxiv", root="./data/OGB")[0]
        else:
            return None
        deg = torch.zeros(data.num_nodes, dtype=torch.long)
        deg.scatter_add_(0, data.edge_index[0],
                         torch.ones(data.edge_index.shape[1], dtype=torch.long))
        return deg.numpy()
    except Exception as e:
        print(f"  Skipping {name}: {e}")
        return None


if __name__ == "__main__":
    plots = [(n, d) for n in DATASETS if (d := load_degrees(n)) is not None]
    if not plots:
        print("No datasets found. Run setup_datasets.py first.")
        exit(1)

    fig, axes = plt.subplots(1, len(plots), figsize=(5 * len(plots), 4.5))
    if len(plots) == 1:
        axes = [axes]

    for i, (name, deg) in enumerate(plots):
        ax = axes[i]
        bins = np.logspace(0, np.log10(deg.max() + 1), 50)
        ax.hist(deg, bins=bins, color=COLORS[i], alpha=0.8, edgecolor="white", lw=0.5)

        # Bucket threshold lines
        for thresh in BUCKET_THRESHOLDS:
            if thresh < deg.max():
                ax.axvline(thresh, color="gray", ls="--", alpha=0.7, lw=1)
                ax.text(thresh * 1.1, ax.get_ylim()[1] * 0.5,
                        f"≤{thresh}", fontsize=8, color="gray", rotation=90)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Node Degree", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{name}\nmin={deg.min()} max={deg.max()} mean={deg.mean():.1f}",
                     fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2)
        print(f"  {name}: min={deg.min()} max={deg.max()} mean={deg.mean():.1f} "
              f"nodes={len(deg):,}")

    fig.suptitle("Degree Distributions with Bucket Thresholds", fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/degree_distribution.png", dpi=300, bbox_inches="tight")
    fig.savefig("figures/degree_distribution.pdf", bbox_inches="tight")
    print("Saved → figures/degree_distribution.png + .pdf")
