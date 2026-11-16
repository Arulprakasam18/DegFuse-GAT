"""
Paper Figures: Split Latency — Tier 1 overhead + Tier 2 speedup.

Generates two separate figures instead of one mixed chart:
  figures/latency_tier1.pdf — overhead bars with 10% threshold line
  figures/latency_tier2.pdf — speedup bars on ogbn-arxiv+

Run: python -m plotting.plot_latency_split
"""

import glob
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TIER1_DATASETS = {"Cora", "CiteSeer", "PubMed"}
CROSSOVER_EDGES = 100_000


if __name__ == "__main__":
    files = sorted(glob.glob("results/benchmark_forward_*.json"))
    if not files:
        print("Run benchmark_forward first for each dataset.")
        exit(1)

    tier1, tier2 = [], []
    for fp in files:
        with open(fp) as f:
            r = json.load(f)
        ds = r["dataset"]
        # Skip datasets where PyG OOM'd (value is "OOM" string, not a dict)
        if not isinstance(r.get("pyg"), dict):
            print(f"  Skipping {ds} (PyG OOM)")
            continue
        pyg_ms = r["pyg"]["latency_ms"]
        auto = r.get("degfuse_auto", r.get("degfuse_da", {}))
        auto_ms = auto["latency_ms"] if isinstance(auto, dict) else 0
        fused = r.get("degfuse_fused", r.get("degfuse_sc", {}))
        fused_ms = fused["latency_ms"] if isinstance(fused, dict) else 0
        E = r.get("num_edges", 0)
        entry = {
            "dataset": ds, "edges": E,
            "pyg_ms": pyg_ms, "auto_ms": auto_ms, "fused_ms": fused_ms,
        }
        if ds in TIER1_DATASETS:
            tier1.append(entry)
        else:
            tier2.append(entry)

    os.makedirs("figures", exist_ok=True)

    # ---- Figure 1: Tier 1 — Overhead bars ----
    if tier1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_facecolor("#fafafa")

        names = [t["dataset"] for t in tier1]
        overhead_fused = [(t["fused_ms"] / t["pyg_ms"] - 1) * 100 for t in tier1]
        overhead_auto = [(t["auto_ms"] / t["pyg_ms"] - 1) * 100 for t in tier1]

        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w/2, overhead_fused, w, label="DegFuse Fused", color="#e74c3c", alpha=0.85)
        ax.bar(x + w/2, overhead_auto, w, label="DegFuse Auto", color="#3498db", alpha=0.85)

        # 10% threshold line
        ax.axhline(10, color="#7f8c8d", ls="--", lw=1.5, alpha=0.7, label="10% threshold")
        ax.axhline(0, color="black", lw=0.8)

        # Value labels
        for i, (of, oa) in enumerate(zip(overhead_fused, overhead_auto)):
            ax.text(x[i] - w/2, of + 1, f"{of:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.text(x[i] + w/2, oa + 1, f"{oa:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=12)
        ax.set_ylabel("Latency Overhead vs PyG (%)", fontsize=12)
        ax.set_title("Tier 1: Small Graphs (E < 100k) — Latency Overhead", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig("figures/latency_tier1.png", dpi=300, bbox_inches="tight")
        fig.savefig("figures/latency_tier1.pdf", bbox_inches="tight")
        print("Saved → figures/latency_tier1.png + .pdf")
        plt.close(fig)

    # ---- Figure 2: Tier 2 — Speedup bars ----
    if tier2:
        fig, ax = plt.subplots(figsize=(max(6, len(tier2) * 3), 5))
        ax.set_facecolor("#fafafa")

        names = [f"{t['dataset']}\n({t['edges']//1000}k edges)" for t in tier2]
        speedup_fused = [t["pyg_ms"] / t["fused_ms"] if t["fused_ms"] > 0 else 0 for t in tier2]
        speedup_auto = [t["pyg_ms"] / t["auto_ms"] if t["auto_ms"] > 0 else 0 for t in tier2]

        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w/2, speedup_fused, w, label="DegFuse Fused", color="#2ecc71", alpha=0.85)
        ax.bar(x + w/2, speedup_auto, w, label="DegFuse Auto", color="#27ae60", alpha=0.85)

        ax.axhline(1.0, color="black", lw=1.5, ls="--", alpha=0.6)

        # Value labels
        for i, (sf, sa) in enumerate(zip(speedup_fused, speedup_auto)):
            ax.text(x[i] - w/2, sf + 0.02, f"{sf:.2f}×", ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.text(x[i] + w/2, sa + 0.02, f"{sa:.2f}×", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11)
        ax.set_ylabel("Speedup vs PyG", fontsize=12)
        ax.set_title("Tier 2: Large Graphs (E ≥ 100k) — DegFuse Speedup", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, max(max(speedup_fused), max(speedup_auto)) * 1.2)

        fig.tight_layout()
        fig.savefig("figures/latency_tier2.png", dpi=300, bbox_inches="tight")
        fig.savefig("figures/latency_tier2.pdf", bbox_inches="tight")
        print("Saved → figures/latency_tier2.png + .pdf")
        plt.close(fig)

    if not tier1 and not tier2:
        print("No benchmark data found.")
