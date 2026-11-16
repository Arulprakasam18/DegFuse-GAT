"""
Paper Figure 2: Latency vs Edge Count with Crossover Annotation.

This is the NEW key figure after Tier 1 analysis.
Shows clearly WHERE DegFuse wins (above crossover) and WHY it is slower below.
Provides the honest, complete story: memory + latency tradeoff with clear boundary.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


if __name__ == "__main__":
    if not os.path.exists("results/benchmark_crossover.json"):
        print("Run: python -m benchmarks.benchmark_crossover")
        exit(1)

    with open("results/benchmark_crossover.json") as f:
        d = json.load(f)

    data       = d["data"]
    crossover  = d.get("crossover_edges")
    edges      = [r["edge_count"] for r in data]
    t_pyg      = [r["pyg_ms"]    for r in data]
    t_df       = [r["degfuse_ms"] for r in data]
    speedups   = [r["speedup"]   for r in data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("DegFuse-GAT vs PyG: Latency & Speedup vs Graph Size",
                 fontsize=14, fontweight="bold")

    # ---- Top: latency ----
    ax1.set_facecolor("#fafafa")
    ax1.plot(edges, t_pyg, "o-", color="#e74c3c", lw=2.5, ms=8,
             label="PyG GAT (baseline)")
    ax1.plot(edges, t_df,  "s-", color="#2ecc71", lw=2.5, ms=8,
             label="DegFuse-GAT (ours)")

    if crossover:
        ax1.axvline(crossover, color="#7f8c8d", ls="--", lw=1.5, alpha=0.8)
        ax1.text(crossover * 1.08, ax1.get_ylim()[1] * 0.85,
                 f"Crossover\nE* ≈ {crossover//1000}k", fontsize=10,
                 color="#7f8c8d", fontweight="bold")
        # Shade regions
        ax1.axvspan(edges[0], crossover, alpha=0.05, color="#e74c3c",
                    label="PyG wins (launch overhead dominates)")
        ax1.axvspan(crossover, edges[-1], alpha=0.05, color="#2ecc71",
                    label="DegFuse wins (memory BW savings dominate)")

    ax1.set_xscale("log")
    ax1.set_ylabel("Forward Latency (ms)", fontsize=12)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ---- Bottom: speedup ----
    ax2.set_facecolor("#fafafa")
    colors = ["#2ecc71" if s > 1.0 else "#e74c3c" for s in speedups]
    ax2.bar(range(len(edges)), speedups, color=colors, alpha=0.85, width=0.6)
    ax2.axhline(1.0, color="black", lw=1.5, ls="--", alpha=0.6)
    ax2.set_xticks(range(len(edges)))
    ax2.set_xticklabels([f"{e//1000}k" for e in edges], fontsize=9, rotation=30)
    ax2.set_ylabel("Speedup (PyG / DegFuse)", fontsize=11)
    ax2.set_ylim(0, max(speedups) * 1.2)
    ax2.text(0.02, 0.92, "DegFuse faster →",
             transform=ax2.transAxes, fontsize=9, color="#2ecc71")
    ax2.text(0.02, 0.08, "← PyG faster",
             transform=ax2.transAxes, fontsize=9, color="#e74c3c")

    # Speedup labels
    for i, s in enumerate(speedups):
        ax2.text(i, s + 0.03, f"{s:.2f}×", ha="center", va="bottom",
                 fontsize=8, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    os.makedirs("figures", exist_ok=True)
    plt.tight_layout()
    fig.savefig("figures/crossover_analysis.png", dpi=300, bbox_inches="tight")
    fig.savefig("figures/crossover_analysis.pdf", bbox_inches="tight")
    print("Saved → figures/crossover_analysis.png + .pdf")
