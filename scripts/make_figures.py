"""
Regenerate all paper figures from results/metrics/ JSON files.

Usage:
    python scripts/make_figures.py

Reads all JSON files under results/metrics/ and generates figures
into results/figures/.
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

METRICS_DIR = "results/metrics"
FIGURES_DIR = "results/figures"


def plot_latency_comparison() -> None:
    """Generate forward latency comparison bar chart."""
    files = sorted(glob.glob(os.path.join(METRICS_DIR, "benchmark_forward_*.json")))
    if not files:
        print("  [SKIP] No benchmark_forward_*.json files found")
        return

    datasets, t_pyg, t_sc, t_da = [], [], [], []
    for fp in files:
        with open(fp) as f:
            r = json.load(f)
        if not isinstance(r.get("pyg"), dict):
            continue
        datasets.append(r["dataset"])
        t_pyg.append(r["pyg"]["latency_ms"])
        fused = r.get("degfuse_fused", r.get("degfuse_sc", {}))
        t_sc.append(fused["latency_ms"] if isinstance(fused, dict) else 0)
        auto = r.get("degfuse_auto", r.get("degfuse_da", {}))
        t_da.append(auto["latency_ms"] if isinstance(auto, dict) else 0)

    if not datasets:
        return

    x = np.arange(len(datasets))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 3), 6))
    ax.set_facecolor("#fafafa")
    ax.bar(x - w, t_pyg, w, label="PyG GAT", color="#e74c3c", alpha=0.88)
    ax.bar(x, t_sc, w, label="DegFuse-SC", color="#3498db", alpha=0.88)
    ax.bar(x + w, t_da, w, label="DegFuse-DA (ours)", color="#2ecc71", alpha=0.88)

    for i, (tp, td) in enumerate(zip(t_pyg, t_da)):
        if td > 0:
            ax.text(x[i] + w, td + 0.1, f"{tp/td:.1f}x",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1a6e3c")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylabel("Forward Latency (ms)", fontsize=13)
    ax.set_title("Forward Pass Latency: DegFuse-GAT vs Baselines", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "latency_comparison.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "latency_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved latency_comparison.png + .pdf")


def plot_memory_scaling() -> None:
    """Generate memory vs edge count scaling plot."""
    oom_path = os.path.join(METRICS_DIR, "benchmark_oom.json")
    if not os.path.exists(oom_path):
        print("  [SKIP] benchmark_oom.json not found")
        return

    with open(oom_path) as f:
        data = json.load(f)

    VRAM_LIMIT = 10 * 1024
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#fafafa")

    edges_all, pyg_vals, df_vals, oom_edges = [], [], [], []
    for r in data:
        E = r["edges"]
        edges_all.append(E)
        p = r["pyg_memory_mb"]
        d = r["degfuse_memory_mb"]
        pyg_vals.append(p if p != "OOM" else None)
        df_vals.append(d if d != "OOM" else None)
        if p == "OOM":
            oom_edges.append(E)

    valid_pyg = [(e, m) for e, m in zip(edges_all, pyg_vals) if m is not None]
    valid_df = [(e, m) for e, m in zip(edges_all, df_vals) if m is not None]

    if valid_pyg:
        ex, my = zip(*valid_pyg)
        ax.plot(ex, my, "o-", color="#e74c3c", lw=2.5, ms=8, label="PyG GAT (baseline)")
    if valid_df:
        ex, md = zip(*valid_df)
        ax.plot(ex, md, "s-", color="#2ecc71", lw=2.5, ms=8, label="DegFuse-GAT (ours)")

    for oe in oom_edges:
        ax.annotate("OOM", xy=(oe, VRAM_LIMIT * 1.05),
                    fontsize=11, color="#e74c3c", ha="center", fontweight="bold")

    ax.axhline(y=VRAM_LIMIT, color="gray", ls="--", alpha=0.6, lw=1.5, label="10 GB VRAM limit")
    ax.set_xscale("log")
    ax.set_xlabel("Number of Edges", fontsize=13)
    ax.set_ylabel("Peak GPU Memory (MB)", fontsize=13)
    ax.set_title("Memory Scalability: DegFuse-GAT vs PyG GAT", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "memory_scaling.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "memory_scaling.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved memory_scaling.png + .pdf")


def plot_latency_split() -> None:
    """Generate split tier1/tier2 latency figures."""
    files = sorted(glob.glob(os.path.join(METRICS_DIR, "benchmark_forward_*.json")))
    if not files:
        print("  [SKIP] No benchmark_forward_*.json files found")
        return

    TIER1_DATASETS = {"Cora", "CiteSeer", "PubMed"}
    tier1, tier2 = [], []
    for fp in files:
        with open(fp) as f:
            r = json.load(f)
        ds = r["dataset"]
        if not isinstance(r.get("pyg"), dict):
            continue
        pyg_ms = r["pyg"]["latency_ms"]
        auto = r.get("degfuse_auto", r.get("degfuse_da", {}))
        auto_ms = auto["latency_ms"] if isinstance(auto, dict) else 0
        fused = r.get("degfuse_fused", r.get("degfuse_sc", {}))
        fused_ms = fused["latency_ms"] if isinstance(fused, dict) else 0
        E = r.get("num_edges", 0)
        entry = {"dataset": ds, "edges": E, "pyg_ms": pyg_ms, "auto_ms": auto_ms, "fused_ms": fused_ms}
        if ds in TIER1_DATASETS:
            tier1.append(entry)
        else:
            tier2.append(entry)

    if tier1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_facecolor("#fafafa")
        names = [t["dataset"] for t in tier1]
        overhead_fused = [(t["fused_ms"] / t["pyg_ms"] - 1) * 100 for t in tier1]
        overhead_auto = [(t["auto_ms"] / t["pyg_ms"] - 1) * 100 for t in tier1]
        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w / 2, overhead_fused, w, label="DegFuse Fused", color="#e74c3c", alpha=0.85)
        ax.bar(x + w / 2, overhead_auto, w, label="DegFuse Auto", color="#3498db", alpha=0.85)
        ax.axhline(10, color="#7f8c8d", ls="--", lw=1.5, alpha=0.7, label="10% threshold")
        ax.axhline(0, color="black", lw=0.8)
        for i, (of, oa) in enumerate(zip(overhead_fused, overhead_auto)):
            ax.text(x[i] - w / 2, of + 1, f"{of:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.text(x[i] + w / 2, oa + 1, f"{oa:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=12)
        ax.set_ylabel("Latency Overhead vs PyG (%)", fontsize=12)
        ax.set_title("Tier 1: Small Graphs (E < 100k) — Latency Overhead", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "latency_tier1.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(FIGURES_DIR, "latency_tier1.pdf"), bbox_inches="tight")
        plt.close(fig)
        print("  Saved latency_tier1.png + .pdf")

    if tier2:
        fig, ax = plt.subplots(figsize=(max(6, len(tier2) * 3), 5))
        ax.set_facecolor("#fafafa")
        names = [f"{t['dataset']}\n({t['edges'] // 1000}k edges)" for t in tier2]
        speedup_fused = [t["pyg_ms"] / t["fused_ms"] if t["fused_ms"] > 0 else 0 for t in tier2]
        speedup_auto = [t["pyg_ms"] / t["auto_ms"] if t["auto_ms"] > 0 else 0 for t in tier2]
        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w / 2, speedup_fused, w, label="DegFuse Fused", color="#2ecc71", alpha=0.85)
        ax.bar(x + w / 2, speedup_auto, w, label="DegFuse Auto", color="#27ae60", alpha=0.85)
        ax.axhline(1.0, color="black", lw=1.5, ls="--", alpha=0.6)
        for i, (sf, sa) in enumerate(zip(speedup_fused, speedup_auto)):
            ax.text(x[i] - w / 2, sf + 0.02, f"{sf:.2f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.text(x[i] + w / 2, sa + 0.02, f"{sa:.2f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11)
        ax.set_ylabel("Speedup vs PyG", fontsize=12)
        ax.set_title("Tier 2: Large Graphs (E >= 100k) — DegFuse Speedup", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, max(max(speedup_fused), max(speedup_auto)) * 1.2)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "latency_tier2.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(FIGURES_DIR, "latency_tier2.pdf"), bbox_inches="tight")
        plt.close(fig)
        print("  Saved latency_tier2.png + .pdf")


def plot_crossover() -> None:
    """Generate crossover analysis dual-panel figure."""
    crossover_path = os.path.join(METRICS_DIR, "benchmark_crossover.json")
    if not os.path.exists(crossover_path):
        print("  [SKIP] benchmark_crossover.json not found")
        return

    with open(crossover_path) as f:
        d = json.load(f)

    data = d["data"]
    crossover = d.get("crossover_edges")
    edges = [r["edge_count"] for r in data]
    t_pyg = [r["pyg_ms"] for r in data]
    t_df = [r["degfuse_ms"] for r in data]
    speedups = [r["speedup"] for r in data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("DegFuse-GAT vs PyG: Latency & Speedup vs Graph Size",
                 fontsize=14, fontweight="bold")

    ax1.set_facecolor("#fafafa")
    ax1.plot(edges, t_pyg, "o-", color="#e74c3c", lw=2.5, ms=8, label="PyG GAT (baseline)")
    ax1.plot(edges, t_df, "s-", color="#2ecc71", lw=2.5, ms=8, label="DegFuse-GAT (ours)")

    if crossover:
        ax1.axvline(crossover, color="#7f8c8d", ls="--", lw=1.5, alpha=0.8)
        ax1.text(crossover * 1.08, ax1.get_ylim()[1] * 0.85,
                 f"Crossover\nE* ~ {crossover // 1000}k", fontsize=10,
                 color="#7f8c8d", fontweight="bold")
        ax1.axvspan(edges[0], crossover, alpha=0.05, color="#e74c3c",
                    label="PyG wins (launch overhead dominates)")
        ax1.axvspan(crossover, edges[-1], alpha=0.05, color="#2ecc71",
                    label="DegFuse wins (memory BW savings dominate)")

    ax1.set_xscale("log")
    ax1.set_ylabel("Forward Latency (ms)", fontsize=12)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.set_facecolor("#fafafa")
    colors = ["#2ecc71" if s > 1.0 else "#e74c3c" for s in speedups]
    ax2.bar(range(len(edges)), speedups, color=colors, alpha=0.85, width=0.6)
    ax2.axhline(1.0, color="black", lw=1.5, ls="--", alpha=0.6)
    ax2.set_xticks(range(len(edges)))
    ax2.set_xticklabels([f"{e // 1000}k" for e in edges], fontsize=9, rotation=30)
    ax2.set_ylabel("Speedup (PyG / DegFuse)", fontsize=11)
    ax2.set_ylim(0, max(speedups) * 1.2)
    for i, s in enumerate(speedups):
        ax2.text(i, s + 0.03, f"{s:.2f}x", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "crossover_analysis.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "crossover_analysis.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved crossover_analysis.png + .pdf")


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"Reading metrics from {METRICS_DIR}/")
    print(f"Writing figures to {FIGURES_DIR}/\n")

    plot_latency_comparison()
    plot_memory_scaling()
    plot_latency_split()
    plot_crossover()

    print(f"\nDone. Figures: {os.listdir(FIGURES_DIR)}")


if __name__ == "__main__":
    main()
