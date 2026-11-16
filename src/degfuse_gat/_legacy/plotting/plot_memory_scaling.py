"""Paper Figure 1: Memory vs Edge Count (OOM scalability plot)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if __name__ == "__main__":
    if not os.path.exists("results/benchmark_oom.json"):
        print("Run benchmark_oom first: python -m benchmarks.benchmark_oom")
        exit(1)

    with open("results/benchmark_oom.json") as f:
        data = json.load(f)

    VRAM_LIMIT = 10 * 1024   # 10 GB in MB
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
    valid_df  = [(e, m) for e, m in zip(edges_all,  df_vals) if m is not None]

    if valid_pyg:
        ex, my = zip(*valid_pyg)
        ax.plot(ex, my, "o-", color="#e74c3c", lw=2.5, ms=8, label="PyG GAT (baseline)")
    if valid_df:
        ex, md = zip(*valid_df)
        ax.plot(ex, md, "s-", color="#2ecc71", lw=2.5, ms=8, label="DegFuse-GAT (ours)")

    for oe in oom_edges:
        ax.annotate(
            "OOM ✗", xy=(oe, VRAM_LIMIT * 1.05),
            fontsize=11, color="#e74c3c", ha="center", fontweight="bold"
        )

    ax.axhline(y=VRAM_LIMIT, color="gray", ls="--", alpha=0.6, lw=1.5, label="10 GB VRAM limit")
    ax.set_xscale("log")
    ax.set_xlabel("Number of Edges", fontsize=13)
    ax.set_ylabel("Peak GPU Memory (MB)", fontsize=13)
    ax.set_title("Memory Scalability: DegFuse-GAT vs PyG GAT", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    os.makedirs("figures", exist_ok=True)
    fig.tight_layout()
    fig.savefig("figures/memory_scaling.png", dpi=300, bbox_inches="tight")
    fig.savefig("figures/memory_scaling.pdf", bbox_inches="tight")
    print("Saved → figures/memory_scaling.png + .pdf")
