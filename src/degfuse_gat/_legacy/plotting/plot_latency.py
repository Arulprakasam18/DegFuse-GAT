"""Paper Figure: Forward Latency Comparison Bar Chart."""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    files = sorted(glob.glob("results/benchmark_forward_*.json"))
    if not files:
        print("Run benchmark_forward first for each dataset.")
        exit(1)

    datasets, t_pyg, t_sc, t_da = [], [], [], []
    for fp in files:
        with open(fp) as f:
            r = json.load(f)
        # Skip datasets where PyG OOM'd (value is "OOM" string, not a dict)
        if not isinstance(r.get("pyg"), dict):
            print(f"  Skipping {r['dataset']} (PyG OOM)")
            continue
        datasets.append(r["dataset"])
        t_pyg.append(r["pyg"]["latency_ms"])
        fused = r.get("degfuse_fused", r.get("degfuse_sc", r.get("degfuse_single_config", {})))
        t_sc.append(fused["latency_ms"] if isinstance(fused, dict) else 0)
        auto = r.get("degfuse_auto", r.get("degfuse_da", {}))
        t_da.append(auto["latency_ms"] if isinstance(auto, dict) else 0)

    x = np.arange(len(datasets))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 3), 6))
    ax.set_facecolor("#fafafa")

    b1 = ax.bar(x - w, t_pyg, w, label="PyG GAT",        color="#e74c3c", alpha=0.88)
    b2 = ax.bar(x,     t_sc,  w, label="DegFuse-SC",     color="#3498db", alpha=0.88)
    b3 = ax.bar(x + w, t_da,  w, label="DegFuse-DA (ours)", color="#2ecc71", alpha=0.88)

    # Speedup labels above each DegFuse-DA bar
    for i, (tp, td) in enumerate(zip(t_pyg, t_da)):
        if td > 0:
            ax.text(x[i] + w, td + 0.1, f"{tp/td:.1f}×",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1a6e3c")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylabel("Forward Latency (ms)", fontsize=13)
    ax.set_title("Forward Pass Latency: DegFuse-GAT vs Baselines", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    os.makedirs("figures", exist_ok=True)
    fig.tight_layout()
    fig.savefig("figures/latency_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig("figures/latency_comparison.pdf", bbox_inches="tight")
    print("Saved → figures/latency_comparison.png + .pdf")
