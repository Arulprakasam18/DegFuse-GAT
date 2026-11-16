"""
OOM Scalability Benchmark — The strongest figure in the paper.

Generates synthetic power-law (Zipf) graphs of increasing size.
Runs PyG GAT and DegFuse-GAT. Catches OOM errors.
Shows DegFuse continuing where PyG fails.

Run: python -m benchmarks.benchmark_oom
"""

import json
import os
import numpy as np
import torch


# Synthetic graph sizes: (label, num_nodes, avg_degree)
SCALES = [
    ("S1",   50_000,  20),
    ("S2",  100_000,  50),
    ("S3",  200_000, 100),
    ("S4",  500_000, 160),
    ("S5", 1_000_000, 200),
    ("S6", 2_000_000, 200),
    ("S7", 3_000_000, 200),
]


def generate_synthetic_graph(N: int, avg_degree: int, seed: int = 42, device="cuda"):
    """
    Generate a power-law graph using Zipf degree distribution.
    Returns (row_ptr, col_idx, edge_index, num_edges).
    """
    rng  = np.random.default_rng(seed)
    # Zipf(2.0) gives power-law degrees, clipped to avoid extreme hubs
    degs = np.clip(rng.zipf(2.0, N).astype(np.int64), 1, avg_degree * 20)

    row_ptr_np = np.zeros(N + 1, dtype=np.int64)
    cols_list  = []
    for i in range(N):
        d = int(degs[i])
        cols_list.append(rng.integers(0, N, size=d, dtype=np.int64))
        row_ptr_np[i + 1] = row_ptr_np[i] + d

    col_idx_np = np.concatenate(cols_list)
    E          = len(col_idx_np)
    src_np     = np.repeat(np.arange(N, dtype=np.int64), degs)

    row_ptr = torch.tensor(row_ptr_np, dtype=torch.long, device=device)
    col_idx = torch.tensor(col_idx_np, dtype=torch.long, device=device)
    ei      = torch.tensor(np.stack([src_np, col_idx_np]), dtype=torch.long, device=device)

    return row_ptr, col_idx, ei, E


def test_pyg(ei, x, nf, nc, device):
    from src.degfuse_gat.models.baseline_gat import BaselineGAT
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    try:
        m = BaselineGAT(nf, 64, nc, 8, 2, 0.0).to(device).eval()
        with torch.no_grad():
            m(x, ei)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        del m; torch.cuda.empty_cache()
        return peak
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "OOM"


def test_degfuse(rp, ci, x, nf, nc, device):
    from src.degfuse_gat.models.degfuse_gat_model import DegFuseGAT
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    try:
        m = DegFuseGAT(nf, 64, nc, 8, 2, 0.0, dispatch_mode="fused").to(device).eval()
        with torch.no_grad():
            m(x, rp, ci)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        del m; torch.cuda.empty_cache()
        return peak
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "OOM"


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    nf, nc  = 128, 40   # synthetic feature/class dims
    results = []

    for label, N, avg_deg in SCALES:
        print(f"\n[{label}] N={N:,}  avg_degree={avg_deg}")
        rp, ci, ei, E = generate_synthetic_graph(N, avg_deg, device=device)
        x = torch.randn(N, nf, device=device)
        print(f"  Generated: E={E:,}")

        pyg_mb = test_pyg(ei, x, nf, nc, device)
        df_mb  = test_degfuse(rp, ci, x, nf, nc, device)
        print(f"  PyG:      {pyg_mb}")
        print(f"  DegFuse:  {df_mb}")

        results.append({
            "scale":             label,
            "nodes":             N,
            "edges":             E,
            "avg_degree":        avg_deg,
            "pyg_memory_mb":     pyg_mb,
            "degfuse_memory_mb": df_mb,
        })

        del rp, ci, ei, x
        torch.cuda.empty_cache()

    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/benchmark_oom.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved → results/metrics/benchmark_oom.json")
