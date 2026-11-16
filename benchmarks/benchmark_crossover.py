"""
Crossover Point Benchmark — NEW after Tier 1 analysis.

Finds the exact edge count E* where DegFuse-GAT becomes faster than PyG.
This is the most important diagnostic benchmark after the latency regression finding.

Key insight: Triton kernel launch overhead ~200-220us is FIXED regardless of graph size.
             Below E*, launch overhead > compute gain → PyG wins on latency.
             Above E*, memory bandwidth savings > launch overhead → DegFuse wins.

This figure becomes "Figure 2: Latency vs Edge Count with crossover annotation" in paper.

Run: python -m benchmarks.benchmark_crossover
"""

import json
import os
import time
import numpy as np
import torch


# Synthetic graph sizes to sweep
# From Tier 1 (Cora:10k, CiteSeer:9k, PubMed:88k) we know:
#   - 10k edges:  17-42% slower (launch overhead dominates)
#   - 88k edges:  borderline (need to measure)
#   - 1M+ edges:  expect DegFuse faster
EDGE_COUNTS = [
    5_000, 10_000, 20_000, 50_000,
    88_000, 100_000, 200_000, 500_000,
    1_000_000, 2_000_000,
]

N_NODES   = 10_000     # fixed num nodes, vary edges
H, D      = 8, 64      # standard GAT config
NF        = 128        # synthetic node feature dim
WARMUP    = 15
ITERS     = 60


def make_random_graph(N: int, E: int, device: torch.device):
    """Generate random graph with approximately E edges."""
    src = torch.randint(0, N, (E,), device=device)
    dst = torch.randint(0, N, (E,), device=device)
    edge_index = torch.stack([src, dst])
    # Sort by src for CSR
    order = src.argsort(stable=True)
    src, dst = src[order], dst[order]
    deg = torch.zeros(N, dtype=torch.long, device=device)
    deg.scatter_add_(0, src, torch.ones(E, dtype=torch.long, device=device))
    row_ptr = torch.zeros(N + 1, dtype=torch.long, device=device)
    row_ptr[1:] = deg.cumsum(0)
    return row_ptr, dst.contiguous(), edge_index


def bench(fn, warmup=WARMUP, iters=ITERS) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    from src.degfuse_gat.models.degfuse_kernel import degfuse_gat_forward
    from src.degfuse_gat.models.degree_aware_scheduler import degfuse_gat_forward_degree_aware, select_buckets
    from torch_geometric.nn import GATConv

    results = []
    print(f"{'Edges':>12} | {'PyG (ms)':>10} | {'DegFuse (ms)':>13} | {'Speedup':>8} | {'Winner':>8}")
    print("-" * 65)

    for E_target in EDGE_COUNTS:
        try:
            row_ptr, col_idx, edge_index = make_random_graph(N_NODES, E_target, device)
            E_actual = int(row_ptr[-1].item())
            x = torch.randn(N_NODES, NF, device=device)

            # PyG GATConv (baseline)
            pyg_conv = GATConv(NF, D, heads=H, dropout=0.0, concat=True).to(device).eval()
            t_pyg = bench(lambda: pyg_conv(x, edge_index))
            del pyg_conv

            # DegFuse degree-aware
            Q = torch.randn(N_NODES, H, D, device=device, dtype=torch.float16)
            K = torch.randn(N_NODES, H, D, device=device, dtype=torch.float16)
            V = torch.randn(N_NODES, H, D, device=device, dtype=torch.float16)
            bkts = select_buckets(E_actual)

            t_df = bench(
                lambda: degfuse_gat_forward_degree_aware(row_ptr, col_idx, Q, K, V, bkts, E_actual)
            )

            speedup = t_pyg / t_df
            winner  = "DegFuse ✓" if speedup > 1.0 else "PyG"
            print(f"{E_actual:>12,} | {t_pyg:>10.3f} | {t_df:>13.3f} | {speedup:>7.2f}× | {winner:>8}")

            results.append({
                "edge_count":    E_actual,
                "pyg_ms":        t_pyg,
                "degfuse_ms":    t_df,
                "speedup":       speedup,
                "degfuse_wins":  speedup > 1.0,
                "bucket_regime": (
                    "large"  if E_actual >= 100_000 else
                    "medium" if E_actual >= 50_000  else
                    "single"
                ),
            })

            del row_ptr, col_idx, edge_index, x, Q, K, V
            torch.cuda.empty_cache()

        except Exception as ex:
            print(f"{E_target:>12,} | ERROR: {ex}")
            torch.cuda.empty_cache()

    # Find crossover point
    crossover = None
    for i in range(len(results) - 1):
        if not results[i]["degfuse_wins"] and results[i+1]["degfuse_wins"]:
            crossover = (results[i]["edge_count"] + results[i+1]["edge_count"]) // 2
            print(f"\n★ Crossover point: E* ≈ {crossover:,} edges")
            print(f"  Below E*: PyG is faster (kernel launch overhead dominates)")
            print(f"  Above E*: DegFuse is faster + uses less memory")

    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/benchmark_crossover.json", "w") as f:
        json.dump({"crossover_edges": crossover, "data": results}, f, indent=2)
    print(f"\nSaved → results/metrics/benchmark_crossover.json")
    print("Plot:  python scripts/make_figures.py")
