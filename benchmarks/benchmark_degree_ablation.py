"""
Degree-Aware Ablation Study (Table 4 in paper).

Compares:
  1. DegFuse degree-aware (multi-bucket, optimal per bucket)
  2. DegFuse single-config with each fixed config (1/2/4/8 warps)

Shows that degree-aware is the best overall strategy.

Run: python -m benchmarks.benchmark_degree_ablation --dataset Cora
"""

import argparse
import json
import os
import time
import torch

from src.degfuse_gat.utils.compat import apply_patches
apply_patches()


def bench_fn(fn, warmup: int = 20, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Cora",
                   choices=["Cora", "CiteSeer", "PubMed", "ogbn-arxiv"])
    p.add_argument("--warmup",  type=int, default=20)
    p.add_argument("--iters",   type=int, default=100)
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    from benchmarks.benchmark_forward import load_dataset
    from src.degfuse_gat.models.degfuse_gat_model import DegFuseGAT
    from src.degfuse_gat.models.csr_utils import edge_index_to_csr, print_csr_stats
    from src.degfuse_gat.models.degree_aware_scheduler import print_bucket_stats

    data, nf, nc = load_dataset(args.dataset, device)
    row_ptr, col_idx, _ = edge_index_to_csr(data.edge_index, data.num_nodes)
    print_csr_stats(row_ptr, args.dataset)
    print_bucket_stats(row_ptr)

    results = []

    # Degree-aware (multi-bucket)
    m = DegFuseGAT(nf, 64, nc, 8, 2, 0.0, dispatch_mode="fused").to(device).eval()
    t = bench_fn(lambda: m(data.x, row_ptr, col_idx), args.warmup, args.iters)
    results.append({"name": "degree_aware", "latency_ms": t})
    print(f"\n  Degree-Aware (multi-bucket):   {t:.3f} ms  ← best expected")
    del m; torch.cuda.empty_cache()

    # Single-config variants
    configs = [
        ("warps=1, BLOCK_E=32",  [(99_999_999, 1,  32)]),
        ("warps=2, BLOCK_E=64",  [(99_999_999, 2,  64)]),
        ("warps=4, BLOCK_E=128", [(99_999_999, 4, 128)]),
        ("warps=8, BLOCK_E=256", [(99_999_999, 8, 256)]),
    ]

    for label, buckets in configs:
        m = DegFuseGAT(nf, 64, nc, 8, 2, 0.0, dispatch_mode="fused").to(device).eval()
        t = bench_fn(
            lambda bkts=buckets: m(data.x, row_ptr, col_idx, bkts),
            args.warmup, args.iters
        )
        results.append({"name": label, "latency_ms": t})
        print(f"  Single-config {label}:  {t:.3f} ms")
        del m; torch.cuda.empty_cache()

    os.makedirs("results/metrics", exist_ok=True)
    fname = f"results/metrics/ablation_{args.dataset}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {fname}")
