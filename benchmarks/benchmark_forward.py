"""
Forward Latency Benchmark (Table 2 in paper).

Measures per-forward-pass latency (ms) and throughput (edges/sec)
for PyG baseline, DegFuse single-config, and DegFuse degree-aware.

Run: python -m benchmarks.benchmark_forward --dataset Cora
"""

import argparse
import json
import os
import time
import torch

from src.degfuse_gat.utils.compat import apply_patches
apply_patches()


def load_dataset(name: str, device: torch.device):
    """Returns (data, num_features, num_classes)."""
    if name in ["Cora", "CiteSeer", "PubMed"]:
        from torch_geometric.datasets import Planetoid
        ds   = Planetoid(root="./data/Planetoid", name=name)
        data = ds[0].to(device)
        return data, ds.num_node_features, ds.num_classes

    elif name == "ogbn-arxiv":
        from ogb.nodeproppred import PygNodePropPredDataset
        ds   = PygNodePropPredDataset(name="ogbn-arxiv", root="./data/OGB")
        data = ds[0].to(device)
        data.y = data.y.squeeze()
        return data, data.x.shape[1], ds.num_classes

    elif name == "ogbn-products":
        from ogb.nodeproppred import PygNodePropPredDataset
        ds   = PygNodePropPredDataset(name="ogbn-products", root="./data/OGB")
        data = ds[0].to(device)
        data.y = data.y.squeeze()
        return data, data.x.shape[1], ds.num_classes

    elif name == "ogbn-proteins":
        from ogb.nodeproppred import PygNodePropPredDataset
        ds   = PygNodePropPredDataset(name="ogbn-proteins", root="./data/OGB")
        data = ds[0].to(device)
        # ogbn-proteins has no node features (x=None), only edge_attr [E,8]
        # Create dummy node features for latency/memory benchmarking
        nf = 8
        data.x = torch.randn(data.num_nodes, nf, device=device)
        nc = ds.num_classes if ds.num_classes else 112  # 112 binary labels
        data.y = data.y.squeeze()
        return data, nf, nc

    elif name == "Reddit":
        from torch_geometric.datasets import Reddit
        ds   = Reddit(root="./data/Reddit")
        data = ds[0].to(device)
        return data, data.x.shape[1], ds.num_classes

    else:
        raise ValueError(f"Unknown dataset: {name}")


def bench_fn(fn, warmup: int = 20, iters: int = 100) -> float:
    """Returns mean latency in milliseconds."""
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
    p.add_argument("--dataset",  default="Cora",
                   choices=["Cora", "CiteSeer", "PubMed",
                            "ogbn-arxiv", "ogbn-products", "ogbn-proteins", "Reddit"])
    p.add_argument("--hidden",   type=int, default=64)
    p.add_argument("--heads",    type=int, default=8)
    p.add_argument("--warmup",   type=int, default=20)
    p.add_argument("--iters",    type=int, default=100)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "This benchmark requires a CUDA GPU"

    data, nf, nc = load_dataset(args.dataset, device)
    E = data.num_edges

    from src.degfuse_gat.models.baseline_gat import BaselineGAT
    from src.degfuse_gat.models.degfuse_gat_model import DegFuseGAT
    from src.degfuse_gat.models.csr_utils import edge_index_to_csr, print_csr_stats

    row_ptr, col_idx, _ = edge_index_to_csr(data.edge_index, data.num_nodes)
    print_csr_stats(row_ptr, args.dataset)

    results = {"dataset": args.dataset, "num_edges": E}

    # ---- PyG baseline ----
    t1 = None
    try:
        m1 = BaselineGAT(nf, args.hidden, nc, args.heads, 2, 0.0).to(device).eval()
        with torch.no_grad():
            t1 = bench_fn(lambda: m1(data.x, data.edge_index), args.warmup, args.iters)
        results["pyg"] = {"latency_ms": t1, "throughput_medges_s": E / t1 / 1e3}
        print(f"  PyG baseline:      {t1:.3f} ms  |  {E/t1/1e3:.2f} Medges/s")
        del m1
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        results["pyg"] = "OOM"
        print(f"  PyG baseline:      OOM  ({e})")
    torch.cuda.empty_cache()

    # ---- DegFuse fused (always Triton kernel) ----
    t2 = None
    try:
        m2 = DegFuseGAT(nf, args.hidden, nc, args.heads, 2, 0.0, dispatch_mode="fused").to(device).eval()
        with torch.no_grad():
            t2 = bench_fn(lambda: m2(data.x, row_ptr, col_idx), args.warmup, args.iters)
        results["degfuse_fused"] = {"latency_ms": t2, "throughput_medges_s": E / t2 / 1e3}
        speedup_str = f"  |  speedup {t1/t2:.2f}x" if t1 else ""
        print(f"  DegFuse (fused):   {t2:.3f} ms  |  {E/t2/1e3:.2f} Medges/s{speedup_str}")
        del m2
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        results["degfuse_fused"] = "OOM"
        print(f"  DegFuse (fused):   OOM  ({e})")
    torch.cuda.empty_cache()

    # ---- DegFuse auto (adaptive dispatch) ----
    t3 = None
    try:
        m3 = DegFuseGAT(nf, args.hidden, nc, args.heads, 2, 0.0, dispatch_mode="auto").to(device).eval()
        with torch.no_grad():
            t3 = bench_fn(lambda: m3(data.x, row_ptr, col_idx, data.edge_index), args.warmup, args.iters)
        results["degfuse_auto"] = {"latency_ms": t3, "throughput_medges_s": E / t3 / 1e3}
        if t1:
            results["speedup_auto_vs_pyg"] = t1 / t3
        speedup_str = f"  |  speedup {t1/t3:.2f}x" if t1 else ""
        print(f"  DegFuse (auto):    {t3:.3f} ms  |  {E/t3/1e3:.2f} Medges/s{speedup_str}")
        del m3
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        results["degfuse_auto"] = "OOM"
        print(f"  DegFuse (auto):    OOM  ({e})")
    torch.cuda.empty_cache()

    os.makedirs("results/metrics", exist_ok=True)
    fname = f"results/metrics/benchmark_forward_{args.dataset}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {fname}")
