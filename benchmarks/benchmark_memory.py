"""
Peak VRAM Benchmark (Table 3 in paper).

Run: python -m benchmarks.benchmark_memory --dataset Cora
"""

import argparse
import json
import os
import torch

from src.degfuse_gat.utils.compat import apply_patches
apply_patches()


def measure_peak_mb(fn, device) -> float:
    """Run fn once, return peak VRAM in MB."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Cora",
                   choices=["Cora", "CiteSeer", "PubMed",
                            "ogbn-arxiv", "ogbn-products", "ogbn-proteins", "Reddit"])
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    from benchmarks.benchmark_forward import load_dataset
    from src.degfuse_gat.models.baseline_gat import BaselineGAT
    from src.degfuse_gat.models.degfuse_gat_model import DegFuseGAT
    from src.degfuse_gat.models.csr_utils import edge_index_to_csr

    data, nf, nc = load_dataset(args.dataset, device)
    row_ptr, col_idx, _ = edge_index_to_csr(data.edge_index, data.num_nodes)
    results = {"dataset": args.dataset}

    # PyG
    try:
        m = BaselineGAT(nf, 64, nc, 8, 2, 0.0).to(device).eval()
        with torch.no_grad():
            mb = measure_peak_mb(lambda: m(data.x, data.edge_index), device)
        results["pyg"] = {"peak_mb": mb}
        print(f"  PyG baseline:   {mb:.1f} MB")
        del m
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        results["pyg"] = "OOM"
        print(f"  PyG baseline:   OOM  ({e})")
    torch.cuda.empty_cache()

    # DegFuse degree-aware
    try:
        m = DegFuseGAT(nf, 64, nc, 8, 2, 0.0, dispatch_mode="fused").to(device).eval()
        with torch.no_grad():
            mb = measure_peak_mb(lambda: m(data.x, row_ptr, col_idx), device)
        results["degfuse_da"] = {"peak_mb": mb}
        print(f"  DegFuse-GAT:    {mb:.1f} MB")
        del m
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        results["degfuse_da"] = "OOM"
        print(f"  DegFuse-GAT:    OOM  ({e})")
    torch.cuda.empty_cache()

    os.makedirs("results/metrics", exist_ok=True)
    fname = f"results/metrics/benchmark_memory_{args.dataset}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved → {fname}")
