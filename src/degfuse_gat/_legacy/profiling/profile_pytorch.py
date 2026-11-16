"""
PyTorch Profiler — Forward pass kernel breakdown.

Generates TensorBoard-compatible trace files showing:
  - CUDA kernel timings (SDDMM, softmax, SpMM in baseline vs fused in ours)
  - Memory usage timeline
  - CPU/GPU sync points

Run: python -m profiling.profile_pytorch --dataset Cora
Then: tensorboard --logdir profiling/tb_logs/
"""

import argparse
import builtins
import os
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule

# Auto-answer OGB prompts: "y" for downloads, "n" for updates
_original_input = builtins.input
def _auto_input(prompt=""):
    p = str(prompt).lower()
    if "update" in p:
        return "n"
    return "y"
builtins.input = _auto_input

# Fix PyTorch 2.6+ weights_only default change breaking OGB
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Cora",
                   choices=["Cora", "CiteSeer", "PubMed", "ogbn-arxiv"])
    p.add_argument("--model",   default="both",
                   choices=["pyg", "degfuse", "both"])
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    from benchmarks.benchmark_forward import load_dataset
    from src.baseline_gat import BaselineGAT
    from src.degfuse_gat_model import DegFuseGAT
    from src.csr_utils import edge_index_to_csr

    data, nf, nc = load_dataset(args.dataset, device)
    row_ptr, col_idx, _ = edge_index_to_csr(data.edge_index, data.num_nodes)

    os.makedirs("profiling/tb_logs", exist_ok=True)

    def do_profile(name, forward_fn, log_dir):
        print(f"\nProfiling: {name}  →  {log_dir}")
        os.makedirs(log_dir, exist_ok=True)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=3, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for step in range(7):
                with record_function(f"{name}_forward"):
                    forward_fn()
                prof.step()

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        print(f"TensorBoard: tensorboard --logdir={log_dir}")

    if args.model in ["pyg", "both"]:
        m = BaselineGAT(nf, 64, nc, 8, 2, 0.0).to(device).eval()
        do_profile(
            "PyG-GAT",
            lambda: m(data.x, data.edge_index),
            f"profiling/tb_logs/pyg_{args.dataset}",
        )
        del m; torch.cuda.empty_cache()

    if args.model in ["degfuse", "both"]:
        m = DegFuseGAT(nf, 64, nc, 8, 2, 0.0, dispatch_mode="fused").to(device).eval()
        do_profile(
            "DegFuse-GAT",
            lambda: m(data.x, row_ptr, col_idx),
            f"profiling/tb_logs/degfuse_{args.dataset}",
        )
        del m; torch.cuda.empty_cache()
