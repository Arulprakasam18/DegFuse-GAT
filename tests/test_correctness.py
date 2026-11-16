"""
Correctness tests for DegFuse-GAT.

Test 1: Online softmax variants match standard softmax reference
Test 2: Triton kernel output matches PyTorch reference implementation
Test 3: Degree-aware output matches single-config output

Run: python -m tests.test_correctness
"""

import sys
import torch


def test_online_softmax_variants():
    """Verify all three Python online softmax variants match standard softmax."""
    print("=" * 60)
    print("TEST 1: Online softmax Python reference variants")
    print("=" * 60)

    from src.degfuse_gat.models.online_softmax import (
        standard_softmax_aggregate,
        online_softmax_reference,
        online_softmax_tiled_reference,
    )

    torch.manual_seed(42)
    # Variable-length neighbors: [3, 2, 3, 2, 2] neighbors per node
    row_ptr = torch.tensor([0, 3, 5, 8, 10, 12], dtype=torch.long)
    scores  = torch.randn(12)
    values  = torch.randn(12, 16)

    ref  = standard_softmax_aggregate(scores, row_ptr, values)
    o1   = online_softmax_reference(scores, row_ptr, values)
    o2   = online_softmax_tiled_reference(scores, row_ptr, values, tile_size=2)
    o3   = online_softmax_tiled_reference(scores, row_ptr, values, tile_size=4)

    d1 = (ref - o1).abs().max().item()
    d2 = (ref - o2).abs().max().item()
    d3 = (ref - o3).abs().max().item()

    print(f"  online_softmax_reference       max_diff = {d1:.2e}")
    print(f"  online_softmax_tiled (tile=2)  max_diff = {d2:.2e}")
    print(f"  online_softmax_tiled (tile=4)  max_diff = {d3:.2e}")

    ok = d1 < 1e-5 and d2 < 1e-5 and d3 < 1e-5
    print(f"  Result: {'PASSED ✓' if ok else 'FAILED ✗'}")
    return ok


def test_triton_kernel():
    """Verify Triton fused kernel matches PyTorch reference on Cora."""
    print("=" * 60)
    print("TEST 2: Triton kernel correctness vs PyTorch reference")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  SKIPPED — no CUDA device found")
        return True

    try:
        import triton  # noqa: F401
    except ImportError:
        print("  SKIPPED — Triton not installed")
        return True

    from torch_geometric.datasets import Planetoid
    from src.degfuse_gat.models.csr_utils import edge_index_to_csr
    from src.degfuse_gat.models.degfuse_kernel import degfuse_gat_forward

    device = torch.device("cuda")
    ds   = Planetoid(root="./data/Planetoid", name="Cora")
    data = ds[0].to(device)
    row_ptr, col_idx, _ = edge_index_to_csr(data.edge_index, data.num_nodes)

    N  = data.num_nodes
    H  = 4
    D  = 16   # must be power of 2
    torch.manual_seed(42)

    Q = torch.randn(N, H, D, device=device, dtype=torch.float16)
    K = torch.randn(N, H, D, device=device, dtype=torch.float16)
    V = torch.randn(N, H, D, device=device, dtype=torch.float16)

    # Triton fused kernel
    out_triton, _ = degfuse_gat_forward(row_ptr, col_idx, Q, K, V, BLOCK_E=32)

    # PyTorch reference (float32)
    ref = torch.zeros(N, H, D, device=device, dtype=torch.float32)
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    for i in range(N):
        s, e = int(row_ptr[i].item()), int(row_ptr[i + 1].item())
        if s == e:
            continue
        j = col_idx[s:e]
        for h in range(H):
            sc = Kf[j, h, :] @ Qf[i, h, :]   # [deg]
            a  = torch.softmax(sc, dim=0)      # [deg]
            ref[i, h] = a @ Vf[j, h, :]       # [D]

    diff = (out_triton.float() - ref).abs().max().item()
    print(f"  Max absolute difference (triton vs reference): {diff:.2e}")
    # float16 kernel vs float32 reference — tolerance 5e-2 is appropriate
    ok = diff < 5e-2
    print(f"  Result: {'PASSED ✓' if ok else 'FAILED ✗'}")
    return ok


def test_degree_aware_vs_single_config():
    """Degree-aware and single-config kernels must produce the same output."""
    print("=" * 60)
    print("TEST 3: Degree-aware == single-config output")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  SKIPPED — no CUDA device found")
        return True

    try:
        import triton  # noqa: F401
    except ImportError:
        print("  SKIPPED — Triton not installed")
        return True

    from torch_geometric.datasets import Planetoid
    from src.degfuse_gat.models.csr_utils import edge_index_to_csr
    from src.degfuse_gat.models.degfuse_kernel import degfuse_gat_forward
    from src.degfuse_gat.models.degree_aware_scheduler import degfuse_gat_forward_degree_aware

    device = torch.device("cuda")
    ds   = Planetoid(root="./data/Planetoid", name="Cora")
    data = ds[0].to(device)
    row_ptr, col_idx, _ = edge_index_to_csr(data.edge_index, data.num_nodes)

    N, H, D = data.num_nodes, 4, 16
    torch.manual_seed(99)
    Q = torch.randn(N, H, D, device=device, dtype=torch.float16)
    K = torch.randn(N, H, D, device=device, dtype=torch.float16)
    V = torch.randn(N, H, D, device=device, dtype=torch.float16)

    out_sc, _  = degfuse_gat_forward(row_ptr, col_idx, Q, K, V)
    out_da     = degfuse_gat_forward_degree_aware(row_ptr, col_idx, Q, K, V)

    diff = (out_sc.float() - out_da.float()).abs().max().item()
    print(f"  Max absolute difference (single-config vs degree-aware): {diff:.2e}")
    ok = diff < 1e-2
    print(f"  Result: {'PASSED ✓' if ok else 'FAILED ✗'}")
    return ok


if __name__ == "__main__":
    results = []
    results.append(test_online_softmax_variants())
    print()
    results.append(test_triton_kernel())
    print()
    results.append(test_degree_aware_vs_single_config())
    print()
    all_ok = all(results)
    print("=" * 60)
    print(f"OVERALL: {'ALL TESTS PASSED ✓' if all_ok else 'SOME TESTS FAILED ✗'}")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)
