"""
Degree-Aware Kernel Scheduler

Nodes are partitioned into degree buckets with tuned (num_warps, BLOCK_E)
per bucket. Regime-aware bucket selection reduces kernel launch overhead
on smaller graphs.

Regimes:
  - Large  (E >= 100k): 3 buckets
  - Medium (50k <= E < 100k): 2 buckets
  - Single (E < 50k): 1 bucket
"""

import torch
import triton
import triton.language as tl
from typing import List, Dict, Optional

from src.degfuse_gat.models.csr_utils import get_degrees

# -----------------------------------------------------------------------
# Regime-aware bucket configurations
# -----------------------------------------------------------------------

# Large graphs (E >= 100k): 3 buckets, coarser thresholds
BUCKETS_LARGE = [
    (64,        1,   64),
    (256,       2,  128),
    (99_999_999, 4, 256),
]

# Medium graphs (50k <= E < 100k): 2 buckets
BUCKETS_MEDIUM = [
    (128,       2,   64),
    (99_999_999, 4, 128),
]

# Single-config fallback (E < 50k or adaptive disabled)
BUCKETS_SINGLE = [
    (99_999_999, 4,  64),
]

# Default (backward compat)
DEFAULT_BUCKETS = BUCKETS_LARGE


def select_buckets(num_edges: int) -> List:
    """
    Select bucket config based on graph size regime.
    This is the key fix: fewer launches on smaller graphs.
    """
    if num_edges >= 100_000:
        return BUCKETS_LARGE
    elif num_edges >= 50_000:
        return BUCKETS_MEDIUM
    else:
        return BUCKETS_SINGLE


# -----------------------------------------------------------------------
# Bucket kernel — same algorithm but with node_ids indirection
# -----------------------------------------------------------------------

@triton.jit
def _degfuse_bucket_kernel(
    row_ptr, col_idx,
    Q, K, V,
    Out, LSE,
    node_ids,
    N:       tl.constexpr,
    H:       tl.constexpr,
    D:       tl.constexpr,
    B:       tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_b >= B:
        return

    pid_n = tl.load(node_ids + pid_b)
    start = tl.load(row_ptr + pid_n)
    end   = tl.load(row_ptr + pid_n + 1)
    deg   = end - start

    d_offs = tl.arange(0, D)
    q_base = (pid_n * H + pid_h) * D
    q = tl.load(Q + q_base + d_offs, mask=d_offs < D, other=0.0).to(tl.float32)

    # Online softmax running state — ALL float32 (never float16)
    m = tl.full([], float("-inf"), dtype=tl.float32)
    l = tl.zeros([], dtype=tl.float32)
    o = tl.zeros([D], dtype=tl.float32)

    n_tiles = tl.cdiv(deg, BLOCK_E)
    for t in range(0, n_tiles):
        e_offs = tl.arange(0, BLOCK_E)
        e_idx  = start + t * BLOCK_E + e_offs
        mask_e = e_idx < end

        j = tl.load(col_idx + e_idx, mask=mask_e, other=0)

        kv_base = (j * H + pid_h) * D
        k_ptrs  = kv_base[:, None] + d_offs[None, :]
        v_ptrs  = kv_base[:, None] + d_offs[None, :]
        mk      = mask_e[:, None] & (d_offs[None, :] < D)

        k = tl.load(K + k_ptrs, mask=mk, other=0.0).to(tl.float32)
        v = tl.load(V + v_ptrs, mask=mk, other=0.0).to(tl.float32)

        scores = tl.sum(k * q[None, :], axis=1)
        scores = tl.where(mask_e, scores, float("-inf"))

        mb = tl.max(scores, axis=0)
        p  = tl.exp(scores - mb)
        p  = tl.where(mask_e, p, 0.0)
        lb = tl.sum(p, axis=0)
        ob = tl.sum(v * p[:, None], axis=0)

        m_new = tl.maximum(m, mb)
        alpha = tl.exp(m  - m_new)
        beta  = tl.exp(mb - m_new)
        l = l * alpha + lb * beta
        o = o * alpha + ob * beta
        m = m_new

    o = tl.where(l > 0, o / l, 0.0)
    out_base = (pid_n * H + pid_h) * D
    tl.store(Out + out_base + d_offs, o.to(tl.float16), mask=d_offs < D)
    tl.store(LSE + pid_n * H + pid_h, m + tl.log(l + 1e-8))


# -----------------------------------------------------------------------
# Python helpers
# -----------------------------------------------------------------------

def bucket_nodes_by_degree(
    row_ptr:  torch.Tensor,
    buckets:  Optional[List] = None,
    num_edges: Optional[int] = None,
) -> List[Dict]:
    """
    Partition nodes into degree buckets.
    If buckets=None, auto-selects regime based on num_edges.
    Skips empty buckets (no kernel launch overhead for them).
    """
    if buckets is None:
        E = int(row_ptr[-1].item()) if num_edges is None else num_edges
        buckets = select_buckets(E)

    buckets  = sorted(buckets, key=lambda x: x[0])
    degrees  = get_degrees(row_ptr)
    N        = degrees.shape[0]
    assigned = torch.zeros(N, dtype=torch.bool, device=row_ptr.device)
    result   = []

    for i, (max_deg, nw, be) in enumerate(buckets):
        if i == len(buckets) - 1:
            mask = ~assigned
        else:
            mask = (degrees <= max_deg) & (~assigned)

        nids = torch.where(mask)[0].contiguous()
        assigned[mask] = True

        if nids.numel() > 0:   # ← skip empty buckets (no launch overhead)
            result.append({
                "node_ids":   nids,
                "num_warps":  nw,
                "BLOCK_E":    be,
                "max_degree": max_deg,
                "count":      nids.numel(),
            })

    return result


def print_bucket_stats(
    row_ptr:   torch.Tensor,
    buckets:   Optional[List] = None,
    num_edges: Optional[int]  = None,
) -> None:
    E = int(row_ptr[-1].item()) if num_edges is None else num_edges
    regime = (
        "LARGE (3 buckets)"  if E >= 100_000 else
        "MEDIUM (2 buckets)" if E >= 50_000  else
        "SINGLE (1 bucket)"
    )
    print(f"  Bucket regime: {regime}  [E={E:,}]")
    for i, b in enumerate(bucket_nodes_by_degree(row_ptr, buckets, num_edges)):
        pct = 100.0 * b["count"] / (row_ptr.shape[0] - 1)
        print(
            f"    Bucket {i}: deg<={b['max_degree']:>9,} | "
            f"nodes={b['count']:>8,} ({pct:5.1f}%) | "
            f"num_warps={b['num_warps']} BLOCK_E={b['BLOCK_E']}"
        )


# -----------------------------------------------------------------------
# Degree-aware forward
# -----------------------------------------------------------------------

def degfuse_gat_forward_degree_aware(
    row_ptr:   torch.Tensor,
    col_idx:   torch.Tensor,
    Q:         torch.Tensor,
    K:         torch.Tensor,
    V:         torch.Tensor,
    buckets:   Optional[List] = None,
    num_edges: Optional[int]  = None,
) -> torch.Tensor:
    """
    Launch fused kernel once per non-empty degree bucket.
    Auto-selects bucket regime from graph size if buckets=None.

    Returns: Out [N, H, D] float16
    """
    N, H, D = Q.shape
    assert (D & (D - 1)) == 0, f"D={D} must be a power of 2"
    assert Q.dtype == torch.float16 and Q.is_contiguous()

    Out = torch.zeros(N, H, D, device=Q.device, dtype=Q.dtype)
    LSE = torch.empty(N, H, device=Q.device, dtype=torch.float32)

    for b in bucket_nodes_by_degree(row_ptr, buckets, num_edges):
        nids = b["node_ids"]
        B    = nids.shape[0]
        grid = (B, H)

        _degfuse_bucket_kernel[grid](
            row_ptr, col_idx, Q, K, V, Out, LSE, nids,
            N=N, H=H, D=D, B=B,
            BLOCK_E=b["BLOCK_E"],
            num_warps=b["num_warps"],
        )

    return Out
