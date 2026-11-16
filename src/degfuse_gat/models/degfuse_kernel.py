"""
DegFuse-GAT: Fused Triton Kernel (SDDMM + Online Softmax + SpMM)

Design decisions:
  - Q, K, V are float16; all accumulators (m, l, o) are float32
  - D (head_dim) must be a power of 2 for tl.arange(0, D)
  - BLOCK_E must be a power of 2 (Triton constexpr)
  - Grid: one Triton program per (node, head) pair
  - Online softmax: mathematically identical to standard softmax (no approx)
  - Intermediate attention score tensor S is NEVER written to global memory
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Core fused kernel (single-config, no degree-aware scheduling)
# ---------------------------------------------------------------------------

@triton.jit
def degfuse_gat_fwd_kernel(
    # Pointers
    row_ptr, col_idx,
    Q, K, V,
    Out, LSE,
    # Shapes (constexpr → compiled at launch time)
    N:      tl.constexpr,
    H:      tl.constexpr,
    D:      tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """
    One Triton program handles one (node, head) pair.

    Memory access pattern:
      Q[node, head, :]  — loaded once from global (coalesced across heads)
      K[neighbor, head, :] — gathered per-neighbor (irregular)
      V[neighbor, head, :] — gathered per-neighbor (irregular)
      Out[node, head, :] — stored once
    """
    pid_n = tl.program_id(0)   # node index
    pid_h = tl.program_id(1)   # head index

    if pid_n >= N:
        return

    # ---- Load CSR row range ----
    start = tl.load(row_ptr + pid_n)
    end   = tl.load(row_ptr + pid_n + 1)
    deg   = end - start

    # ---- Load query vector Q[node, head, :] ----
    d_offs = tl.arange(0, D)
    q_base = (pid_n * H + pid_h) * D
    q = tl.load(Q + q_base + d_offs, mask=d_offs < D, other=0.0).to(tl.float32)

    # ---- Online softmax running state ----
    m = tl.full([], float("-inf"), dtype=tl.float32)   # running max
    l = tl.zeros([], dtype=tl.float32)                  # running normalizer
    o = tl.zeros([D], dtype=tl.float32)                 # output accumulator

    # ---- Tile loop over neighbors ----
    n_tiles = tl.cdiv(deg, BLOCK_E)
    for t in range(0, n_tiles):
        e_offs = tl.arange(0, BLOCK_E)
        e_idx  = start + t * BLOCK_E + e_offs      # [BLOCK_E] global edge indices
        mask_e = e_idx < end                        # valid edge mask

        # Gather neighbor node IDs
        j = tl.load(col_idx + e_idx, mask=mask_e, other=0)  # [BLOCK_E]

        # Gather K[j, head, :] and V[j, head, :] via irregular gather
        # kv_base[b] = (j[b] * H + head) * D — start of feature for neighbor b
        kv_base = (j * H + pid_h) * D                        # [BLOCK_E]
        k_ptrs  = kv_base[:, None] + d_offs[None, :]         # [BLOCK_E, D]
        v_ptrs  = kv_base[:, None] + d_offs[None, :]         # [BLOCK_E, D]
        mk      = mask_e[:, None] & (d_offs[None, :] < D)    # [BLOCK_E, D]

        k = tl.load(K + k_ptrs, mask=mk, other=0.0).to(tl.float32)  # [BLOCK_E, D]
        v = tl.load(V + v_ptrs, mask=mk, other=0.0).to(tl.float32)  # [BLOCK_E, D]

        # ---- SDDMM: score = Q_i · K_j (dot-product attention) ----
        # scores[b] = sum_d( k[b,d] * q[d] )
        scores = tl.sum(k * q[None, :], axis=1)              # [BLOCK_E]
        scores = tl.where(mask_e, scores, float("-inf"))      # mask padding

        # ---- Tile-local softmax stats ----
        mb = tl.max(scores, axis=0)                           # scalar tile max
        p  = tl.exp(scores - mb)                              # [BLOCK_E]
        p  = tl.where(mask_e, p, 0.0)
        lb = tl.sum(p, axis=0)                                # scalar tile sum
        ob = tl.sum(v * p[:, None], axis=0)                  # [D] tile weighted sum

        # ---- Online merge with running state ----
        m_new = tl.maximum(m, mb)
        alpha = tl.exp(m  - m_new)   # rescale old state
        beta  = tl.exp(mb - m_new)   # rescale new tile
        l = l * alpha + lb * beta
        o = o * alpha + ob * beta
        m = m_new

    # ---- Normalize and store ----
    o = tl.where(l > 0, o / l, 0.0)

    out_base = (pid_n * H + pid_h) * D
    tl.store(Out + out_base + d_offs, o.to(tl.float16), mask=d_offs < D)

    # Store log-sum-exp for potential use in backward pass
    tl.store(LSE + pid_n * H + pid_h, m + tl.log(l + 1e-8))


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def degfuse_gat_forward(
    row_ptr:  torch.Tensor,   # [N+1] int64
    col_idx:  torch.Tensor,   # [E]   int64
    Q:        torch.Tensor,   # [N, H, D] float16
    K:        torch.Tensor,   # [N, H, D] float16
    V:        torch.Tensor,   # [N, H, D] float16
    BLOCK_E:  int = 128,
    num_warps: int = 4,
) -> tuple:
    """
    Launch the fused SDDMM+Softmax+SpMM Triton kernel.

    Returns:
        Out : [N, H, D] float16
        LSE : [N, H]    float32 (log-sum-exp per node-head pair)
    """
    assert Q.dtype == torch.float16, "Q must be float16"
    assert Q.is_contiguous(), "Q must be contiguous — call .contiguous() first"
    assert K.is_contiguous() and V.is_contiguous()

    N, H, D = Q.shape
    # D must be a power of 2 for tl.arange(0, D) in kernel
    assert (D & (D - 1)) == 0, f"D={D} must be a power of 2 (e.g. 8, 16, 32, 64)"

    Out = torch.empty_like(Q)
    LSE = torch.empty(N, H, device=Q.device, dtype=torch.float32)

    grid = (N, H)

    degfuse_gat_fwd_kernel[grid](
        row_ptr, col_idx, Q, K, V, Out, LSE,
        N=N, H=H, D=D,
        BLOCK_E=BLOCK_E,
        num_warps=num_warps,
    )

    return Out, LSE
