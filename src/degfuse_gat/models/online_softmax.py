"""
Reference Python implementations of online softmax for graph neighborhoods.

These are used ONLY for correctness verification of the Triton kernel.
They are NOT used in training (too slow).

Three variants are provided:
  1. standard_softmax_aggregate  — baseline two-pass softmax
  2. online_softmax_reference    — scalar one-pass online softmax
  3. online_softmax_tiled_reference — tiled one-pass (matches Triton tile logic)
"""

import torch
from typing import Tuple


def standard_softmax_aggregate(
    scores: torch.Tensor,   # [E] float — attention scores per edge
    row_ptr: torch.Tensor,  # [N+1] int64
    values: torch.Tensor,   # [E, D] float — neighbor feature vectors
) -> torch.Tensor:
    """
    Two-pass standard softmax + aggregation.
    This is the mathematically correct reference.
    Output shape: [N, D]
    """
    N = row_ptr.shape[0] - 1
    D = values.shape[1]
    out = torch.zeros(N, D, device=scores.device, dtype=values.dtype)

    for i in range(N):
        s, e = int(row_ptr[i].item()), int(row_ptr[i + 1].item())
        if s == e:
            continue
        sc = scores[s:e].float()
        v  = values[s:e].float()
        attn = torch.softmax(sc, dim=0)
        out[i] = (attn.unsqueeze(1) * v).sum(0).to(values.dtype)

    return out


def online_softmax_reference(
    scores: torch.Tensor,
    row_ptr: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """
    One-pass scalar online softmax (Algorithm 1 from FlashAttention paper).
    Processes one edge at a time — matches Triton tile merge logic.
    Output shape: [N, D]
    """
    N = row_ptr.shape[0] - 1
    D = values.shape[1]
    out = torch.zeros(N, D, device=scores.device, dtype=values.dtype)

    for i in range(N):
        s, e = int(row_ptr[i].item()), int(row_ptr[i + 1].item())
        if s == e:
            continue

        m = float("-inf")   # running max
        l = 0.0             # running sum of exp
        o = torch.zeros(D, device=scores.device, dtype=torch.float32)

        for idx in range(s, e):
            sc  = float(scores[idx].item())
            v   = values[idx].float()
            m_new     = max(m, sc)
            alpha     = torch.exp(torch.tensor(m - m_new,  dtype=torch.float32))
            beta      = torch.exp(torch.tensor(sc - m_new, dtype=torch.float32))
            l = l * alpha.item() + beta.item()
            o = o * alpha + v * beta
            m = m_new

        out[i] = (o / l).to(values.dtype)

    return out


def online_softmax_tiled_reference(
    scores: torch.Tensor,
    row_ptr: torch.Tensor,
    values: torch.Tensor,
    tile_size: int = 32,
) -> torch.Tensor:
    """
    Tiled online softmax — closely mirrors the Triton kernel tile loop.
    Each tile processes `tile_size` neighbors and merges with running state.
    Output shape: [N, D]
    """
    N = row_ptr.shape[0] - 1
    D = values.shape[1]
    out = torch.zeros(N, D, device=scores.device, dtype=values.dtype)

    for i in range(N):
        s, e = int(row_ptr[i].item()), int(row_ptr[i + 1].item())
        if s == e:
            continue

        m = float("-inf")
        l = 0.0
        o = torch.zeros(D, device=scores.device, dtype=torch.float32)

        for ts in range(s, e, tile_size):
            te   = min(ts + tile_size, e)
            t_sc = scores[ts:te].float()
            t_v  = values[ts:te].float()

            # Tile-local max and softmax
            mb  = t_sc.max().item()
            p   = torch.exp(t_sc - mb)         # [tile]
            lb  = p.sum().item()
            ob  = (p.unsqueeze(1) * t_v).sum(0)  # [D]

            # Merge tile into running state
            m_new = max(m, mb)
            alpha = torch.exp(torch.tensor(m  - m_new, dtype=torch.float32))
            beta  = torch.exp(torch.tensor(mb - m_new, dtype=torch.float32))
            l = l * alpha.item() + lb * beta.item()
            o = o * alpha        + ob * beta
            m = m_new

        out[i] = (o / l).to(values.dtype)

    return out
