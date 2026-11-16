"""
CSR Utility Functions for DegFuse-GAT.

Converts PyTorch Geometric edge_index (COO) to CSR (Compressed Sparse Row)
format, which is required by the Triton fused kernel.

CSR format:
  row_ptr : [N+1] int64  — row_ptr[i]:row_ptr[i+1] is the edge slice for node i
  col_idx : [E]   int64  — destination node IDs for each edge
"""

import torch
from typing import Optional, Tuple


def edge_index_to_csr(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_attr: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Convert PyG edge_index (COO) to CSR row_ptr + col_idx.

    Args:
        edge_index : [2, E] int64 — (src, dst) pairs
        num_nodes  : int — total number of nodes N
        edge_attr  : [E, *] optional edge features (reordered to match CSR)

    Returns:
        row_ptr : [N+1] int64 on same device as edge_index
        col_idx : [E]   int64
        edge_attr_sorted : reordered edge_attr (or None)
    """
    src, dst = edge_index[0], edge_index[1]
    E = src.shape[0]
    device = edge_index.device

    # Sort by source node for CSR row order
    sort_idx = torch.argsort(src, stable=True)
    src_sorted = src[sort_idx]
    dst_sorted = dst[sort_idx]

    # Degree per node (out-degree)
    deg = torch.zeros(num_nodes, dtype=torch.long, device=device)
    deg.scatter_add_(0, src_sorted, torch.ones(E, dtype=torch.long, device=device))

    # row_ptr via cumsum
    row_ptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    row_ptr[1:] = torch.cumsum(deg, dim=0)

    edge_attr_sorted = edge_attr[sort_idx] if edge_attr is not None else None

    return row_ptr, dst_sorted.contiguous(), edge_attr_sorted


def get_degrees(row_ptr: torch.Tensor) -> torch.Tensor:
    """Return per-node degree tensor from row_ptr. Shape [N]."""
    return (row_ptr[1:] - row_ptr[:-1]).contiguous()


def print_csr_stats(row_ptr: torch.Tensor, name: str = "Graph") -> None:
    """Print degree statistics for a CSR graph."""
    deg = get_degrees(row_ptr).float()
    N = deg.shape[0]
    E = int(row_ptr[-1].item())
    print(
        f"  [{name}] N={N:,}  E={E:,}  "
        f"deg min={deg.min().item():.0f}  max={deg.max().item():.0f}  "
        f"mean={deg.mean().item():.2f}  std={deg.std().item():.2f}"
    )
