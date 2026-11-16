"""
DegFuse-GAT Model

Adaptive dispatch strategy:
  - Small graphs (E < edge_threshold): PyG GATConv fallback (faster on tiny graphs)
  - Medium graphs (50k <= E < 100k):   DegFuse 2-bucket config (memory + partial speedup)
  - Large graphs  (E >= 100k):         DegFuse 3-bucket config (memory + speedup)

Research contribution:
  PRIMARY:   Memory reduction 3-4x -> enables large-scale GAT
  SECONDARY: Speed improvement on E >= 100k edges
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from src.degfuse_gat.models.csr_utils import edge_index_to_csr, print_csr_stats
from src.degfuse_gat.models.degfuse_kernel import degfuse_gat_forward
from src.degfuse_gat.models.degree_aware_scheduler import (
    degfuse_gat_forward_degree_aware,
    print_bucket_stats,
    select_buckets,
)


class DegFuseGATLayer(nn.Module):
    """
    Single DegFuse-GAT layer with adaptive dispatch.

    dispatch_mode:
      "auto"   — select based on num_edges at runtime (recommended)
      "fused"  — always use Triton kernel (even on small graphs)
      "pyg"    — always use PyG GATConv (baseline comparison)
    """
    def __init__(
        self,
        in_features:   int,
        out_features:  int,
        num_heads:     int   = 8,
        dropout:       float = 0.6,
        dispatch_mode: str   = "auto",
        edge_threshold: int  = 100_000,
    ):
        super().__init__()
        self.H              = num_heads
        self.D              = out_features
        self.dropout        = dropout
        self.dispatch_mode  = dispatch_mode
        self.edge_threshold = edge_threshold

        self.W_q = nn.Linear(in_features, num_heads * out_features, bias=False)
        self.W_k = nn.Linear(in_features, num_heads * out_features, bias=False)
        self.W_v = nn.Linear(in_features, num_heads * out_features, bias=False)
        self.W_o = nn.Linear(num_heads * out_features, num_heads * out_features)

        for w in [self.W_q, self.W_k, self.W_v]:
            nn.init.xavier_uniform_(w.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.zeros_(self.W_o.bias)

    def _dispatch(
        self,
        x:       torch.Tensor,
        row_ptr: torch.Tensor,
        col_idx: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> str:
        """Decide which kernel path to use."""
        if self.dispatch_mode == "pyg":
            return "pyg"
        if self.dispatch_mode == "fused":
            return "fused"
        # auto mode
        E = int(row_ptr[-1].item())
        if E < self.edge_threshold and edge_index is not None:
            return "pyg"       # tiny graph: PyG is faster
        return "fused"         # medium/large: DegFuse wins

    def forward(
        self,
        x:          torch.Tensor,          # [N, in_features] float32
        row_ptr:    torch.Tensor,          # [N+1] int64
        col_idx:    torch.Tensor,          # [E]   int64
        edge_index: Optional[torch.Tensor] = None,  # needed for PyG fallback
        buckets:    Optional[List]         = None,
    ) -> torch.Tensor:
        N = x.shape[0]
        H, D = self.H, self.D
        scale = D ** -0.5
        E = int(row_ptr[-1].item())

        path = self._dispatch(x, row_ptr, col_idx, edge_index)

        if path == "pyg":
            # PyG fallback path — faster on tiny graphs
            from torch_geometric.nn import GATConv
            if not hasattr(self, "_pyg_conv"):
                self._pyg_conv = GATConv(
                    x.shape[1], D, heads=H,
                    dropout=self.dropout, concat=True
                ).to(x.device)
            return self._pyg_conv(x, edge_index)

        # Compute Q, K, V projections (always fp32 for autograd)
        Q_f32 = self.W_q(x).view(N, H, D).contiguous() * scale
        K_f32 = self.W_k(x).view(N, H, D).contiguous()
        V_f32 = self.W_v(x).view(N, H, D).contiguous()

        if self.training and self.dropout > 0:
            V_f32 = F.dropout(V_f32, p=self.dropout, training=True)

        if self.training:
            # ---- PyTorch sparse path for training (autograd-compatible) ----
            out = self._pytorch_sparse_forward(
                Q_f32, K_f32, V_f32, row_ptr, col_idx, N, H, D, E
            )
        else:
            # ---- Triton fused path for inference (fast, no autograd needed) ----
            Q_fp16 = Q_f32.half()
            K_fp16 = K_f32.half()
            V_fp16 = V_f32.half()

            D_pad = 1 << (D - 1).bit_length()
            if D_pad != D:
                pad_size = D_pad - D
                Q_fp16 = F.pad(Q_fp16, (0, pad_size))
                K_fp16 = F.pad(K_fp16, (0, pad_size))
                V_fp16 = F.pad(V_fp16, (0, pad_size))

            if buckets is None:
                buckets = select_buckets(E)

            out = degfuse_gat_forward_degree_aware(
                row_ptr, col_idx, Q_fp16, K_fp16, V_fp16, buckets, num_edges=E
            )

            if D_pad != D:
                out = out[:, :, :D]

            out = out.float()

        return self.W_o(out.view(N, H * D))

    def _pytorch_sparse_forward(self, Q, K, V, row_ptr, col_idx, N, H, D, E):
        """
        Pure PyTorch SDDMM+Softmax+SpMM — autograd-compatible.
        Uses CSR format with scatter ops for differentiable sparse attention.
        """
        # Expand row indices from CSR
        row_counts = row_ptr[1:] - row_ptr[:-1]  # [N]
        row_indices = torch.repeat_interleave(
            torch.arange(N, device=Q.device), row_counts
        )  # [E]

        # Gather Q for source nodes and K for target nodes
        Q_src = Q[row_indices]       # [E, H, D]
        K_dst = K[col_idx[:E]]       # [E, H, D]
        V_dst = V[col_idx[:E]]       # [E, H, D]

        # SDDMM: attention scores
        scores = (Q_src * K_dst).sum(dim=-1)  # [E, H]

        # Sparse softmax per source node
        # Subtract max per node for numerical stability
        scores_max = torch.zeros(N, H, device=Q.device, dtype=Q.dtype)
        scores_max.scatter_reduce_(
            0, row_indices.unsqueeze(1).expand(-1, H), scores, reduce="amax",
            include_self=True
        )
        scores = scores - scores_max[row_indices]
        alpha = torch.exp(scores)  # [E, H]

        # Normalize
        alpha_sum = torch.zeros(N, H, device=Q.device, dtype=Q.dtype)
        alpha_sum.scatter_add_(0, row_indices.unsqueeze(1).expand(-1, H), alpha)
        alpha = alpha / (alpha_sum[row_indices] + 1e-8)

        # SpMM: weighted sum of V
        weighted_v = V_dst * alpha.unsqueeze(-1)  # [E, H, D]
        out = torch.zeros(N, H, D, device=Q.device, dtype=Q.dtype)
        out.scatter_add_(
            0, row_indices.unsqueeze(1).unsqueeze(2).expand(-1, H, D), weighted_v
        )

        return out


class DegFuseGAT(nn.Module):
    """
    Multi-layer DegFuse-GAT for node classification.

    dispatch_mode:
      "auto"   — recommended: PyG on small, DegFuse on large
      "fused"  — always Triton (for memory benchmarks)
      "pyg"    — always PyG (baseline)
    """
    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        out_channels:    int,
        num_heads:       int   = 8,
        num_layers:      int   = 2,
        dropout:         float = 0.6,
        dispatch_mode:   str   = "auto",
        edge_threshold:  int   = 100_000,
    ):
        super().__init__()
        self.num_layers    = num_layers
        self.dropout       = dropout
        self.dispatch_mode = dispatch_mode
        self.layers        = nn.ModuleList()

        def make_layer(in_f, out_f, heads):
            return DegFuseGATLayer(
                in_f, out_f, heads, dropout,
                dispatch_mode, edge_threshold
            )

        if num_layers == 1:
            self.layers.append(make_layer(in_channels, out_channels, 1))
        else:
            self.layers.append(make_layer(in_channels, hidden_channels, num_heads))
            for _ in range(num_layers - 2):
                self.layers.append(
                    make_layer(num_heads * hidden_channels, hidden_channels, num_heads)
                )
            self.layers.append(
                make_layer(num_heads * hidden_channels, out_channels, 1)
            )

    def forward(
        self,
        x:          torch.Tensor,
        row_ptr:    torch.Tensor,
        col_idx:    torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        buckets:    Optional[List]         = None,
    ) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, row_ptr, col_idx, edge_index, buckets)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
