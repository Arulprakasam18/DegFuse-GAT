"""
Baseline GAT using PyTorch Geometric GATConv.
Used as the comparison target in all benchmarks.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class BaselineGAT(torch.nn.Module):
    """
    Standard multi-layer GAT using PyG GATConv.

    Identical hyperparameters to DegFuseGAT to ensure fair comparison.
    Uses dense softmax over neighbor scores (no fusion).
    """
    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        out_channels:    int,
        num_heads:       int   = 8,
        num_layers:      int   = 2,
        dropout:         float = 0.6,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout    = dropout
        self.convs      = torch.nn.ModuleList()

        if num_layers == 1:
            self.convs.append(
                GATConv(in_channels, out_channels, heads=1,
                        dropout=dropout, concat=False)
            )
        else:
            self.convs.append(
                GATConv(in_channels, hidden_channels,
                        heads=num_heads, dropout=dropout)
            )
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(num_heads * hidden_channels, hidden_channels,
                            heads=num_heads, dropout=dropout)
                )
            self.convs.append(
                GATConv(num_heads * hidden_channels, out_channels,
                        heads=1, concat=False, dropout=dropout)
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
