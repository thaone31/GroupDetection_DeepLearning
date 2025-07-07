
"""
GraphSAGE implementation using PyTorch Geometric.
- Two layers: first expands to 2x out_channels, then projects to out_channels.
- Forward pass applies ReLU after the first layer.
"""

import torch
from torch_geometric.nn import SAGEConv

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 2 * out_channels)
        self.conv2 = SAGEConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
