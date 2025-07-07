import torch
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.utils import from_networkx

def gat_embedding(G, dim=64, heads=2, epochs=100, device=None):
    """
    Simple GAT embedding for node features. If G has no features, use identity or random.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = from_networkx(G)
    n = G.number_of_nodes()
    if not hasattr(data, 'x') or data.x is None:
        # Always use random features of shape (n, dim) to avoid shape mismatch
        data.x = torch.randn((n, dim), dtype=torch.float32)
    in_channels = data.x.shape[1]
    class GAT(torch.nn.Module):
        def __init__(self, in_channels, out_channels, heads):
            super().__init__()
            self.gat1 = GATConv(in_channels, out_channels, heads=heads, concat=True)
            self.gat2 = GATConv(out_channels * heads, out_channels, heads=1, concat=False)
        def forward(self, x, edge_index):
            x = self.gat1(x, edge_index).relu()
            x = self.gat2(x, edge_index)
            return x
    model = GAT(in_channels, dim, heads).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        # Unsupervised: reconstruct features (shapes always [n, dim])
        loss = torch.nn.functional.mse_loss(out, data.x)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        emb = model(data.x, data.edge_index).cpu().numpy()
    return emb
