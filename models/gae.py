import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from sklearn.cluster import KMeans
import numpy as np

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class GAE:
    def __init__(self, out_channels=16, feature_type="deepwalk", feature_dim=64, **kwargs):
        self.out_channels = out_channels
        self.embeddings = None
        self.feature_type = feature_type
        self.feature_dim = feature_dim
        self.kwargs = kwargs

    def fit(self, G, features=None, encoder_type="gcn"):
        data = from_networkx(G)
        if features is not None:
            x = torch.tensor(features, dtype=torch.float)
        else:
            x = torch.eye(G.number_of_nodes(), dtype=torch.float)
        data.x = x

        # Chọn encoder
        if encoder_type == "sage":
            from models.sage import SAGE as EncoderImpl
        else:
            EncoderImpl = Encoder

        encoder = EncoderImpl(in_channels=x.shape[1], out_channels=self.out_channels)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
        encoder.train()
        for epoch in range(300):
            optimizer.zero_grad()
            z = encoder(data.x, data.edge_index)
            adj_pred = torch.sigmoid(torch.matmul(z, z.t()))
            adj_true = torch.zeros_like(adj_pred)
            row, col = data.edge_index
            adj_true[row, col] = 1
            loss = torch.nn.functional.binary_cross_entropy(adj_pred, adj_true)
            loss.backward()
            optimizer.step()
        self.embeddings = encoder(data.x, data.edge_index).detach().cpu().numpy()

    def get_embedding(self):
        return self.embeddings

    def predict(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(self.embeddings)
        return kmeans.labels_