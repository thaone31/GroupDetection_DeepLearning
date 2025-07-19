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

    def fit(self, G, features=None, encoder_type="gcn", labels=None, mask_label=None, alpha=1.0, beta=1.0):
        """
        labels: torch.LongTensor (N,) or None. Node labels, -1 for unlabeled.
        mask_label: torch.BoolTensor (N,) or None. True for labeled nodes.
        alpha, beta: weights for unsup and semi-supervised loss.
        """
        data = from_networkx(G)
        if features is not None:
            x = torch.tensor(features, dtype=torch.float)
        else:
            x = torch.eye(G.number_of_nodes(), dtype=torch.float)
        data.x = x

        # Only use GCN encoder
        encoder = Encoder(in_channels=x.shape[1], out_channels=self.out_channels)
        # Add classifier head for semi-supervised
        num_classes = None
        if labels is not None and (labels >= 0).sum() > 0:
            num_classes = int(labels.max().item() + 1)
            classifier = torch.nn.Linear(self.out_channels, num_classes).to(x.device)
            params = list(encoder.parameters()) + list(classifier.parameters())
        else:
            classifier = None
            params = encoder.parameters()
        optimizer = torch.optim.Adam(params, lr=0.01)
        encoder.train()
        if classifier is not None:
            classifier.train()
        for epoch in range(300):
            optimizer.zero_grad()
            z = encoder(data.x, data.edge_index)
            # Unsupervised loss: adjacency reconstruction
            adj_pred = torch.sigmoid(torch.matmul(z, z.t()))
            adj_true = torch.zeros_like(adj_pred)
            row, col = data.edge_index
            adj_true[row, col] = 1
            unsup_loss = torch.nn.functional.binary_cross_entropy(adj_pred, adj_true)
            # Semi-supervised loss (if label available)
            if classifier is not None and labels is not None and (labels >= 0).sum() > 0:
                mask = (labels >= 0) if mask_label is None else mask_label
                if mask.sum() > 0:
                    logits = classifier(z[mask])
                    target = labels[mask]
                    semi_loss = torch.nn.functional.cross_entropy(logits, target)
                else:
                    semi_loss = torch.tensor(0.0, device=x.device)
                loss = alpha * unsup_loss + beta * semi_loss
            else:
                loss = unsup_loss
            loss.backward()
            optimizer.step()
        self.embeddings = encoder(data.x, data.edge_index).detach().cpu().numpy()

    def get_embedding(self):
        return self.embeddings

    def predict(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(self.embeddings)
        return kmeans.labels_