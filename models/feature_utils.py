import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from joblib import Parallel, delayed
from gensim.models import Word2Vec
import networkx as nx
import numpy as np
from node2vec import Node2Vec

class DeepWalkEnhancedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, num_layers=2, dropout=0.3, activation='relu'):
        super().__init__()
        layers = []
        dim_in = input_dim
        for i in range(num_layers):
            dim_out = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(nn.Linear(dim_in, dim_out))
            if i < num_layers - 1:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                else:
                    raise ValueError('Unsupported activation')
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

def deepwalk_enhanced_embedding(G, dim=64, walk_length=None, num_walks=None, window=5, workers=1,
                                hidden_dim=128, num_layers=2, dropout=0.3, activation='relu', epochs=50, batch_size=32, device=None):
    """
    DeepWalk embedding + MLP nonlinear (BatchNorm, Dropout).
    """
    # Step 1: DeepWalk embedding
    emb = deepwalk_embedding(G, dim=dim, walk_length=walk_length, num_walks=num_walks, window=window, workers=workers)
    # Step 2: MLP nonlinear
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = torch.tensor(emb, dtype=torch.float32).to(device)
    model = DeepWalkEnhancedMLP(input_dim=dim, hidden_dim=hidden_dim, output_dim=dim, num_layers=num_layers, dropout=dropout, activation=activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        for (batch_X,) in loader:
            out = model(batch_X)
            loss = F.mse_loss(out, batch_X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        enhanced_emb = model(X).cpu().numpy()
    return enhanced_emb

def node_deep_sum(G, dim=64, alpha=0.7, beta=0.3, **kwargs):
    """
    Weighted sum (average) of node2vec and deepwalk embeddings.
    alpha, beta: weights for node2vec and deepwalk (alpha + beta = 1)
    """
    node2vec_emb = node2vec_embedding(G, dim=dim, **kwargs)
    deepwalk_emb = deepwalk_embedding(G, dim=dim, **kwargs)
    emb_sum = alpha * node2vec_emb + beta * deepwalk_emb
    return emb_sum


def _single_walk(G, walk_length, start_node):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        neighbors = list(G.neighbors(cur))
        if neighbors:
            walk.append(np.random.choice(neighbors))
        else:
            break
    return [str(n) for n in walk]


def deepwalk_embedding(G, dim=64, walk_length=None, num_walks=None, window=5, workers=1):
    # Logic tự động hợp lý nhất:
    # - walk_length: tỉ lệ với log2(số node), nhưng không quá dài (giảm overfit, tăng đa dạng)
    # - num_walks: tỉ lệ với log2(số node), nhưng không quá nhiều (giảm trùng lặp, tiết kiệm compute)
    n_nodes = G.number_of_nodes()
    if walk_length is None:
        walk_length = max(10, min(40, int(np.log2(n_nodes + 1) * 6)))
    if num_walks is None:
        num_walks = max(5, min(20, int(np.log2(n_nodes + 1) * 2)))
    nodes = list(G.nodes())
    walks = Parallel(n_jobs=workers)(
        delayed(_single_walk)(G, walk_length, node)
        for _ in range(num_walks) for node in nodes
    )
    model = Word2Vec(walks, vector_size=dim, window=window, min_count=0, sg=1, workers=workers, epochs=1)
    emb = np.array([model.wv[str(n)] for n in G.nodes()])
    return emb
    
def node2vec_embedding(G, dim=64, walk_length=40, num_walks=10, window=5, p=1, q=1, workers=1):
    node2vec = Node2Vec(G, dimensions=dim, walk_length=walk_length, num_walks=num_walks, p=p, q=q, workers=workers)
    model = node2vec.fit(window=window, min_count=1)
    emb = np.array([model.wv[str(n)] for n in G.nodes()])
    return emb