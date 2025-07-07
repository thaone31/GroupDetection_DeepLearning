import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx

class SimpleGraphTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.output_proj(x)
        return x

def graph_transformer_embedding(G, dim=64, hidden_dim=64, num_layers=2, n_heads=4, dropout=0.1, epochs=20, device=None):
    # Node features: degree as baseline
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    degrees = np.array([G.degree[n] for n in G.nodes()]).reshape(-1, 1)
    X = torch.tensor(degrees, dtype=torch.float32).to(device)
    model = SimpleGraphTransformer(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, n_heads=n_heads, dropout=dropout).to(device)
    # Autoencoder: encoder = transformer, decoder = linear
    decoder = nn.Linear(hidden_dim, 1).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=1e-3)
    model.train()
    decoder.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        encoded = model(X.unsqueeze(0)).squeeze(0)  # [N, hidden_dim]
        recon = decoder(encoded)  # [N, 1]
        loss = F.mse_loss(recon, X)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        emb = model(X.unsqueeze(0)).squeeze(0).cpu().numpy()
    return emb
