import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

class AnomalyDetectionModel(nn.Module):
    def __init__(self, input_dim=10, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)

class SimpleTransformer(nn.Module):
    def __init__(self, dim=32, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z):
        z = z.unsqueeze(1)
        attn_out, _ = self.attn(z, z, z)
        return self.linear(attn_out.squeeze(1)).squeeze()