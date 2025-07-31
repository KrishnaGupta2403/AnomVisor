import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output
        x = self.norm2(x)
        x = x + self.ff(x)
        return x

class AETransformer(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, latent_dim)
        )
        self.transformer = TransformerBlock(latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        z_trans = self.transformer(z.unsqueeze(1)).squeeze(1)
        x_recon = self.decoder(z_trans)
        return z_trans, x_recon

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            z_trans, x_recon = self.forward(x)
        return x_recon, z_trans
