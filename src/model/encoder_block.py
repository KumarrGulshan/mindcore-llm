import torch
import torch.nn as nn
from src.model.multihead_attention import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1️⃣ Self-attention
        attn_out, _ = self.attn(x)
        x = x + self.dropout(attn_out)  # Residual connection
        x = self.norm1(x)               # Layer normalization

        # 2️⃣ Feed Forward
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)    # Residual connection
        x = self.norm2(x)               # Layer normalization
        return x


if __name__ == "__main__":
    batch_size, seq_len, embed_dim, num_heads, ff_dim = 2, 5, 512, 8, 2048
    x = torch.randn(batch_size, seq_len, embed_dim)
    block = EncoderBlock(embed_dim, num_heads, ff_dim)
    out = block(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
