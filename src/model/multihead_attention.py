import torch
import torch.nn as nn
from src.model.attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attention = ScaledDotProductAttention()

    def forward(self, query, key=None, value=None):
        # If no key/value are provided, it's self-attention
        if key is None and value is None:
            key, value = query, query

        batch_size = query.size(0)

        # Linear projections
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply attention
        out, attn = self.attention(Q, K, V)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # Final projection
        out = self.out_proj(out)
        return out, attn
