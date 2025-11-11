import torch
import torch.nn as nn
from src.model.multihead_attention import MultiHeadAttention

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        # Masked self-attention (decoder’s own words)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)

        # Encoder-decoder attention (focus on encoder output)
        self.enc_dec_attn = MultiHeadAttention(embed_dim, num_heads)

        # Feed Forward Network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        # Layer Norms & Dropouts
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output):
        # 1️⃣ Masked Self-Attention (decoder only attends to previous words)
        attn_out, _ = self.self_attn(x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # 2️⃣ Encoder-Decoder Attention (focus on encoder outputs)
        attn_out, _ = self.enc_dec_attn(x, enc_output, enc_output)
        x = x + self.dropout(attn_out)
        x = self.norm2(x)

        # 3️⃣ Feed Forward Network
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm3(x)

        return x


if __name__ == "__main__":
    batch_size, seq_len, embed_dim, num_heads, ff_dim = 2, 5, 512, 8, 2048

    x = torch.randn(batch_size, seq_len, embed_dim)
    enc_out = torch.randn(batch_size, seq_len, embed_dim)

    decoder_block = DecoderBlock(embed_dim, num_heads, ff_dim)
    out = decoder_block(x, enc_out)

    print("Decoder input shape:", x.shape)
    print("Encoder output shape:", enc_out.shape)
    print("Decoder output shape:", out.shape)
