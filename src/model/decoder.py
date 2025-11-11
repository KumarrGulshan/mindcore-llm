import torch
import torch.nn as nn
from src.model.decoder_block import DecoderBlock
from src.model.transformer import EmbeddingLayer
from src.config.model_config import Config

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len=512):
        super().__init__()
        # Embedding layer for decoder input tokens
        self.embedding = EmbeddingLayer(vocab_size, embed_dim, max_len)

        # Stack of decoder blocks
        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

        # Final normalization layer
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tgt, enc_output):
        """
        tgt: target token IDs, shape [batch, seq_len]
        enc_output: output from encoder, shape [batch, seq_len, embed_dim]
        """
        # Convert target token IDs → embeddings
        x = self.embedding(tgt)

        # Pass through all decoder layers
        for layer in self.layers:
            x = layer(x, enc_output)

        # Final normalization
        x = self.norm(x)
        return x


if __name__ == "__main__":
    vocab_size = Config.vocab_size
    embed_dim = Config.embed_dim
    num_heads = Config.n_heads
    ff_dim = Config.ffn_dim
    num_layers = Config.n_layers
    max_len = Config.max_seq_len

    decoder = TransformerDecoder(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len)

    # Dummy input
    tgt = torch.randint(0, vocab_size, (2, 10))       # target sequence
    enc_out = torch.randn(2, 10, embed_dim)           # encoder output

    out = decoder(tgt, enc_out)

    print("Decoder input shape:", tgt.shape)
    print("Encoder output shape:", enc_out.shape)
    print("Decoder output shape:", out.shape)
