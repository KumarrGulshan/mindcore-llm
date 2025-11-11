import torch
import torch.nn as nn
from src.model.encoder_block import EncoderBlock
from src.model.transformer import EmbeddingLayer
from src.config.model_config import Config

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len=512):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size, embed_dim, max_len)
        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Convert token IDs → embeddings
        x = self.embedding(x)

        # Pass through all encoder layers
        for layer in self.layers:
            x = layer(x)

        # Final layer normalization
        x = self.norm(x)
        return x


if __name__ == "__main__":
    vocab_size = Config.vocab_size
    embed_dim = Config.embed_dim
    num_heads = Config.n_heads
    ff_dim = Config.ffn_dim
    num_layers = Config.n_layers
    max_len = Config.max_seq_len

    encoder = TransformerEncoder(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len)
    
    # Dummy input (batch=2, seq=10)
    sample_input = torch.randint(0, vocab_size, (2, 10))
    output = encoder(sample_input)
    
    print("Input shape:", sample_input.shape)
    print("Output shape:", output.shape)
