import torch
import torch.nn as nn
from src.config.model_config import Config

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len=512):
        super().__init__()
        # Converts token IDs → dense embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Adds position information
        self.position_embedding = nn.Embedding(max_len, embed_dim)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        token_embeds = self.token_embedding(x)
        position_embeds = self.position_embedding(positions)
        return token_embeds + position_embeds


if __name__ == "__main__":
    # Load settings from Config
    vocab_size = Config.vocab_size
    embed_dim = Config.embed_dim
    max_seq_len = Config.max_seq_len

    # Initialize the embedding layer
    embedding = EmbeddingLayer(vocab_size, embed_dim, max_seq_len)

    # Create a sample batch (batch_size = 2, seq_len = 10)
    sample_input = torch.randint(0, vocab_size, (2, 10))
    output = embedding(sample_input)

    print("Input shape:", sample_input.shape)
    print("Output shape:", output.shape)