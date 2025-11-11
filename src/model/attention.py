import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: [batch_size, seq_len, embed_dim]
        mask: optional, [batch_size, seq_len, seq_len]
        """
        d_k = Q.size(-1)  # embedding dimension
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Multiply by values
        output = torch.matmul(attn_weights, V)  # [batch_size, seq_len, embed_dim]
        return output, attn_weights

# ---------------- Test Attention with Embedding Output ----------------
if __name__ == "__main__":
    from src.config.model_config import Config
    from src.model.transformer import EmbeddingLayer

    # Initialize embedding layer
    embedding = EmbeddingLayer(Config.vocab_size, Config.embed_dim, Config.max_seq_len)
    
    # Dummy input: batch_size=2, seq_len=5
    sample_input = torch.randint(0, Config.vocab_size, (2, 5))
    embeds = embedding(sample_input)  # [2, 5, embed_dim]
    
    # Initialize attention
    attention = ScaledDotProductAttention()
    
    # For simplicity, use Q=K=V=embeds
    output, weights = attention(embeds, embeds, embeds)
    
    print("Embedding output shape:", embeds.shape)
    print("Attention output shape:", output.shape)
    print("Attention weights shape:", weights.shape)
