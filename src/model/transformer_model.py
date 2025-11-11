import torch
import torch.nn as nn
from src.model.encoder import TransformerEncoder
from src.model.decoder import TransformerDecoder
from src.config.model_config import Config

class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_encoder_layers, num_decoder_layers, max_len=512):
        super().__init__()
        # Encoder
        self.encoder = TransformerEncoder(vocab_size, embed_dim, num_heads, ff_dim, num_encoder_layers, max_len)

        # Decoder
        self.decoder = TransformerDecoder(vocab_size, embed_dim, num_heads, ff_dim, num_decoder_layers, max_len)

        # Final linear layer to produce logits over vocab
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt):
        # 1️⃣ Encode source sequence
        enc_output = self.encoder(src)

        # 2️⃣ Decode target sequence
        dec_output = self.decoder(tgt, enc_output)

        # 3️⃣ Produce logits for each token
        logits = self.fc_out(dec_output)
        return logits


if __name__ == "__main__":
    vocab_size = Config.vocab_size
    embed_dim = Config.embed_dim
    num_heads = Config.n_heads
    ff_dim = Config.ffn_dim
    num_encoder_layers = Config.n_layers
    num_decoder_layers = Config.n_layers
    max_len = Config.max_seq_len

    model = Transformer(
        vocab_size, embed_dim, num_heads, ff_dim,
        num_encoder_layers, num_decoder_layers, max_len
    )

    # Dummy input
    src = torch.randint(0, vocab_size, (2, 10))  # encoder input
    tgt = torch.randint(0, vocab_size, (2, 10))  # decoder input
    out = model(src, tgt)

    print("Source input shape:", src.shape)
    print("Target input shape:", tgt.shape)
    print("Output logits shape:", out.shape)
