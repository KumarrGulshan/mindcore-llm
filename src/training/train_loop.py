import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.dataset_loader import create_dataloader
from src.data.tokenizer import Tokenizer
from src.model.transformer_model import Transformer
from src.config.model_config import Config


class Trainer:
    def __init__(self):
        print("🔧 Initializing Trainer...")

        # Device
        self.device = Config.device
        print(f"Using device: {self.device}")

        # Load vocab
        self.tokenizer = Tokenizer()
        self._load_vocab(Config.VOCAB_PATH)
        vocab_size = len(self.tokenizer.vocab)

        # Load data
        token_ids = self._load_corpus(Config.DATA_PATH)
        self.dataloader = create_dataloader(
            token_ids,
            seq_len=Config.seq_len,
            batch_size=Config.batch_size,
        )

        # Initialize Transformer
        self.model = Transformer(
            vocab_size=vocab_size,
            embed_dim=Config.embed_dim,
            num_heads=Config.n_heads,
            ff_dim=Config.ffn_dim,
            num_encoder_layers=Config.n_layers,
            num_decoder_layers=Config.n_layers,
            max_len=Config.max_seq_len,
        ).to(self.device)

        # Loss & optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.learning_rate)

    def _load_vocab(self, path):
        self.tokenizer.vocab = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                word, idx = line.strip().split("\t")
                self.tokenizer.vocab[word] = int(idx)
        print(f"📘 Loaded vocab with {len(self.tokenizer.vocab)} tokens.")

    def _load_corpus(self, path):
        token_ids = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                words = line.lower().split()
                ids = [self.tokenizer.vocab[w] for w in words if w in self.tokenizer.vocab]
                token_ids.extend(ids)
        print(f"📄 Total tokens in corpus: {len(token_ids)}")
        return token_ids

    def train(self):
        print("🚀 Starting training...")
        self.model.train()

        for epoch in range(Config.num_epochs):
            print(f"\n📅 Epoch {epoch+1}/{Config.num_epochs}")

            total_loss = 0
            for batch, (x, y) in enumerate(tqdm(self.dataloader)):

                x = x.to(self.device)  # Encoder input
                y = y.to(self.device)  # True target

                # Prepare decoder input (shifted right with BOS token)
                bos_id = self.tokenizer.vocab.get("<bos>", 0)
                decoder_input = torch.zeros_like(y)
                decoder_input[:, 1:] = y[:, :-1]
                decoder_input[:, 0] = bos_id

                # Forward pass
                logits = self.model(x, decoder_input)  # src=x, tgt=decoder_input
                logits = logits.view(-1, logits.size(-1))
                y_flat = y.view(-1)

                # Compute loss
                loss = self.loss_fn(logits, y_flat)

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if batch % 50 == 0:
                    print(f"Batch {batch} | Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(self.dataloader)
            print(f"🔥 Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

            # Save checkpoint
            checkpoint_path = f"{Config.MODEL_PATH}_epoch{epoch+1}.pt"
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"💾 Saved model checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
