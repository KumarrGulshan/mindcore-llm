# src/training/train_loop.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.dataset import create_dataloaders_from_corpus
from src.model.transformer_model import Transformer
from src.config.model_config import Config

class Trainer:
    def __init__(self):
        print("📦 Loading dataset...")
        # Load tokenizer + dataloaders
        self.tokenizer, self.train_loader, self.val_loader = create_dataloaders_from_corpus(
            corpus_path=Config.DATA_PATH,
            vocab_path=Config.TOKENIZER_PATH,
            seq_len=Config.seq_len,
            batch_size=Config.batch_size,
            val_split=0.05,
        )

        print("🚀 Initializing model...")
        self.model = Transformer(
            vocab_size = self.tokenizer.tokenizer.get_vocab_size(),  # use tokenizer vocab size
            embed_dim=Config.embed_dim,
            num_heads=Config.n_heads,
            ff_dim=Config.ffn_dim,
            num_encoder_layers=Config.n_layers,
            num_decoder_layers=Config.n_layers,
            max_len=Config.max_seq_len,
        ).to(Config.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.device = Config.device
        print(f"Trainer initialized. Using device: {self.device}\n")

    def train(self):
        print("🏋️ Starting training...")
        for epoch in range(Config.num_epochs):
            print(f"\n📅 Epoch {epoch+1}/{Config.num_epochs}")
            self.model.train()
            total_loss = 0

            for batch_idx, (x, y) in enumerate(tqdm(self.train_loader)):
                x, y = x.to(self.device), y.to(self.device)

                # Prepare decoder input: shift target by one with <bos> token
                bos_id = self.tokenizer.tokenizer.token_to_id("<bos>")
                if bos_id is None:
                 bos_id = 0

                decoder_input = torch.zeros_like(y)
                decoder_input[:, 1:] = y[:, :-1]
                decoder_input[:, 0] = bos_id

                # Forward pass
                logits = self.model(x, decoder_input)
                logits = logits.view(-1, logits.size(-1))
                y_flat = y.view(-1)

                # Compute loss
                loss = self.criterion(logits, y_flat)
                total_loss += loss.item()

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_idx % 50 == 0:
                    print(f"Batch {batch_idx} | Loss: {loss.item():.4f}")

            avg_train_loss = total_loss / len(self.train_loader)
            print(f"🔥 Epoch {epoch+1} Avg Train Loss: {avg_train_loss:.4f}")

            # Validation
            val_loss = self.evaluate()
            print(f"💻 Validation Loss: {val_loss:.4f} | Perplexity: {torch.exp(torch.tensor(val_loss)):.2f}")

            # Save checkpoint
            checkpoint_path = f"{Config.MODEL_PATH}_epoch{epoch+1}.pt"
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)

                bos_id = self.tokenizer.vocab.get("<bos>", 0)
                decoder_input = torch.zeros_like(y)
                decoder_input[:, 1:] = y[:, :-1]
                decoder_input[:, 0] = bos_id

                logits = self.model(x, decoder_input)
                logits = logits.view(-1, logits.size(-1))
                y_flat = y.view(-1)

                loss = self.criterion(logits, y_flat)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
