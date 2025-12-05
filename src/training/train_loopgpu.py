import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os  # ➤ ADDED

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
            vocab_size=self.tokenizer.tokenizer.get_vocab_size(),
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

        # ➤ ADDED — Auto-resume
        self.start_epoch = 0
        self._load_latest_checkpoint()

    # ➤ ADDED FUNCTION: Automatically loads last checkpoint
    def _load_latest_checkpoint(self):
        checkpoint_dir = "models/checkpoints"
        if not os.path.exists(checkpoint_dir):
            print("📁 No checkpoint directory found. Starting fresh.")
            return

        # find latest checkpoint file
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
        if not checkpoints:
            print("📁 No checkpoints found. Starting fresh.")
            return

        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.split("_epoch")[1].split(".")[0]))

        latest = checkpoints[-1]
        checkpoint_path = os.path.join(checkpoint_dir, latest)

        print(f"🔄 Loading latest checkpoint: {checkpoint_path}")

        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        # Extract epoch
        self.start_epoch = int(latest.split("_epoch")[1].split(".")[0])
        print(f"🔁 Resuming from epoch {self.start_epoch}")

    def train(self):
        print("🏋️ Starting training...")

        for epoch in range(self.start_epoch, Config.num_epochs):  # ➤ MODIFIED
            print(f"\n📅 Epoch {epoch+1}/{Config.num_epochs}")
            self.model.train()
            total_loss = 0

            for batch_idx, (x, y) in enumerate(tqdm(self.train_loader)):
                x, y = x.to(self.device), y.to(self.device)

                # Prepare decoder input: shift target by one with <bos> token
                try:
                    bos_id = self.tokenizer.tokenizer.token_to_id("<bos>")
                except AttributeError:
                    bos_id = self.tokenizer.tokenizer.encode("<bos>")[0]

                decoder_input = torch.zeros_like(y)
                decoder_input[:, 1:] = y[:, :-1]
                decoder_input[:, 0] = bos_id
                decoder_input = decoder_input.to(self.device)

                logits = self.model(x, decoder_input)
                logits = logits.view(-1, logits.size(-1))
                y_flat = y.view(-1)

                loss = self.criterion(logits, y_flat)
                total_loss += loss.item()

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
            checkpoint_path = f"models/checkpoints/model.pt_epoch{epoch+1}.pt"
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)

                try:
                    bos_id = self.tokenizer.tokenizer.token_to_id("<bos>")
                except AttributeError:
                    bos_id = self.tokenizer.tokenizer.encode("<bos>")[0]

                decoder_input = torch.zeros_like(y)
                decoder_input[:, 1:] = y[:, :-1]
                decoder_input[:, 0] = bos_id
                decoder_input = decoder_input.to(self.device)

                logits = self.model(x, decoder_input)
                logits = logits.view(-1, logits.size(-1))
                y_flat = y.view(-1)

                loss = self.criterion(logits, y_flat)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)


if __name__== "__main__":
    trainer = Trainer()
    trainer.train()