import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.data.bpe_tokenizer import BPETokenizer
from src.config.model_config import Config

# ---------------------------
# Build long token list from corpus using BPE
# ---------------------------
def corpus_to_token_ids(bpe, corpus_path):
    token_ids = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            ids = bpe.encode(line)
            token_ids.extend(ids)

    return token_ids


# ---------------------------
# Dataset: sliding window sequence
# ---------------------------
class LMSequenceDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx):
        inp = self.token_ids[idx:idx + self.seq_len]
        tgt = self.token_ids[idx + 1:idx + self.seq_len + 1]
        return torch.tensor(inp), torch.tensor(tgt)


# ---------------------------
# Create train + val loaders
# ---------------------------
def create_dataloaders_from_corpus(
    corpus_path=Config.DATA_PATH,
    vocab_path=Config.TOKENIZER_PATH,
    merges_path=Config.MERGES_PATH,
    seq_len=Config.seq_len,
    batch_size=Config.batch_size,
    val_split=0.05,
):

    # Load BPE tokenizer
    bpe = BPETokenizer(vocab_path, merges_path)

    # Convert whole corpus to token ids
    token_ids = corpus_to_token_ids(bpe, corpus_path)

    dataset = LMSequenceDataset(token_ids, seq_len)

    total = len(dataset)
    val_count = int(total * val_split)
    train_count = total - val_count

    train_ds, val_ds = random_split(dataset, [train_count, val_count])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return bpe, train_loader, val_loader
