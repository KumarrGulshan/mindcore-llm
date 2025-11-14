# src/config.py

import torch

class Config:
    # Model architecture
    vocab_size = 23641
    max_seq_len = 128
    embed_dim = 512
    n_heads = 8
    n_layers = 6
    ffn_dim = 2048
    seq_len = 32
    dropout = 0.1

    # Training parameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 3e-4

    # File paths
    DATA_PATH = "data/samples/corpus.txt" 
    MODEL_PATH = "checkpoints/model.pt"
    VOCAB_PATH = "data/vocab.txt"

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
