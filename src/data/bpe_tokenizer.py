from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

class BPETokenizer:
    def __init__(self, vocab_path="data/vocab.json", merges_path="data/merges.txt"):
        self.vocab_path = vocab_path
        self.merges_path = merges_path

        if os.path.exists(vocab_path) and os.path.exists(merges_path):
            # Load saved tokenizer
            self.tokenizer = Tokenizer.from_file(vocab_path)
        else:
            self.tokenizer = None

    def train(self, corpus_path, vocab_size=8000):
        print("Training BPE tokenizer...")

        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]
        )

        tokenizer.train([corpus_path], trainer)

        # Save tokenizer
        tokenizer.model.save("data/")

        # Save full tokenizer.json (easier loading)
        tokenizer.save(self.vocab_path)

        self.tokenizer = tokenizer

        print(f"Tokenizer trained! Vocab size: {vocab_size}")

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)
    

if __name__ == "__main__":
        corpus = "data/samples/corpus.txt"
        tok = BPETokenizer()
        tok.train(corpus, vocab_size=8000)

        test = "Never stop learning new things."
        ids = tok.encode(test)
        print("Encoded:", ids)
        print("Decoded:", tok.decode(ids))
