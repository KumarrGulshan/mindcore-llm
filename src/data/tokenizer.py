class Tokenizer:
    def __init__(self, vocab=None):
        # Initialize with an optional existing vocabulary
        self.vocab = vocab or {}

    def build_vocab(self, texts):
        # Build vocab from list of text strings
        tokens = set(word for text in texts for word in text.lower().split())
        self.vocab = {word: idx for idx, word in enumerate(sorted(tokens))}
        print(f"✅ Vocabulary built with {len(self.vocab)} tokens.")

    def encode(self, text):
        # Convert sentence into token IDs
        return [self.vocab[word] for word in text.lower().split() if word in self.vocab]

    def decode(self, ids):
        # Convert token IDs back to text
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return " ".join([inv_vocab[i] for i in ids])


# -------------------------------------------------------
# 🧠 Test code: this part runs only if file is executed directly
# -------------------------------------------------------
if __name__ == "__main__":
    # 1️⃣ Read your corpus file
    with open("data/samples/corpus.txt", "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]

    # 2️⃣ Build vocab
    tokenizer = Tokenizer()
    tokenizer.build_vocab(texts)

    # 3️⃣ Test encoding/decoding
    sample = "Never stop learning new things in life."
    encoded = tokenizer.encode(sample)
    print("Encoded:", encoded)
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

    print("\n📊 Total vocab size:", len(tokenizer.vocab))
