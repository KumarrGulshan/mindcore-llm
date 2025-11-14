import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    """
    This dataset:
    - Takes a long token list (entire corpus)
    - Splits it into chunks of fixed seq_len
    - Returns (input_ids, target_ids)
    """

    def __init__(self, token_ids, seq_len=32):
        super().__init__()
        self.seq_len = seq_len
        self.token_ids = token_ids

        # total number of sequences we can extract
        self.num_sequences = len(token_ids) - seq_len

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        # input: first 32 tokens
        input_ids = self.token_ids[index : index + self.seq_len]

        # target: same sequence but shifted by 1
        target_ids = self.token_ids[index + 1 : index + self.seq_len + 1]

        return torch.tensor(input_ids), torch.tensor(target_ids)


def create_dataloader(token_ids, seq_len=32, batch_size=4):
    """
    Returns a PyTorch DataLoader with batching enabled.
    """

    dataset = TextDataset(token_ids, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader


if __name__ == "__main__":
    # Demo run
    sample_tokens = [1,2,3,4,5,6,7,8,9,10] * 10  # fake long sequence

    loader = create_dataloader(sample_tokens, seq_len=5, batch_size=2)

    for x, y in loader:
        print("Input batch:\n", x)
        print("Target batch:\n", y)
        break
