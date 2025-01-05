import torch

class RandomDataset:
    def __init__(self, vocab_size: int, seq_len: int = 1024, num_codebooks: int = 0):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
    def __len__(self): return 1_000
    def __getitem__(self, idx) -> dict:
        if self.num_codebooks > 0: ids = torch.randint(0, self.vocab_size, (self.num_codebooks, self.seq_len))
        else: ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {'input_ids': ids, 'labels': ids}