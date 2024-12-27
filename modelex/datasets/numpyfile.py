import numpy as np

class NumpyDataset:
    def __init__(self, path: str, dtype: np.dtype = np.uint16, seq_len: int = 1024):
        np_file = np.memmap(path, dtype=dtype, mode='r')
        n_samples = len(np_file) // seq_len
        self.np_file = np_file[:n_samples * seq_len].reshape(n_samples, seq_len)
    def __len__(self): return self.np_file.shape[0]
    def __getitem__(self, idx) -> dict:
        if idx < 0: idx += len(self)
        if idx >= len(self): raise IndexError("Index out of range")
        ids = np.int32(self.np_file[idx])
        return {'input_ids': ids, 'labels': ids}