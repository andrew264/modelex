import torch
from torch import Tensor
from typing import Union


class H5Dataset:
    def __init__(self, paths: Union[str, list[str]]):
        try:
            import h5py
        except ImportError as e:
            print("Please install h5py to use H5Dataset")
            raise e
        if isinstance(paths, str): paths = [paths]
        self.h5py = h5py
        self.files = [h5py.File(path, 'r') for path in paths]
        self.index = []
        for file_idx, file in enumerate(self.files):
            if 'input_ids' in file and 'labels' in file:
                num_rows = len(file['input_ids'])
                for row_idx in range(num_rows): self.index.append((file_idx, row_idx))
    def __len__(self): return len(self.index)
    def __getitem__(self, idx) -> dict[str, Tensor]:
        file_idx, row_idx = self.index[idx]
        file = self.files[file_idx]
        input_ids = torch.tensor(file["input_ids"][row_idx], dtype=torch.long)
        labels = torch.tensor(file["labels"][row_idx], dtype=torch.long)
        return {'input_ids': input_ids, 'labels': labels}
    def close(self):
        for f in self.files: f.close()
    def __del__(self): self.close()
