import torch
from torch import Tensor


class H5Dataset:
    def __init__(self, paths: list[str]):
        try:
            import h5py
        except ImportError as e:
            print("Please install h5py to use H5Dataset")
            raise e

        self.h5py = h5py
        self.files = [h5py.File(path, 'r') for path in paths]

        self.index = []
        for file_idx, file in enumerate(self.files):
            for key in file.keys():
                self.index.append((file_idx, key))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx) -> dict[str, Tensor]:
        file_idx, key = self.index[idx]
        dataset = self.files[file_idx][key]
        ids = torch.tensor(dataset[:])
        return {'input_ids': ids, 'labels': ids}

    def close(self):
        for f in self.files:
            f.close()

    def __del__(self):
        self.close()
