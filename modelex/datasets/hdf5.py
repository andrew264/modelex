import h5py
import torch
from torch import Tensor

class H5Dataset:
    def __init__(self, hdf5_path, ):
        self.hdf5_file = h5py.File(hdf5_path, 'r')
        self.keys = list(self.hdf5_file.keys())
    def __len__(self): return len(self.keys)
    def __getitem__(self, idx) -> dict[str, Tensor]:
        ids = torch.tensor(self.hdf5_file[self.keys[idx]][:])
        return {'input_ids': ids, 'labels': ids}