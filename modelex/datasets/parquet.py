import glob
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from torch.utils.data import Dataset

class ParquetCustomDataReader(Dataset):
    def __init__(self, path: str) -> None:
        self.pq_file = pq.ParquetFile(path)
    def __len__(self) -> int:
        return self.pq_file.num_row_groups
    def __getitem__(self, idx) -> Dict[str, List[int]]:
        return self.pq_file.read_row_group(idx).to_pydict()
    def __iter__(self) -> Iterator[Dict[str, List[int]]]:
        for i in range(len(self)): yield self[i]

class ParquetTextDataset:
    def __init__(self, path: str, tokenizer_path: Optional[str] = None) -> None:
        self._tokenizer = Tokenizer.from_file(tokenizer_path) if tokenizer_path else None
        self.parquet_files = glob.glob(f"{path}/**/*.parquet", recursive=True)
        self.file_row_counts = []
        self.total_rows = 0
        self.cumulative_rows = [0]
        for file in self.parquet_files:
            parquet_file = pq.ParquetFile(file)
            file_row_count = parquet_file.metadata.num_rows
            self.file_row_counts.append(file_row_count)
            self.total_rows += file_row_count
            self.cumulative_rows.append(self.total_rows)
        self.parquet_cache = {}
    def __len__(self) -> int: return self.total_rows

    def __getitem__(self, idx: int) -> Union[Dict[str, np.ndarray], str]:
        if idx < 0: idx = self.total_rows + idx
        if not 0 <= idx < self.total_rows: raise IndexError("Index out of range")

        file_idx = self._binary_search(idx)
        relative_idx = idx - self.cumulative_rows[file_idx]

        parquet_file = self._get_parquet_file(self.parquet_files[file_idx])
        row_group_idx = relative_idx // parquet_file.metadata.row_group(0).num_rows
        row_group = parquet_file.read_row_group(row_group_idx)
        row_in_group = relative_idx % row_group.num_rows

        text: str = row_group.to_pandas().iloc[row_in_group].text
        if self._tokenizer is not None:
            tokenized = self._tokenizer.encode(text)
            ids = np.array(tokenized.ids, dtype=np.int32)
            return {'input_ids': ids, 'labels': ids}
        return text

    def _binary_search(self, index):
        left, right = 0, len(self.cumulative_rows) - 1
        while left < right:
            mid = (left + right) // 2
            if self.cumulative_rows[mid] <= index: left = mid + 1
            else: right = mid
        return left - 1

    def _get_parquet_file(self, file_path: str) -> pq.ParquetFile:
        if file_path not in self.parquet_cache: self.parquet_cache[file_path] = pq.ParquetFile(file_path)
        return self.parquet_cache[file_path]