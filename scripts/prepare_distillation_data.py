import argparse
import os
from typing import Any, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from llama_cpp import Llama
from tqdm import tqdm

from custom_data import ParquetCustomDataReader

def exists(x: Optional[Any]) -> bool: return x is not None

def get_logits(model: Llama, tokens: List[int]) -> List[np.ndarray]:
    model.reset()
    model.generate(tokens, top_k=-1, top_p=1, temp=0.).__next__()
    return [np.array(t, dtype=np.float32) for t in model.eval_logits]

def main(data: str, path: Optional[str], name: Optional[str], out: str, n_ctx: int) -> None:
    ### creating model
    kwargs = dict(logits_all=True, n_ctx=n_ctx, offload_kqv=False, n_threads=os.cpu_count()-1)
    if exists(path):
        model = Llama(model_path=path, **kwargs)
    else:
        model = Llama.from_pretrained(repo_id=name, filename="*Q8*.gguf", **kwargs)

    ### setup data
    dataset = ParquetCustomDataReader(data)

    schema = pa.schema([('input_ids', pa.int32()), ('labels', pa.int32()), ('teacher_logits', pa.list_(pa.float32()))])
    with pq.ParquetWriter(out, schema) as writer:
        for d in tqdm(dataset):
            logits = get_logits(model, d['input_ids'][:n_ctx])
            item = dict(input_ids=np.array(d['input_ids'], dtype=np.int32),
                        labels=np.array(d['labels'], dtype=np.int32),
                        teacher_logits=logits)
            table = pa.Table.from_pydict(item)
            writer.write_table(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train model")
    parser.add_argument("data", type=str, help="Path to the parquet dataset (required)")
    parser.add_argument("--path", type=str, default=None, help="Path to a GGUF model (optional)")
    parser.add_argument("--name", type=str, default=None, help="Huggingface GGUF Model Name (optional)")
    parser.add_argument("--out", type=str, default=None, help="Path to save the dataset (optional)")
    parser.add_argument("--n_ctx", type=int, default=8192, help="Max Context Length for the model (optional)")
    args = parser.parse_args()
    data: str = args.data
    path: Optional[str] = args.path
    name: Optional[str] = args.name
    out: Optional[str] = args.out
    if exists(path) == exists(name): raise ValueError("You must specify either a path 'path' or a name 'name', but not both.")
    if not exists(out): out = data.removesuffix('.parquet') + "-distil.parquet"
    main(data, path, name, out, args.n_ctx)