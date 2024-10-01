import argparse
import importlib
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def parse_class_args(s: str) -> Dict[str, List[str]]:
    result = {}
    for class_info in s.split():
        class_name, *args = class_info.split(':')
        if args: args = args[0].split(',')
        else: args = []
        result[class_name] = args
    return result


def create_instance_from_string(class_path: str, *args):
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(*args)


def main(out_file: str, datasets: List[Tuple[str, List[str]]]) -> None:
    schema = pa.schema([('input_ids', pa.int64()), ('labels', pa.int64()),])
    with pq.ParquetWriter(out_file, schema) as writer:
        for d in datasets:
            for item in tqdm(create_instance_from_string(d[0], *d[1])):
                table = pa.Table.from_pydict(item)
                writer.write_table(table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="write dataset into parquet")
    parser.add_argument("--file", help="Path to save the dataset (required)", nargs=1)
    parser.add_argument('--datasets', type=parse_class_args, help="List of class names with args, e.g., 'ClassName:arg1,arg2'")
    args = parser.parse_args()
    out_file = args.file[0]
    if args.datasets:
        datasets = [item for item in args.datasets.items()]
        main(out_file=out_file, datasets=datasets)
    else:
        raise ValueError('No datasets were provided')
