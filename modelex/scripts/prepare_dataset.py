import argparse
import importlib
from typing import Dict, List

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
    print(f'Creating instance of {class_path} with args: {args}')
    return cls(*args)

parser = argparse.ArgumentParser(description="write dataset into parquet")
parser.add_argument("--file", help="Path to save the dataset (required)", nargs=1)
parser.add_argument('--datasets', type=parse_class_args, help="List of class names with args, e.g., 'ClassName:arg1,arg2'")

def main(args) -> None:
    out_file = args.file[0]
    if args.datasets:
        datasets = [item for item in args.datasets.items()]
    else:
        raise ValueError('No datasets were provided')

    schema = pa.schema([('input_ids', pa.int32()), ('labels', pa.int32()), ])
    with pq.ParquetWriter(out_file, schema) as writer:
        for d in datasets:
            for item in tqdm(create_instance_from_string(d[0], *d[1])):
                table = pa.Table.from_pydict(item)
                writer.write_table(table)

if __name__ == '__main__':
    main(args=parser.parse_args())