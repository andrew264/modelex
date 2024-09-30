import argparse
import ast
import importlib
import msgpack

def parse_class_and_kwargs(class_string: str) -> tuple[str, dict]:
    if ':' in class_string:
        class_path, kwargs_str = class_string.split(':', 1)
        kwargs = ast.literal_eval(kwargs_str)
    else:
        class_path = class_string
        kwargs = {}
    return class_path, kwargs

def create_instance_from_string(class_path: str, **kwargs):
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**kwargs)

def main(out_file: str, datasets: list[tuple[str, dict]])->None:
    with open(out_file, 'wb') as f:
        for d in datasets:
            for item in create_instance_from_string(d[0], **d[1]):
                packed: bytes = msgpack.packb(item, use_bin_type=True)
                f.write(packed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="write dataset into messagepack")
    parser.add_argument("--file", help="Path to save the dataset (required)", nargs=1)
    parser.add_argument('--datasets', nargs='+', help="List of class names with optional kwargs, e.g., 'ClassName:{\"parameter\": value}'")
    args = parser.parse_args()
    out_file = args.file
    datasets = []
    if args.datasets:
        for dataset in args.datasets:
            datasets.append(parse_class_and_kwargs(dataset))
    else: raise ValueError('No datasets were provided')
    main(out_file=out_file, datasets=datasets)
