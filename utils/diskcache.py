import shutil
import struct
from collections import OrderedDict
from typing import List, Union
import torch
import tempfile
import os
import hashlib
import functools
import atexit
import inspect

def tensor_cache(max_cache_size: Union[int, float] = 5):
    max_cache_size_mib = max_cache_size * 1024
    cache = OrderedDict()
    total_cache_size = 0
    # temp_dir = tempfile.mkdtemp()
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".tensor_cache")
    os.makedirs(cache_dir, exist_ok=True)
    def cleanup_cache():
        shutil.rmtree(cache_dir)

    atexit.register(cleanup_cache)

    def generate_cache_key(input_data: Union[torch.Tensor, List[int]]) -> str:
        if isinstance(input_data, torch.Tensor):
            data_bytes = input_data.numpy().tobytes()
        elif isinstance(input_data, list) and all(isinstance(i, int) for i in input_data):
            data_bytes = struct.pack(f"{len(input_data)}i", *input_data)
        else:
            raise TypeError("Input must be a torch.Tensor or List[int]")

        return hashlib.sha256(data_bytes).hexdigest() + ".pt"

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal total_cache_size

            # Detect if we're in a class method (self is the first argument)
            is_method = inspect.signature(func).parameters.get("self") is not None
            if is_method:
                self, input_data = args[0], args[1]
            else:
                input_data = args[0]

            # Generate cache key
            cache_key = generate_cache_key(input_data)
            cache_path = os.path.join(cache_dir, cache_key)

            if cache_key in cache:
                cache.move_to_end(cache_key)
                return torch.load(cache[cache_key], weights_only=True)

            if os.path.exists(cache_path):
                output = torch.load(cache_path, weights_only=True)
                cache[cache_key] = cache_path
                cache.move_to_end(cache_key)
                return output

            # Compute and cache result
            output = func(*args, **kwargs)

            with open(cache_path, 'wb') as f:
                torch.save(output, f)

            cache[cache_key] = cache_path
            output_size = os.path.getsize(cache_path) / (1024 * 1024)
            total_cache_size += output_size

            while total_cache_size > max_cache_size_mib:
                _, oldest_path = cache.popitem(last=False)
                if os.path.exists(oldest_path):
                    total_cache_size -= os.path.getsize(oldest_path) / (1024 * 1024)
                    os.remove(oldest_path)

            return output

        return wrapper

    return decorator
