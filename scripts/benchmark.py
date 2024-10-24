import gc
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tabulate import tabulate
from torch.nn.attention import sdpa_kernel, SDPBackend

from models.config import ModelCfg
from models.inference_model import LLM
from models.layers.attention import Attention
from models.layers.mlp import MLP
from models.layers.transformer_block import Block

############################################################################################################
### config
device = torch.device('cuda')
dtype = torch.bfloat16
torch.set_float32_matmul_precision('high')
dummy_cfg = ModelCfg()
dummy_cfg.hidden_size = 2560
dummy_cfg.intermediate_size = 6912
dummy_cfg.num_layers = 24
dummy_cfg.num_kv_heads = 8
dummy_cfg.max_seq_len = 2048
batch_size = 2
############################################################################################################

MODEL_PARTS = {
    'attention': Attention(dummy_cfg, 0).to(device=device, dtype=dtype),
    'mlp': MLP(dummy_cfg).to(device=device, dtype=dtype),
    'transformer_block': Block(dummy_cfg, 0).to(device=device, dtype=dtype),
    'full': LLM(dummy_cfg).to(device=device, dtype=dtype)
}
compiled = True
if compiled:
    for k, model in MODEL_PARTS.items():
        MODEL_PARTS[k] = torch.compile(model, mode='max-autotune')

def benchmark_function(func: Callable, kwargs: Dict[str, Any], iterations: int = 1000) -> Tuple[Dict[str, float], List[float]]:
    execution_times = []
    gc.collect()
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    func(**kwargs)  # warmup
    func(**kwargs)

    for _ in range(iterations):
        start_time = time.perf_counter()
        func(**kwargs)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        execution_times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    sorted_times = sorted(execution_times)
    percentile_1 = np.percentile(execution_times, 1)
    percentile_99 = np.percentile(execution_times, 99)

    stats = {'iterations': iterations, 'average': statistics.mean(execution_times), 'median': statistics.median(execution_times),
             'std_dev': statistics.stdev(execution_times), 'min': min(execution_times), 'max': max(execution_times), '1%_low': percentile_1,
             '99%_high': percentile_99, '1%_lows_avg': statistics.mean(sorted_times[:max(int(iterations * 0.01), 1)]), }
    return stats, execution_times

def plot_execution_times(func_name: str, execution_times: List[float], kwargs: Dict[str, Any]) -> None:
    plt.figure(figsize=(12, 6))

    sns.set_style("whitegrid")
    plt.plot(execution_times, linewidth=1, alpha=0.7)

    mean_time = np.mean(execution_times)
    plt.axhline(y=mean_time, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_time:.3f}ms')

    def get_tensor_shape(x):
        if isinstance(x, torch.Tensor): return f"torch.Tensor{tuple(x.shape)}"
        if isinstance(x, tuple):
            out = 'Tuple('
            for it in x:
                out += get_tensor_shape(it) + ", "
            out += ')'
            return out
        else: return repr(x)

    kwargs_str = ', '.join(f'{k}: {get_tensor_shape(v)}' for k, v in kwargs.items())

    plt.title(f'Execution Times for {func_name}\nParameters: {kwargs_str}', pad=20)
    plt.xlabel('Iteration')
    plt.ylabel('Execution Time (ms)')

    plt.legend()
    plt.tight_layout()
    Path('plots').mkdir(exist_ok=True)

    filename = f'plots/{func_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as {filename}")

def print_results(func_name: str, stats: Dict[str, float]) -> None:
    formatted_stats = {key: f"{value:.3f}" if isinstance(value, float) else value for key, value in stats.items()}
    table_data = [[key.replace('_', ' ').title(), value] for key, value in formatted_stats.items()]

    print(f"\nBenchmark Results for {func_name} (times in milliseconds):")
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid', numalign='right'))


############################################################################################################
def flash_scaled_dot_product_attention(**kwargs):
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]): F.scaled_dot_product_attention(**kwargs)
def mem_eff_scaled_dot_product_attention(**kwargs):
    with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]): F.scaled_dot_product_attention(**kwargs)
def cudnn_scaled_dot_product_attention(**kwargs):
    with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION]): F.scaled_dot_product_attention(**kwargs)
############################################################################################################


############################################################################################################
### Model Blocks
def attention_block(**kwargs):
    with torch.no_grad(): MODEL_PARTS['attention'](**kwargs)
def mlp_block(**kwargs):
    with torch.no_grad(): MODEL_PARTS['mlp'](**kwargs)
def transformer_block(**kwargs):
    with torch.no_grad(): MODEL_PARTS['transformer_block'](**kwargs)
def full_model_forward(**kwargs):
    with torch.no_grad(): MODEL_PARTS['full'](**kwargs)
############################################################################################################


############################################################################################################
### Inputs
def sdpa_inputs(device, dtype) -> dict:
    return {'query': torch.randn((batch_size, dummy_cfg.num_heads, dummy_cfg.max_position_embeddings, dummy_cfg.head_dim), dtype=dtype, device=device),
            'key': torch.randn((batch_size, dummy_cfg.num_heads, dummy_cfg.max_position_embeddings, dummy_cfg.head_dim), dtype=dtype, device=device),
            'value': torch.randn((batch_size, dummy_cfg.num_heads, dummy_cfg.max_position_embeddings, dummy_cfg.head_dim), dtype=dtype, device=device), 'is_causal': True}
def rms_norm_inputs(device, dtype) -> dict:
    return {'input': torch.randn((batch_size, dummy_cfg.max_position_embeddings, dummy_cfg.hidden_size), dtype=dtype, device=device),
            'normalized_shape': (dummy_cfg.hidden_size,), 'weight': torch.ones(dummy_cfg.hidden_size, dtype=dtype, device=device)}
def model_inputs(device, dtype) -> dict:
    return {'x': torch.randint(dummy_cfg.vocab_size, (batch_size, dummy_cfg.max_position_embeddings), device=device),
            'attn_mask': torch.ones((batch_size, dummy_cfg.max_position_embeddings), device=device)}
def attn_inputs(device, dtype) -> dict:
    return {'x': torch.randn((batch_size, dummy_cfg.max_position_embeddings, dummy_cfg.hidden_size), dtype=dtype, device=device),
            'freqs': (torch.randn((batch_size, dummy_cfg.max_position_embeddings, dummy_cfg.head_dim), dtype=dtype, device=device),
                      torch.randn((batch_size, dummy_cfg.max_position_embeddings, dummy_cfg.head_dim), dtype=dtype, device=device))}
def transformer_block_inputs(device, dtype) -> dict:
    return {'x': torch.randn((batch_size, dummy_cfg.max_position_embeddings, dummy_cfg.hidden_size), dtype=dtype, device=device),}
############################################################################################################

def main():
    num_iterations = 250

    benchmark_tasks: Dict[Callable, Dict[str, Any]] = {
        torch.matmul: {'input': torch.randn((dummy_cfg.hidden_size, dummy_cfg.max_position_embeddings), dtype=dtype, device=device), 'other': torch.randn((batch_size, dummy_cfg.max_position_embeddings, dummy_cfg.vocab_size), dtype=dtype, device=device)},
        flash_scaled_dot_product_attention: sdpa_inputs(device, dtype),
        mem_eff_scaled_dot_product_attention: sdpa_inputs(device, dtype),
        cudnn_scaled_dot_product_attention: sdpa_inputs(device, dtype),
        F.rms_norm: rms_norm_inputs(device, dtype),

        attention_block: attn_inputs(device, dtype),
        mlp_block: transformer_block_inputs(device, dtype),
        transformer_block: attn_inputs(device, dtype),
        full_model_forward: model_inputs(device, dtype)
    }

    print("Starting benchmark suite...")
    for func, kwargs in benchmark_tasks.items():
        func_name = func.__name__
        print(f"\nBenchmarking {func_name}...")
        try:
            stats, execution_times = benchmark_function(func, kwargs, iterations=num_iterations)
            print_results(func_name, stats)
            plot_execution_times(func_name, execution_times, kwargs)
        except Exception as e:
            print(f"Error benchmarking {func_name}: {str(e)}")

    print("\nBenchmark suite completed!")

if __name__ == "__main__":
    main()