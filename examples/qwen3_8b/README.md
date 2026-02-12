# Qwen3-8B Custom CUDA Kernels

Optimized CUDA kernels for [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) on H100 GPUs.

## Model Architecture

| Parameter | Value |
|-----------|-------|
| hidden_size | 4096 |
| num_hidden_layers | 32 |
| rms_norm_eps | 1e-6 |
| num_attention_heads | 32 |
| RMSNorm modules | 65 (32 * 2 + 1) |

## Benchmark Results

**Device:** NVIDIA H100 80GB HBM3
**Precision:** bfloat16

### RMSNorm Micro-benchmarks

| Config | Custom (ms) | PyTorch (ms) | Speedup |
|:-------|:-----------:|:------------:|:-------:|
| [1x128x4096] | 0.040 | 0.062 | **1.58x** |
| [1x512x4096] | 0.038 | 0.064 | **1.69x** |
| [1x1024x4096] | 0.037 | 0.071 | **1.90x** |
| [1x2048x4096] | 0.045 | 0.091 | **2.03x** |
| [1x4096x4096] | 0.071 | 0.150 | **2.12x** |
| [4x512x4096] | 0.056 | 0.093 | **1.67x** |
| [8x256x4096] | 0.045 | 0.092 | **2.06x** |
| [1x8192x4096] | 0.109 | 0.269 | **2.47x** |

**Average Speedup: 1.94x**

### Memory Bandwidth Analysis

| Metric | Value |
|--------|-------|
| Config | [1x2048x4096] |
| Total data moved | 33.56 MB |
| Kernel latency | 0.045 ms |
| Achieved bandwidth | 747.1 GB/s |
| H100 theoretical | 3350 GB/s |
| Bandwidth efficiency | 22.3% |

## Quick Start

```bash
# Install the kernels
cd examples/qwen3_8b
uv pip install -e .

# Run benchmark
python benchmark_rmsnorm.py

# Run with different precision
python benchmark_rmsnorm.py --dtype fp16
```

## Usage with Transformers

```python
from transformers import AutoModelForCausalLM
import torch
from qwen3_kernels import rmsnorm

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

# Patch RMSNorm modules
def patch_rmsnorm(model):
    for name, module in model.named_modules():
        if 'RMSNorm' in type(module).__name__:
            eps = getattr(module, 'variance_epsilon', 1e-6)
            def make_forward(mod, epsilon):
                def forward(x):
                    return rmsnorm(x, mod.weight, eps=epsilon)
                return forward
            module.forward = make_forward(module, eps)

patch_rmsnorm(model)

# Generate text
inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
```

## Expected Model Impact

- RMSNorm modules: 65 (32 layers * 2 + 1 final)
- RMSNorm % of total compute: ~3-5%
- Expected end-to-end speedup: ~2-3%

For larger speedups, combine with:
- Flash Attention 2: `attn_implementation='flash_attention_2'`
- torch.compile: `model = torch.compile(model)`

## Files

```
examples/qwen3_8b/
├── kernel_src/
│   └── rmsnorm.cu          # Vectorized RMSNorm kernel
├── torch-ext/
│   ├── torch_binding.cpp   # PyTorch C++ bindings
│   └── qwen3_kernels/      # Python API
│       └── __init__.py
├── benchmark_rmsnorm.py    # Benchmark script
├── setup.py                # Build configuration
├── pyproject.toml          # Project metadata
├── build.toml              # Kernel build config
└── README.md               # This file
```

## Kernel Optimizations

The RMSNorm kernel is optimized for H100 (sm_90) with:

1. **Vectorized memory access**: Uses `__nv_bfloat162` for 2-element vectorized loads/stores
2. **Warp shuffle reductions**: No shared memory bank conflicts
3. **Coalesced memory patterns**: Maximizes memory bandwidth utilization
4. **Loop unrolling**: `#pragma unroll 4` for reduced instruction overhead

## torch.compile Compatibility

The kernel is registered as a PyTorch custom op, enabling compatibility with `torch.compile`:

```python
from qwen3_kernels import rmsnorm

# Works with torch.compile
model = torch.compile(model)  # OK, kernel is registered
```
