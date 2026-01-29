---
name: h100-diffusers-kernels
description: "Provides guidance for writing optimized CUDA kernels for H100 GPUs (sm_90) targeting diffusers library models like LTX-Video, Stable Diffusion, and DiT. Applies when working with attention, normalization, RoPE, activations, or custom kernel development for diffusion transformers."
disable-model-invocation: false
user-invocable: true
allowed-tools: "Read, Grep, Glob, Bash"
argument-hint: "kernel type: attention, rmsnorm, rope, adaln, geglu"
---

# H100 CUDA Kernels for Diffusers

This skill provides patterns and guidance for developing optimized CUDA kernels targeting NVIDIA H100 GPUs (compute capability 9.0) for use with the HuggingFace diffusers library.

## When This Skill Applies

Use this skill when:
- Writing new CUDA kernels for diffusion models
- Optimizing existing kernels for H100 architecture
- Implementing custom attention, normalization, or activation layers
- Integrating kernels with diffusers pipelines (LTX-Video, Stable Diffusion, FLUX, DiT)
- Debugging kernel performance issues on H100

## Working Example

A complete working example is available at `examples/ltx_video/`. This example demonstrates:
- Custom CUDA kernels (RMSNorm, RoPE 3D, GEGLU, AdaLN) for LTX-Video
- Build system setup with setup.py, build.toml, and flake.nix
- PyTorch C++ bindings and Python API
- Video generation script using diffusers

**Example benchmarks on H100:**
```
RMSNorm [2x1024x2048]: 0.054 ms
GEGLU [2x1024x4096]: 0.030 ms
RoPE 3D [batch=2, seq=480, heads=8]: 1.670 ms
```

## Project Structure

```
hardware_kernel/
├── examples/
│   └── ltx_video/              # ← Complete working example
│       ├── kernel_src/         # CUDA kernels
│       ├── torch-ext/          # PyTorch bindings
│       ├── setup.py            # pip install -e .
│       ├── build.toml          # kernel-builder config
│       └── generate_video.py   # Video generation script
├── build.toml              # Kernel builder config (sm_90 targeting)
├── kernel_src/             # CUDA kernel implementations
│   ├── attention.cu        # Flash attention (BLOCK_SIZE_M=128, BLOCK_SIZE_N=64)
│   ├── layernorm.cu        # RMSNorm/LayerNorm with warp reductions
│   ├── rope.cu             # 1D and 3D rotary embeddings
│   ├── adaln.cu            # Adaptive layer norm for DiT
│   ├── geglu.cu            # GELU-gated linear units
│   └── groupnorm.cu        # Group normalization
├── torch-ext/
│   ├── torch_binding.cpp   # PyTorch C++ bindings
│   └── ltx_kernels/
│       └── __init__.py     # Python API
└── tests/
    └── test_kernels.py     # Kernel tests
```

## H100 Architecture Reference

| Spec | Value | Optimization Impact |
|------|-------|---------------------|
| SMs | 132 | Grid sizing: aim for multiples of 132 |
| Threads/SM | 2048 | Max 16 blocks of 128 threads per SM |
| Shared Memory | 192 KB/SM | Large tiles possible |
| L2 Cache | 50 MB | Reuse across blocks |
| Memory BW | 3.35 TB/s | Coalesced access critical |
| Warp Size | 32 | All reductions use warp shuffles |
| Registers | 255/thread | Register tiling for small arrays |

## Core Kernel Patterns

### 1. Warp Shuffle Reductions

All normalization kernels use warp-level reductions:

```cuda
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}
```

### 2. Block Sizes for Attention

Flash attention uses these block sizes for H100:
- `BLOCK_SIZE_M = 128` (query block)
- `BLOCK_SIZE_N = 64` (key/value block)
- `BLOCK_SIZE_K = 64`
- `NUM_WARPS = 8`

### 3. Thread Configuration

For element-wise ops (RoPE, GEGLU):
```cuda
constexpr int BLOCK_SIZE = 256;
int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
```

For reduction ops (LayerNorm, RMSNorm):
```cuda
int threads = min(hidden_size, 1024);
threads = (threads + 32 - 1) / 32 * 32;  // Round to warp boundary
```

## Supported Data Types

All kernels support three precision modes:
- `__half` (FP16) - Default for inference
- `__nv_bfloat16` (BF16) - Preferred for training
- `float` (FP32) - Reference/debugging

Entry point naming convention:
```cpp
void kernel_forward_fp16(...);
void kernel_forward_bf16(...);
void kernel_forward_fp32(...);
```

## Building Kernels

### With Nix (Recommended)
```bash
# Build kernels
nix run .#build-and-copy --max-jobs 2 --cores 8 -L

# Or with flake
nix flake update && nix run .#build-and-copy -L
```

### With pip/uv (Development fallback)
```bash
# Using uv (faster)
uv pip install -e .

# Using pip
pip install -e .
```

**setup.py example:**
```python
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cuda_sources = [
    "kernel_src/rmsnorm.cu",
    "kernel_src/rope.cu",
    "kernel_src/geglu.cu",
]
cpp_sources = ["torch-ext/torch_binding.cpp"]

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3", "-std=c++17", "--use_fast_math",
        "-arch=sm_90",
        "-gencode=arch=compute_90,code=sm_90",
    ],
}

setup(
    name="ltx-kernels",
    ext_modules=[
        CUDAExtension(
            name="ltx_kernels._ops",
            sources=cpp_sources + cuda_sources,
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

### build.toml Configuration
```toml
[general]
name = "ltx_kernels"
backends = ["cuda"]

[kernel.your_kernel]
backend = "cuda"
depends = []
src = ["kernel_src/your_kernel.cu"]
cuda-capabilities = ["9.0"]
```

### flake.nix Configuration
```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    kernel-builder.url = "github:huggingface/kernel-builder";
  };

  outputs = { self, nixpkgs, kernel-builder }:
    kernel-builder.lib.genFlakeOutputs {
      path = ./.;
    };
}
```

## PyTorch Integration

### C++ Binding Pattern
```cpp
void your_kernel_forward(
    torch::Tensor& output,
    const torch::Tensor& input,
    // ... other params
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");

    const at::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (input.scalar_type() == at::kHalf) {
        your_kernel_forward_fp16(..., stream);
    } else if (input.scalar_type() == at::kBFloat16) {
        your_kernel_forward_bf16(..., stream);
    } else if (input.scalar_type() == at::kFloat) {
        your_kernel_forward_fp32(..., stream);
    }
}
```

### Python API Pattern
```python
def your_kernel(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    ops.your_kernel_forward(out, input.contiguous())
    return out
```

## Diffusers Integration

### Custom Attention Processor
```python
from diffusers import LTXPipeline
from ltx_kernels import attention, rmsnorm, rope

class CustomAttnProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, **kwargs):
        q = attn.to_q(hidden_states)
        k = attn.to_k(encoder_hidden_states or hidden_states)
        v = attn.to_v(encoder_hidden_states or hidden_states)

        # Apply custom RoPE
        q, k = rope(q, k, theta_base=10000.0)

        # Run optimized attention
        out = attention(q, k, v, scale=attn.scale)
        return attn.to_out[1](attn.to_out[0](out))

pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video")
pipe.transformer.set_attn_processor(CustomAttnProcessor())
```

## Kernel-Specific Guidelines

### Attention
- Input layout: `[batch, heads, seq_len, head_dim]`
- Uses online softmax (numerically stable)
- Fused Q@K^T with scaling

### RMSNorm
- Input layout: `[..., hidden_size]`
- Epsilon default: 1e-6 (matches LTX-Video)
- Weight-only (no bias)

### RoPE
- 1D: `[batch, seq, heads, head_dim]` - for text
- 3D: `[batch, t*h*w, heads, head_dim]` - for video
- Dimension split for 3D: `head_dim // 3` each for t, h, w

### AdaLN
- Formula: `norm(x) * weight * (1 + scale) + shift`
- Scale/shift from timestep MLP: `[batch, hidden]`
- Used in DiT blocks for conditioning

### GEGLU
- Input: `[batch, seq, 2*hidden]`
- Output: `[batch, seq, hidden]`
- Uses tanh approximation by default (faster)

## Performance Profiling

```bash
# NVIDIA Nsight Systems
nsys profile -o kernel_profile python your_script.py

# NVIDIA Nsight Compute (detailed kernel analysis)
ncu --set full --csv -o metrics.csv python your_script.py
```

## Common Issues and Solutions

### 1. Type Conversion Errors with FP16/BF16

**Problem:** PyTorch compiles with `-D__CUDA_NO_HALF_OPERATORS__` which disables implicit type conversions:
```
error: no suitable conversion function from "__half" to "float" exists
```

**Solution:** Add explicit type conversion helper functions in your .cu files:
```cuda
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Type conversion helpers (required for PyTorch compatibility)
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float from_float(float x, float*) { return x; }
__device__ __forceinline__ __half from_float(float x, __half*) { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat16 from_float(float x, __nv_bfloat16*) { return __float2bfloat16(x); }

// Usage in kernels:
float val = to_float(input[idx]);
output[idx] = from_float(result, (scalar_t*)nullptr);
```

### 2. Missing CUDA Headers in torch_binding.cpp

**Problem:** Undeclared types `__half`, `__nv_bfloat16`

**Solution:** Include required headers:
```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAGuard.h>
```

### 3. Bank Conflicts in Shared Memory
Add padding for 32-bank conflict avoidance:
```cuda
__shared__ float data[32][33];  // 33 instead of 32
```

### 4. Poor Occupancy
Check register usage:
```bash
nvcc --ptxas-options=-v your_kernel.cu
```

### 5. Memory Coalescing
Ensure 128-byte aligned accesses for optimal bandwidth.

### 6. Build Fails with "No module named torch"
Add torch to build dependencies in pyproject.toml:
```toml
[build-system]
requires = ["setuptools", "wheel", "torch>=2.0"]
```

## Video Generation Script Example

See `examples/ltx_video/generate_video.py` for a complete example. Key usage:

```bash
# Build kernels first
cd examples/ltx_video
uv pip install -e .

# Generate video
uv run python generate_video.py \
    --prompt "A golden retriever running in a park" \
    --num-frames 25 \
    --steps 30
```

**Script structure:**
```python
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

# Import custom kernels
from ltx_kernels import rmsnorm, rope_3d, geglu

# Load pipeline
pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Generate
output = pipe(
    prompt="A golden retriever running in a park",
    num_frames=25,
    height=480,
    width=704,
    num_inference_steps=30,
)

export_to_video(output.frames[0], "output.mp4", fps=24)
```

## See Also

- [kernel-templates.md](kernel-templates.md) - Complete kernel templates
- [h100-optimization-guide.md](h100-optimization-guide.md) - Deep dive on H100 optimizations
- [examples/ltx_video/](../../../examples/ltx_video/) - Complete working example
