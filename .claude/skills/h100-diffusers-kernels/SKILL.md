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

## Quick Start

**For integrating kernels into diffusers pipelines, start with the minimal example:**
```bash
python scripts/ltx_kernel_injection_example.py
```

## When This Skill Applies

Use this skill when:
- Writing new CUDA kernels for diffusion models
- Optimizing existing kernels for H100 architecture
- Implementing custom attention, normalization, or activation layers
- Integrating kernels with diffusers pipelines (LTX-Video, Stable Diffusion, FLUX, DiT)
- Debugging kernel performance issues on H100

## Working Example

A complete working example is available at `examples/ltx_video/`. This demonstrates:
- Custom CUDA kernels (RMSNorm, RoPE 3D, GEGLU, AdaLN)
- Build system setup with setup.py, build.toml, and flake.nix
- PyTorch C++ bindings and Python API
- Video generation script using diffusers

**Example benchmarks on H100:**
```
RMSNorm [2x1024x2048]: 0.054 ms
GEGLU [2x1024x4096]: 0.030 ms
```

## Project Structure

```
hardware_kernel/
├── examples/
│   └── ltx_video/              # Complete working example
│       ├── kernel_src/         # CUDA kernels
│       ├── torch-ext/          # PyTorch bindings
│       ├── setup.py            # pip install -e .
│       └── generate_video.py   # Video generation script
├── kernel_src/                 # CUDA kernel implementations
│   ├── layernorm.cu           # RMSNorm/LayerNorm
│   ├── rope.cu                # 1D and 3D rotary embeddings
│   ├── geglu.cu               # GELU-gated linear units
│   └── adaln.cu               # Adaptive layer norm
└── torch-ext/
    ├── torch_binding.cpp      # PyTorch C++ bindings
    └── ltx_kernels/__init__.py
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

## Core Kernel Patterns

### Warp Shuffle Reductions
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

### Block Sizes for Attention
- `BLOCK_SIZE_M = 128`, `BLOCK_SIZE_N = 64`, `BLOCK_SIZE_K = 64`
- `NUM_WARPS = 8`

### Thread Configuration

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

## Building Kernels

### With Nix (Recommended)
```bash
nix run .#build-and-copy --max-jobs 2 --cores 8 -L
```

### With pip/uv
```bash
uv pip install -e .
```

### build.toml Configuration
```toml
[general]
name = "ltx_kernels"
backends = ["cuda"]

[kernel.your_kernel]
backend = "cuda"
src = ["kernel_src/your_kernel.cu"]
cuda-capabilities = ["9.0"]
```

## Diffusers Integration

> **See [diffusers-integration.md](references/diffusers-integration.md) for the complete guide.**

### Critical Pitfalls

#### 1. RMSNorm Weight May Be None

LTX-Video uses `elementwise_affine=False` for some RMSNorm modules:
```python
# Transformer blocks: NO WEIGHT
self.norm1 = RMSNorm(dim, elementwise_affine=False)

# Attention modules: HAS WEIGHT
self.norm_q = torch.nn.RMSNorm(..., elementwise_affine=True)
```

**Solution:** Handle both cases:
```python
has_weight = hasattr(module, 'weight') and module.weight is not None
if has_weight:
    output = rmsnorm(x, module.weight, eps=eps)
else:
    weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    output = rmsnorm(x, weight, eps=eps)
```

#### 2. Diffusers RMSNorm != torch.nn.RMSNorm

```python
# WRONG - misses diffusers RMSNorm
if isinstance(module, torch.nn.RMSNorm):

# CORRECT - catches all RMSNorm variants
if type(module).__name__ == 'RMSNorm':
```

#### 3. LTX-Video Uses GELU, Not GEGLU

LTX-Video uses `activation_fn="gelu-approximate"`. Don't patch GEGLU for LTX-Video.

#### 4. Inject Kernels BEFORE CPU Offloading

```python
pipe = LTXPipeline.from_pretrained(...)
pipe.to("cuda")
inject_optimized_kernels(pipe)  # BEFORE offloading
pipe.enable_model_cpu_offload()  # Now safe
```

### Minimal Integration Pattern

```python
from diffusers import LTXPipeline
from ltx_kernels import rmsnorm

def patch_rmsnorm_modules(model):
    """Patch all RMSNorm modules to use custom kernel."""
    for name, module in model.named_modules():
        if type(module).__name__ == 'RMSNorm':
            eps = getattr(module, 'eps', 1e-6)
            has_weight = hasattr(module, 'weight') and module.weight is not None

            if has_weight:
                def make_forward(mod, epsilon):
                    def forward(x):
                        return rmsnorm(x, mod.weight, eps=epsilon)
                    return forward
                module.forward = make_forward(module, eps)
            else:
                def make_forward(epsilon):
                    def forward(x):
                        w = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
                        return rmsnorm(x, w, eps=epsilon)
                    return forward
                module.forward = make_forward(eps)

# Usage
pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
pipe.to("cuda")
patch_rmsnorm_modules(pipe.transformer)
pipe.enable_model_cpu_offload()
```

## Kernel-Specific Guidelines

### RMSNorm
- Input layout: `[..., hidden_size]`
- Epsilon default: 1e-6
- **Weight may be None** if `elementwise_affine=False`

### RoPE
- 1D: `[batch, seq, heads, head_dim]` - for text
- 3D: `[batch, t*h*w, heads, head_dim]` - for video
- LTX-Video computes its own RoPE via `LTXVideoRotaryPosEmbed`

### GEGLU vs GELU
- **GEGLU**: Input `[batch, seq, 2*hidden]` -> Output `[batch, seq, hidden]`
- **GELU**: Standard activation
- **LTX-Video uses GELU, NOT GEGLU**

### AdaLN
- Formula: `norm(x) * weight * (1 + scale) + shift`
- Used in DiT blocks for conditioning

## Performance Profiling

```bash
# NVIDIA Nsight Systems
nsys profile -o profile python your_script.py

# NVIDIA Nsight Compute
ncu --set full -o metrics python your_script.py
```

## Common Issues

> **See [troubleshooting.md](references/troubleshooting.md) for all common issues and solutions.**

Quick fixes:
- **"NoneType has no attribute contiguous"**: RMSNorm weight is None, create ones
- **isinstance() not matching**: Use `type(module).__name__` instead
- **GEGLU not called**: Model uses GELU, not GEGLU
- **Patching doesn't persist**: Inject before `enable_model_cpu_offload()`

## See Also

- [ltx_kernel_injection_example.py](scripts/ltx_kernel_injection_example.py) - **Minimal working example (~150 lines) - START HERE**
- [diffusers-integration.md](references/diffusers-integration.md) - Complete integration guide
- [troubleshooting.md](references/troubleshooting.md) - Common issues and solutions
- [kernel-templates.md](references/kernel-templates.md) - Complete kernel templates
- [h100-optimization-guide.md](references/h100-optimization-guide.md) - Deep dive on H100 optimizations
- [examples/ltx_video/generate_video.py](../../../examples/ltx_video/generate_video.py) - Full LTX-Video script
