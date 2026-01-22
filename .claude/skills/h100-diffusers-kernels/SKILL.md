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

## Project Structure

```
hardware_kernel/
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

### With Docker (kernel-builder)
```bash
docker run --rm --mount type=bind,source=$(pwd),target=/kernelcode \
  -w /kernelcode ghcr.io/huggingface/kernel-builder:main build
```

### With Nix
```bash
nix run .#build-and-copy --max-jobs 2 --cores 8 -L
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

## Common Issues

1. **Bank conflicts in shared memory**: Add padding for 32-bank conflict avoidance
2. **Poor occupancy**: Check register usage with `--ptxas-options=-v`
3. **Memory coalescing**: Ensure 128-byte aligned accesses
4. **Warp divergence**: Use `__ballot_sync` for conditional execution

## See Also

- [kernel-templates.md](kernel-templates.md) - Complete kernel templates
- [h100-optimization-guide.md](h100-optimization-guide.md) - Deep dive on H100 optimizations
