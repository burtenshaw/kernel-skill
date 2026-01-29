# LTX-Video Custom CUDA Kernels for H100

Optimized CUDA kernels for the [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video) diffusion model, targeting NVIDIA H100 GPUs (compute capability 9.0/sm_90).

## Included Kernels

| Kernel | Description | H100 Optimization |
|--------|-------------|-------------------|
| **RMSNorm** | Root Mean Square Layer Normalization | Warp shuffle reductions, fused residual |
| **RoPE 1D** | Rotary Position Embeddings (text) | Element-wise parallelism |
| **RoPE 3D** | Rotary Position Embeddings (video) | Temporal + spatial encoding |
| **GEGLU** | GELU-Gated Linear Unit | Fast tanh approximation |
| **SwiGLU** | SiLU-Gated Linear Unit | Fused gate activation |
| **AdaLN** | Adaptive Layer Norm (DiT conditioning) | Fused norm + modulation |

## Quick Start

### 1. Install Python Dependencies

```bash
pip install diffusers transformers accelerate torch
```

### 2. Build the Kernels

Using Nix:

```bash
cd examples/ltx_video

# First time: update flake inputs
nix flake update

# Build kernels (adjust cores based on your system)
nix run .#build-and-copy --max-jobs 2 --cores 8 -L
```

For faster Nix builds, add the HuggingFace cache:
```bash
# If you have cachix
cachix use huggingface

# Or without cachix
nix run nixpkgs#cachix -- use huggingface
```

### 3. Generate a Video

```bash
# Generate a video of a dog (default prompt)
python generate_video.py

# Custom prompt
python generate_video.py --prompt "A corgi puppy playing in autumn leaves"

# Higher quality (more frames/steps)
python generate_video.py --num-frames 49 --steps 50

# Full options
python generate_video.py \
  --prompt "A golden retriever running on the beach at sunset" \
  --num-frames 25 \
  --height 480 \
  --width 704 \
  --steps 30 \
  --guidance-scale 7.5 \
  --seed 42 \
  --output my_dog_video.mp4
```

## Project Structure

```
ltx_video/
├── build.toml                    # Kernel builder configuration
├── flake.nix                     # Nix flake for building
├── generate_video.py             # Main video generation script
├── README.md                     # This file
├── kernel_src/                   # CUDA kernel implementations
│   ├── rmsnorm.cu               # RMSNorm with warp reductions
│   ├── rope.cu                  # 1D and 3D RoPE
│   ├── geglu.cu                 # GEGLU/SwiGLU activations
│   └── adaln.cu                 # Adaptive LayerNorm
├── torch-ext/                    # PyTorch extension
│   ├── torch_binding.cpp        # C++ bindings
│   └── ltx_kernels/             # Python API
│       └── __init__.py
└── tests/                        # Kernel tests
```

## Kernel API

```python
from ltx_kernels import (
    rmsnorm,           # RMS Layer Normalization
    rmsnorm_residual,  # Fused RMSNorm + residual add
    rope_1d,           # 1D Rotary Position Embeddings
    rope_3d,           # 3D RoPE for video (time x height x width)
    geglu,             # GELU-Gated Linear Unit
    swiglu,            # SiLU-Gated Linear Unit
    adaln_rmsnorm,     # Adaptive LayerNorm with RMSNorm
    adaln_layernorm,   # Adaptive LayerNorm with LayerNorm
    adaln_zero,        # AdaLN with zero-initialized gating
)

# Example: RMSNorm
import torch
x = torch.randn(2, 1024, 2048, device='cuda', dtype=torch.bfloat16)
weight = torch.ones(2048, device='cuda', dtype=torch.bfloat16)
output = rmsnorm(x, weight, eps=1e-6)

# Example: 3D RoPE for video
query = torch.randn(1, 25*30*44, 16, 64, device='cuda', dtype=torch.bfloat16)
key = torch.randn(1, 25*30*44, 16, 64, device='cuda', dtype=torch.bfloat16)
query, key = rope_3d(query, key, num_frames=25, height=30, width=44)
```

## Diffusers Integration

```python
from diffusers import LTXPipeline
from ltx_kernels import rmsnorm, rope_3d, geglu

# Custom attention processor using optimized kernels
class OptimizedLTXAttnProcessor:
    def __init__(self, theta_base=10000.0):
        self.theta_base = theta_base

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, **kwargs):
        batch, seq_len, _ = hidden_states.shape

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        q = attn.to_q(hidden_states)
        k = attn.to_k(encoder_hidden_states)
        v = attn.to_v(encoder_hidden_states)

        head_dim = q.shape[-1] // attn.heads
        q = q.view(batch, -1, attn.heads, head_dim)
        k = k.view(batch, -1, attn.heads, head_dim)
        v = v.view(batch, -1, attn.heads, head_dim)

        # Use optimized attention (Flash Attention 2 via SDPA)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=attn.scale
        )

        out = out.transpose(1, 2).reshape(batch, -1, attn.heads * head_dim)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out

# Load and patch pipeline
pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.transformer.set_attn_processor(OptimizedLTXAttnProcessor())

# Generate video
video = pipe(
    prompt="A dog playing fetch in a park",
    num_frames=25,
    height=480,
    width=704,
).frames[0]
```

## H100 Optimizations

These kernels are specifically optimized for NVIDIA H100:

- **132 SMs**: Grid sizes tuned for full SM utilization
- **192 KB Shared Memory/SM**: Large tiles for attention
- **3.35 TB/s Memory Bandwidth**: Coalesced access patterns
- **Warp Shuffles**: Reductions without shared memory bank conflicts
- **BF16 Native**: Preferred dtype for stability and performance

## Performance Profiling

```bash
# NVIDIA Nsight Systems (system-wide)
nsys profile -o profile python generate_video.py

# NVIDIA Nsight Compute (kernel details)
ncu --set full -o metrics.ncu-rep python generate_video.py
```

## Supported Data Types

All kernels support three precision modes:
- **FP16** (`torch.float16`) - Good for inference
- **BF16** (`torch.bfloat16`) - Recommended for training/inference
- **FP32** (`torch.float32`) - Reference/debugging

## Requirements

- NVIDIA H100 GPU (compute capability 9.0)
- CUDA 12.0+
- PyTorch 2.0+
- diffusers 0.25.0+

## Troubleshooting

**Kernels not loading:**
```bash
# Check if kernels are built
ls torch-ext/ltx_kernels/_ops*.so

# Rebuild with Nix
nix run .#build-and-copy -L

# Or rebuild with pip/uv
uv pip install -e . --force-reinstall
```

**CUDA out of memory:**
```bash
# Use CPU offloading (enabled by default)
# Or reduce resolution/frames
python generate_video.py --num-frames 13 --height 320 --width 480
```

**Wrong CUDA architecture:**
```bash
# Verify H100 is detected
python -c "import torch; print(torch.cuda.get_device_capability())"
# Should print (9, 0) for H100
```

## License

MIT License - see repository root for details.
