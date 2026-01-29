# H100 Diffusers Skill for Claude Code

Optimized CUDA kernels for H100 GPUs targeting the HuggingFace diffusers library, with a Claude Code skill for guided kernel development.

## What's Included

- **CUDA Kernels**: Optimized implementations for RMSNorm, RoPE (1D/3D), GEGLU, SwiGLU, and AdaLN
- **Python API**: Drop-in replacements for diffusers operations via `ltx_kernels`
- **Claude Code Skill**: Expert guidance for writing custom H100 kernels

## Quick Start

### Installation

```bash
# Install the package
pip install -e .

# With diffusers support
pip install -e ".[diffusers]"
```

### Build Kernels (requires CUDA)

```bash
# Using Docker (recommended)
docker run --rm --mount type=bind,source=$(pwd),target=/kernelcode \
  -w /kernelcode ghcr.io/huggingface/kernel-builder:main build

# Or with Nix
nix run .#build-and-copy --max-jobs 2 --cores 8 -L
```

## Using the Claude Code Skill

This repository includes a Claude Code skill that provides expert guidance for developing optimized CUDA kernels targeting H100 GPUs.

### Activating the Skill

The skill activates automatically when you ask Claude Code about:
- Writing CUDA kernels for diffusion models
- Optimizing attention, normalization, or activation layers
- Integrating custom kernels with diffusers pipelines
- H100-specific optimizations

### Example Prompts

**Writing a new kernel:**
```
Write a fused RMSNorm + residual kernel optimized for H100
```

**Optimizing existing code:**
```
Help me optimize this attention kernel for H100's 192KB shared memory
```

**Integration questions:**
```
How do I add a custom AdaLN kernel to the LTX-Video transformer?
```

**Architecture guidance:**
```
What block sizes should I use for flash attention on H100?
```

### What the Skill Provides

1. **H100 Architecture Reference**: SM count, shared memory, memory bandwidth, warp size
2. **Kernel Templates**: Complete CUDA implementations for common operations
3. **Block Size Guidelines**: Optimal configurations for different kernel types
4. **PyTorch Integration Patterns**: C++ bindings and Python API examples
5. **Performance Profiling**: Commands for nsys and ncu analysis

### Skill Files

The skill documentation is in `.claude/skills/h100-diffusers-kernels/`:
