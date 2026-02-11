# CUDA Kernels Skill for Diffusers and Transformers

This repository provides the `cuda-kernels` skill plus a working LTX-Video example for building, integrating, and benchmarking optimized CUDA kernels for Hugging Face `diffusers` and `transformers`.

## What This Skill Covers

- **GPU targets**: NVIDIA H100 (`sm_90`), A100 (`sm_80`), and T4 (`sm_75`)
- **Libraries and models**: `diffusers` and `transformers` libraries with the following models:
- **Integration workflows**:
  - Custom kernel injection for diffusers and transformers models
  - Hugging Face Kernels Hub integration via `get_kernel(...)`
- **Benchmarking workflows**:
  - End-to-end generation benchmarks against baseline
  - Isolated micro-benchmarks (for example, RMSNorm)
  - Profiling with `nsys` and `ncu`


## Repository Layout

```text
.claude/skills/cuda-kernels/
├── SKILL.md                              # Main skill instructions and workflows
├── scripts/                              # Benchmark and integration examples
└── references/                           # Optimization, integration, and troubleshooting guides

examples/ltx_video/
├── generate_video.py                     # End-to-end benchmark and generation entrypoint
├── benchmark_rmsnorm.py                  # Isolated RMSNorm benchmark
├── kernel_src/                           # CUDA kernel implementations
└── torch-ext/                            # PyTorch extension bindings
```