# CUDA Kernels Skill For Diffusers And Transformers

This repository provides the `cuda-kernels` skill plus two concrete demos:
- `examples/ltx_video`: a diffusers-oriented CUDA kernel example
- `examples/qwen3_8b`: a kernel-builder-native OpenCode demo for a Qwen3 RMSNorm kernel that can be published to the Hugging Face Kernel Hub and benchmarked on HF Jobs

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
- **OpenCode workflow**:
  - Project-local agents in `.opencode/agents/`
  - Publish flow for versioned Hub kernels
  - Published-artifact benchmarking via HF Jobs


## Repository Layout

```text
.opencode/agents/
├── kernel-demo.md                       # Primary OpenCode agent
├── kernel-analyze.md                    # Read-only subagent
├── kernel-publish.md                    # Build/upload subagent
└── kernel-bench.md                      # HF Jobs benchmark subagent

.claude/skills/cuda-kernels/
├── SKILL.md                              # Main skill instructions and workflows
├── scripts/                              # Benchmark and integration examples
└── references/                           # Optimization, integration, and troubleshooting guides

examples/qwen3_8b/
├── benchmarks/benchmark_rmsnorm.py       # `kernels benchmark` workloads
├── scripts/hf_jobs_benchmark.py          # HF Jobs helper
├── torch-ext/                            # Kernel-builder bindings + Python API
└── flake.nix                             # Nix build entrypoint

examples/ltx_video/
├── generate_video.py                     # End-to-end benchmark and generation entrypoint
├── benchmark_rmsnorm.py                  # Isolated RMSNorm benchmark
├── kernel_src/                           # CUDA kernel implementations
└── torch-ext/                            # PyTorch extension bindings
```

## OpenCode Demo

The Qwen3 example is wired for project-local OpenCode agents. From the repo root:

```bash
opencode run --agent kernel-demo "Publish examples/qwen3_8b as <namespace>/<repo> and benchmark version 1 on HF Jobs"
```

That flow is expected to:
- refine the RMSNorm kernel in `examples/qwen3_8b`
- validate and build it with `nix` plus the Kernels CLI
- upload version `1` to the Hub
- benchmark the published artifact on HF Jobs
- return a PyTorch snippet built around `from kernels import get_kernel`

Build/publish requires a Linux host or Linux CI runner because the demo outputs CUDA artifacts.
