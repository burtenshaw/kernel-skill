# Qwen3 RMSNorm Kernel Demo

Kernel-builder-native RMSNorm demo for [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B). The example is tuned for Hopper-class GPUs with CUDA capability `9.0` and is intended to be published to the Hugging Face Kernel Hub as version `1`.

## Model Shape

| Parameter | Value |
|-----------|-------|
| hidden_size | 4096 |
| num_hidden_layers | 32 |
| rms_norm_eps | 1e-6 |
| RMSNorm modules | 65 |

## Layout

```text
examples/qwen3_8b/
├── benchmarks/
│   └── benchmark_rmsnorm.py      # `kernels benchmark` workloads
├── kernel_src/
│   └── rmsnorm.cu                # Hopper-optimized CUDA kernel
├── scripts/
│   └── hf_jobs_benchmark.py      # Bench published artifacts on HF Jobs
├── tests/
│   └── test_qwen3_kernels.py     # CPU fallback / patch helper tests
├── torch-ext/
│   ├── torch_binding.cpp         # Torch op registration + legacy fallback
│   ├── torch_binding.h
│   └── qwen3_kernels/__init__.py # Python wrapper + patch helper
├── benchmark_rmsnorm.py          # Local wrapper around `kernels benchmark`
├── build.toml
├── example.py                    # Hub loading example
├── flake.nix
├── pyproject.toml
├── setup.py
└── README.md
```

## Build And Check

Run the build step from a Linux host or Linux CI runner. This demo targets CUDA artifacts; on Darwin/macOS the flake resolves, but `build-and-copy` does not produce a usable CUDA bundle.

```bash
cd examples/qwen3_8b

nix flake update
nix run .#build-and-copy -L
```

`build/` is the artifact directory you upload to the Hub.

## Local Benchmarking

Use the official Kernels benchmark runner:

```bash
cd examples/qwen3_8b
python benchmark_rmsnorm.py --warmup 20 --iterations 100
```

That wrapper runs:

```bash
uvx --from "kernels[benchmark]" kernels benchmark . --warmup 20 --iterations 100
```

The tracked workloads cover representative Qwen3 inference shapes:
- `short_prompt`: `1x128x4096`
- `medium_prompt`: `1x512x4096`
- `long_prompt`: `1x2048x4096`
- `batch4_prompt`: `4x512x4096`
- `extended_context`: `1x8192x4096`

## Publish To The Hub

```bash
cd examples/qwen3_8b

nix run .#build-and-copy -L
uvx --from kernels kernels upload ./build --repo-id <namespace>/<repo>
uvx --from kernels --with kernel-abi-check kernels check <namespace>/<repo> --revision v1
```

This demo sets `version = 1` in `build.toml`, so published clients should load `v1`. In the currently available CLI, `kernels check` validates the published Hub repo, not a local checkout.

## Benchmark The Published Artifact On HF Jobs

The helper script submits a Jobs run, waits for completion, and prints the final JSON payload:

```bash
uv run scripts/hf_jobs_benchmark.py --repo-id <namespace>/<repo>
```

Hardware selection rules:
- Prefer `h100` if your HF Jobs account exposes it.
- Otherwise use another Hopper-class flavor such as `h200`.
- Do not benchmark this published artifact on `a100-large` unless you first widen `cuda-capabilities` to include `8.0`.

## OpenCode Flow

From the repo root:

```bash
opencode run --agent kernel-demo "Publish examples/qwen3_8b as <namespace>/<repo> and benchmark version 1 on HF Jobs"
```

Project-local agents live in `.opencode/agents/`:
- `kernel-demo`: primary orchestrator
- `kernel-analyze`: read-only implementation analyzer
- `kernel-publish`: validation, build, and upload
- `kernel-bench`: published-artifact HF Jobs benchmark

## Use The Published Kernel

```python
import torch
from kernels import get_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

qwen3_kernels = get_kernel("<namespace>/<repo>", version=1)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

patched = qwen3_kernels.patch_rmsnorm_modules(model)
print(f"Patched RMSNorm modules: {patched}")

inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        pad_token_id=tokenizer.eos_token_id,
    )
```

## Notes

- `setup.py` is kept as a legacy local smoke-test path. The canonical build/publish flow is `nix` plus the Kernels CLI.
- The Python wrapper exposes `patch_rmsnorm_modules(model)` so the same helper works for local smoke tests and Hub-loaded kernels.
- `benchmark_results.json` is a historical sample result file from a prior H100 run; regenerate fresh numbers with `kernels benchmark`.
- Publish builds should run on Linux. HF Jobs is used here for benchmarking the published artifact, not for compiling the kernel bundle.
