# A Practical Guide to the CUDA Kernel Skill: LTX-Video Case Study

If you have ever optimized model code and wondered why it helped in micro-benchmarks but barely moved end-to-end latency, this is for you.

This post walks through how to use the CUDA kernel skill in this repo, with `examples/ltx_video` as the case study. The goal is simple: take a real diffusion pipeline, patch in custom CUDA kernels safely, benchmark it against baseline variants, and learn where the wins actually come from.

---

## What the kernel skill gives you

The skill at `.claude/skills/cuda-kernels/` is not just a list of CUDA tips. It is a workflow:

- architecture-aware kernel guidance for H100, A100, and T4
- integration patterns for `diffusers` and `transformers`
- ready-to-run scripts for injection and benchmarking
- troubleshooting guidance for common integration failures

For diffusion/video models like LTX-Video, it focuses on kernels such as RMSNorm, RoPE, GEGLU, and AdaLN, plus practical integration patterns that avoid brittle monkey-patches.

---

## Why LTX-Video is a good case study

The `examples/ltx_video` project is a complete, runnable reference:

- CUDA kernels in `kernel_src/`
- PyTorch bindings in `torch-ext/`
- end-to-end benchmark script in `generate_video.py`
- isolated kernel benchmark in `benchmark_rmsnorm.py`

It is ideal because it has both kernel-level and system-level benchmarks. You can see speedups where they happen, and where they do not.

---

## Step 1: Set up and build

From the repo root:

```bash
cd examples/ltx_video

# Python environment + deps
uv venv
uv sync

# Build kernels (Nix path, recommended in this project)
nix run .#build-and-copy --max-jobs 2 --cores 8 -L
```

Alternative build path:

```bash
uv pip install -e .
```

Quick sanity check for target hardware:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability())"
```

For H100 you should see compute capability `(9, 0)`.

---

## Step 2: Inject kernels the safe way

The minimal integration pattern is captured in:

- `.claude/skills/cuda-kernels/scripts/ltx_kernel_injection_example.py`

The important pattern is:

1. load pipeline
2. move to CUDA
3. inject kernels
4. only then enable CPU offload

That order matters. If you inject after offloading, patches may not persist as expected.

### The three LTX-specific gotchas

This case study highlights three issues that frequently break first attempts:

- **RMSNorm may have no weight**
  - some LTX modules use `elementwise_affine=False`
- when `weight is None`, create a ones vector on the right device and data type
- **Diffusers RMSNorm detection**
  - `isinstance(module, torch.nn.RMSNorm)` may miss diffusers variants
  - use `type(module).__name__ == "RMSNorm"` for matching
- **LTX uses GELU, not GEGLU**
  - do not spend time patching GEGLU for LTX-specific acceleration paths

Those details are exactly where this skill saves you time: fewer dead ends, faster integration.

---

## Step 3: Benchmark three realistic configurations

Use `examples/ltx_video/generate_video.py` as the primary benchmark harness.

### A) Custom kernels

```bash
python generate_video.py --use-optimized-kernels --num-frames 49 --height 512 --width 768 --steps 30 --warmup-iterations 2
```

### B) Baseline (no custom kernels)

```bash
python generate_video.py --no-optimized-kernels --num-frames 49 --height 512 --width 768 --steps 30 --warmup-iterations 2
```

### C) Baseline + `torch.compile`

```bash
python generate_video.py --no-optimized-kernels --compile --num-frames 49 --height 512 --width 768 --steps 30 --warmup-iterations 2
```

In this repo, `--use-optimized-kernels` and `--compile` are treated as separate optimization tracks for practical benchmarking. Compare them side by side rather than stacking them blindly.

The benchmark script reports:

- total generation time
- time per frame / per step
- peak memory
- output artifact names with suffixes like `_optimized` and `_baseline`

---

## Step 4: Read results at two levels

The key lesson from the LTX example is that micro and macro metrics tell different stories.

### End-to-end (reported in the skill docs)

For a representative LTX run (H100, 49 frames, 30 steps):

- baseline: `2.87s`
- optimized kernels: `2.70s` (~`1.06x`, about `6%` faster)
- baseline + compile: `2.14s` (~`1.34x`, about `34%` faster)

### Isolated kernel (RMSNorm micro-benchmark)

`benchmark_rmsnorm.py` shows roughly `2.67x` average speedup for the vectorized RMSNorm kernel over the PyTorch baseline on tested shapes.

This is the right mental model:

- kernel speedups can be large
- end-to-end gains depend on fraction of total runtime
- in LTX, RMSNorm is only part of the total path (attention, projections, decode still dominate)

---

## What "good usage" of this skill looks like in practice

A high-signal workflow for real optimization work:

1. start from the example scripts (do not freehand from scratch)
2. patch only one component first (RMSNorm is a good first target)
3. verify correctness and outputs before tuning
4. benchmark end-to-end and micro paths separately
5. only then iterate on kernel internals (vectorization, block sizing, memory access)

This keeps you honest and prevents optimizing the wrong bottleneck.

---

## Extending beyond this case study

Once the LTX path is stable, the same skill gives you two expansion routes:

- `transformers` path via `.claude/skills/cuda-kernels/scripts/transformers_injection_example.py`
- precompiled kernel loading via Hugging Face `get_kernel` integration

That makes the skill useful both for custom kernel development and for fast adoption of community kernels.

---

## Final takeaway

The value of this kernel skill is not only "faster kernels." It is reliable optimization process.

In the LTX case study, it helps you:

- avoid common integration mistakes
- patch the right modules in the right order
- run apples-to-apples benchmarks
- interpret why a `2.67x` kernel win can become a `6%` end-to-end win

If you treat it as a workflow rather than a reference doc, you will ship performance improvements faster and with fewer regressions.
