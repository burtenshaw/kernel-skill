---
description: Benchmarks the published Qwen3 RMSNorm kernel on Hugging Face Jobs and returns structured results.
mode: subagent
---

Benchmark the published artifact, not an unpublished checkout.

Use the tracked helper:
- `uv run examples/qwen3_8b/scripts/hf_jobs_benchmark.py --repo-id <repo-id>`

Behavior requirements:
- default to version `1`
- use the helper script's hardware detection
- prefer Hopper-class flavors (`h100` if available, otherwise `h200` variants)
- do not switch to `a100-large` unless the kernel's `cuda-capabilities` were widened to include `8.0`
- wait for the job to complete and return the final JSON payload

Return:
- repo id and version
- selected HF Jobs flavor
- job id and job URL
- workload-level benchmark highlights
- any failure details if the job does not complete successfully

Do not edit files.
