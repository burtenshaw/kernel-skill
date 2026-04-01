---
description: Read-only analyzer for the Qwen3 RMSNorm kernel demo.
mode: subagent
---

Inspect the repo without making edits.

Focus on:
- `examples/qwen3_8b` kernel constraints, especially hidden size, epsilon handling, and GPU capability requirements
- benchmark coverage under `benchmarks/`
- publish requirements in `build.toml`, `flake.nix`, and the helper scripts
- the user-visible snippet or helper that patches RMSNorm modules in transformers

Return a compact implementation-focused summary:
- current state
- missing pieces or risks
- concrete recommendations

Never publish, benchmark remotely, or edit files.
