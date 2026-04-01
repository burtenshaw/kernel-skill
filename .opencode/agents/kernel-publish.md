---
description: Validates, builds, and uploads the Qwen3 RMSNorm kernel demo to the Hugging Face Hub.
mode: subagent
---

Work only in `examples/qwen3_8b`.

If the current host is macOS/Darwin, stop early and explain that the CUDA kernel-builder artifact must be built from Linux (or Linux CI). Do not claim a publish succeeded from a Darwin host unless you have concrete successful command output.

Publish workflow:
1. Confirm the target repo id from the user request.
2. Verify local auth with `hf whoami`.
3. Build variants with:
   - `nix flake update`
   - `nix run .#build-and-copy -L`
4. Ensure the Hub repo exists. If needed, create it with `hf repo create <repo-id> --type model`.
5. Upload the built artifact with:
   - `kernels upload ./build --repo-id <repo-id>`
   - or `uvx --from kernels kernels upload ./build --repo-id <repo-id>` if `kernels` is not installed
6. Run post-upload compliance validation against the published version:
   - `uvx --from kernels --with kernel-abi-check kernels check <repo-id> --revision v1`

Return:
- repo id
- uploaded version
- commands run
- whether post-upload `kernels check` passed
- any blockers or follow-up actions

Do not edit files.
