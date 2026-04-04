#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=1.0"]
# ///

"""Build, publish, and benchmark the Qwen3 RMSNorm kernel on HF Jobs (H200).

This script submits a single HF Job that:
1. Clones the kernel-skill repo from GitHub
2. Builds the Qwen3 RMSNorm kernel with Nix (sm_90 for Hopper/H200)
3. Publishes to the Hugging Face Kernel Hub
4. Benchmarks on H200 hardware

Usage:
    uv run scripts/hf_jobs_build_publish_benchmark.py --repo-id burtenshaw/qwen3-rmsnorm-h200
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time

from huggingface_hub import fetch_job_logs, get_token, inspect_job, run_job

JSON_START = "__KERNELS_RESULTS_JSON__"
JSON_END = "__KERNELS_RESULTS_JSON_END__"
HOPPER_FLAVORS = ("h200", "h200x2", "h200x4", "h200x8", "h100")

# GitHub repo to clone (adjust if different)
KERNEL_SKILL_REPO = "https://github.com/burtenshaw/kernel-skill"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hub repo id to publish to")
    parser.add_argument("--version", type=int, default=1, help="Kernel major version")
    parser.add_argument(
        "--github-repo",
        default=KERNEL_SKILL_REPO,
        help="GitHub repo to clone (default: huggingface/kernel-skill)",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Git branch to clone (default: repository default branch)",
    )
    parser.add_argument(
        "--flavor",
        default="h200",
        help="HF Jobs hardware flavor (default: h200)",
    )
    parser.add_argument("--timeout", default="120m", help="HF Jobs timeout")
    parser.add_argument(
        "--warmup", type=int, default=20, help="Benchmark warmup iterations"
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Benchmark timed iterations"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=15,
        help="Seconds to wait between job status checks",
    )
    return parser.parse_args()


def build_publish_benchmark_script(
    repo_id: str,
    version: int,
    warmup: int,
    iterations: int,
    github_repo: str,
    branch: str | None = None,
) -> str:
    """Generate the bash script to run on HF Jobs."""
    return "\n".join(
        [
            "set -euo pipefail",
            "",
            "# ========================================",
            "# Step 1: Install Nix",
            "# ========================================",
            "echo '=== Installing Nix ==='",
            "apt-get update && apt-get install -y curl xz-utils git",
            "mkdir -m 0755 /nix",
            "groupadd -r nixbld || true",
            "for n in $(seq 1 10); do useradd -c 'Nix build user $n' -d /var/empty -g nixbld -G nixbld -M -N -r -s /run/current-system/sw/bin/nologin nixbld$n || true; done",
            "curl -L https://nixos.org/nix/install | sh",
            "export PATH=$HOME/.nix-profile/bin:$PATH",
            "",
            "# ========================================",
            "# Step 2: Install Python dependencies",
            "# ========================================",
            "echo '=== Installing dependencies ==='",
            "python3 -m pip install --upgrade pip",
            "python3 -m pip install 'kernels[benchmark]' hf-transfer",
            "",
            "# Enable fast uploads",
            "export HF_HUB_ENABLE_HF_TRANSFER=1",
            "",
            "# ========================================",
            "# Step 3: Clone the repo",
            "# ========================================",
            f"echo '=== Cloning {github_repo} ==='",
            f"git clone --depth 1{' --branch ' + branch if branch else ''} {github_repo} /workspace/kernel-skill",
            "cd /workspace/kernel-skill/examples/qwen3_8b",
            "",
            "# ========================================",
            "# Step 4: Build the kernel with Nix",
            "# ========================================",
            "echo '=== Building kernel with Nix ==='",
            "nix --extra-experimental-features 'nix-command flakes' flake update",
            "nix --extra-experimental-features 'nix-command flakes' run .#build-and-copy -L",
            "",
            "# Verify build output exists",
            "echo '=== Build artifacts ==='",
            "ls -la build/",
            "",
            "# ========================================",
            "# Step 5: Publish to Hub",
            "# ========================================",
            "echo '=== Publishing to Hub ==='",
            f"hf repo create {repo_id} --type kernel || true",
            f"kernels upload ./build --repo-id {repo_id}",
            "",
            "# ========================================",
            "# Step 6: Verify the published artifact",
            "# ========================================",
            "echo '=== Verifying published artifact ==='",
            f"kernels check {repo_id} --revision v{version}",
            "",
            "# ========================================",
            "# Step 7: Benchmark on H200",
            "# ========================================",
            "echo '=== Benchmarking on H200 ==='",
            f"kernels benchmark {repo_id}@v{version} --warmup {warmup} --iterations {iterations} --output /tmp/benchmark_results.json",
            "",
            "# Output results as JSON marker",
            f"printf '\\n{JSON_START}\\n'",
            "cat /tmp/benchmark_results.json",
            f"printf '\\n{JSON_END}\\n'",
        ]
    )


def wait_for_job(job_id: str, poll_interval: int) -> tuple[object, str]:
    stages_seen: set[str] = set()
    while True:
        info = inspect_job(job_id=job_id)
        stage = info.status.stage
        if stage not in stages_seen:
            print(f"[hf-jobs] {job_id}: {stage}", file=sys.stderr)
            stages_seen.add(stage)
        if stage in {"COMPLETED", "ERROR", "CANCELED", "CANCELLED"}:
            logs = "\n".join(fetch_job_logs(job_id=job_id))
            return info, logs
        time.sleep(poll_interval)


def extract_results(logs: str) -> dict:
    pattern = re.compile(f"{JSON_START}\\n(.*)\\n{JSON_END}", re.DOTALL)
    match = pattern.search(logs)
    if not match:
        raise SystemExit("Completed job but could not find JSON results in the logs.")
    return json.loads(match.group(1))


def main() -> int:
    args = parse_args()

    token = get_token()
    if token is None:
        raise SystemExit(
            "No Hugging Face token is available. Run `hf auth login` first."
        )

    script = build_publish_benchmark_script(
        args.repo_id,
        args.version,
        args.warmup,
        args.iterations,
        args.github_repo,
        args.branch,
    )

    # Run the job on H200 with Python image + Nix
    job = run_job(
        image="python:3.11-slim",
        command=["/bin/bash", "-lc", script],
        flavor=args.flavor,
        timeout=args.timeout,
        secrets={"HF_TOKEN": token},
    )

    print(
        json.dumps(
            {
                "job_id": job.id,
                "job_url": job.url,
                "repo_id": args.repo_id,
                "version": args.version,
                "selected_flavor": args.flavor,
            },
            indent=2,
        )
    )

    info, logs = wait_for_job(job.id, args.poll_interval)
    if info.status.stage != "COMPLETED":
        print(logs)
        raise SystemExit(f"HF Jobs failed with status {info.status.stage}.")

    results = extract_results(logs)
    payload = {
        "job_id": job.id,
        "job_url": job.url,
        "repo_id": args.repo_id,
        "version": args.version,
        "selected_flavor": args.flavor,
        "benchmark_results": results,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
