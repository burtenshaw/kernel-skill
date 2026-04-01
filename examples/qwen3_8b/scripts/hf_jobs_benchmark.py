#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=1.0"]
# ///

"""Benchmark a published kernel repo on Hugging Face Jobs."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from typing import Iterable

from huggingface_hub import fetch_job_logs, get_token, inspect_job, run_job

JSON_START = "__KERNELS_BENCHMARK_JSON__"
JSON_END = "__KERNELS_BENCHMARK_JSON_END__"
HOPPER_FLAVORS = ("h100", "h200", "h200x2", "h200x4", "h200x8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hub repo id to benchmark")
    parser.add_argument("--version", type=int, default=1, help="Kernel major version")
    parser.add_argument("--namespace", help="Run the job in a specific user or org namespace")
    parser.add_argument(
        "--flavor",
        help="Override the HF Jobs hardware flavor. Must be Hopper-compatible for this demo.",
    )
    parser.add_argument("--timeout", default="90m", help="HF Jobs timeout")
    parser.add_argument("--warmup", type=int, default=20, help="Benchmark warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark timed iterations")
    parser.add_argument(
        "--image",
        default="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        help="Docker image used for the benchmark job",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=15,
        help="Seconds to wait between job status checks",
    )
    return parser.parse_args()


def list_job_flavors() -> list[str]:
    output = subprocess.check_output(["hf", "jobs", "hardware"], text=True)
    flavors: list[str] = []
    for line in output.splitlines():
        if not line or line.startswith("NAME") or line.startswith("---"):
            continue
        parts = line.split()
        if parts:
            flavors.append(parts[0])
    return flavors


def select_flavor(requested: str | None, available: Iterable[str]) -> str:
    available_set = set(available)
    if requested:
        if requested not in available_set:
            raise SystemExit(f"Requested flavor '{requested}' is not available in `hf jobs hardware`.")
        if requested not in HOPPER_FLAVORS:
            raise SystemExit(
                "This demo is compiled only for CUDA capability 9.0 (Hopper). "
                f"Requested flavor '{requested}' is not Hopper-compatible."
            )
        return requested

    for flavor in HOPPER_FLAVORS:
        if flavor in available_set:
            return flavor

    raise SystemExit(
        "No Hopper-class HF Jobs flavor is available. This demo targets compute capability 9.0, "
        "so `a100-large` is not a compatible fallback unless you widen `cuda-capabilities` in build.toml."
    )


def build_job_command(repo_ref: str, warmup: int, iterations: int) -> str:
    return "\n".join(
        [
            "set -euo pipefail",
            "python -m pip install --upgrade pip",
            "python -m pip install 'kernels[benchmark]'",
            (
                "kernels benchmark "
                f"{shlex.quote(repo_ref)} --warmup {warmup} --iterations {iterations} "
                "--output /tmp/results.json"
            ),
            f"printf '\\n{JSON_START}\\n'",
            "cat /tmp/results.json",
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
        raise SystemExit("Completed benchmark job but could not find JSON results in the logs.")
    return json.loads(match.group(1))


def main() -> int:
    args = parse_args()

    token = get_token()
    if token is None:
        raise SystemExit("No Hugging Face token is available. Run `hf auth login` first.")

    available_flavors = list_job_flavors()
    selected_flavor = select_flavor(args.flavor, available_flavors)
    repo_ref = f"{args.repo_id}@v{args.version}"
    command = build_job_command(repo_ref, args.warmup, args.iterations)

    job = run_job(
        image=args.image,
        command=["/bin/bash", "-lc", command],
        flavor=selected_flavor,
        timeout=args.timeout,
        namespace=args.namespace,
        secrets={"HF_TOKEN": token},
    )

    print(
        json.dumps(
            {
                "job_id": job.id,
                "job_url": job.url,
                "repo_id": args.repo_id,
                "version": args.version,
                "selected_flavor": selected_flavor,
            },
            indent=2,
        )
    )

    info, logs = wait_for_job(job.id, args.poll_interval)
    if info.status.stage != "COMPLETED":
        print(logs)
        raise SystemExit(f"HF Jobs benchmark failed with status {info.status.stage}.")

    results = extract_results(logs)
    payload = {
        "job_id": job.id,
        "job_url": job.url,
        "repo_id": args.repo_id,
        "version": args.version,
        "selected_flavor": selected_flavor,
        "benchmark_results": results,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
