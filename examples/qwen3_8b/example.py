#!/usr/bin/env python3
"""Load the published kernel from the Hub and patch a Qwen3 model."""

from __future__ import annotations

import argparse

import torch
from kernels import get_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Published kernel repo id")
    parser.add_argument("--version", type=int, default=1, help="Kernel major version")
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B", help="Transformers model id")
    parser.add_argument("--prompt", default="The capital of France is", help="Prompt to generate from")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    qwen3_kernels = get_kernel(args.repo_id, version=args.version)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    patched = qwen3_kernels.patch_rmsnorm_modules(model)
    print(f"Patched RMSNorm modules: {patched}")

    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
