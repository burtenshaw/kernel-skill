#!/usr/bin/env python3
"""
Micro-benchmark for RMSNorm kernel optimized for Qwen3-8B.

Qwen3-8B Configuration:
- hidden_size: 4096
- num_hidden_layers: 32
- rms_norm_eps: 1e-6
- RMSNorm modules: 65 (32 layers * 2 + 1 final)

Compares:
1. Custom CUDA kernel (vectorized BF16/FP16)
2. PyTorch baseline implementation

Usage:
    cd examples/qwen3_8b
    uv pip install -e .
    python benchmark_rmsnorm.py
"""

import torch
import time
import argparse
from typing import Tuple, Dict, List

# Import custom kernel
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'torch-ext'))

try:
    from qwen3_kernels import rmsnorm
    KERNELS_AVAILABLE = True
except ImportError:
    KERNELS_AVAILABLE = False
    print("Warning: Custom kernels not built. Run 'uv pip install -e .' first.")


def pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference PyTorch implementation of RMSNorm."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def benchmark_kernel(
    func,
    args,
    warmup: int = 10,
    iterations: int = 100,
    name: str = "kernel"
) -> Tuple[float, float, float]:
    """Benchmark a kernel function."""
    # Warmup
    for _ in range(warmup):
        _ = func(*args)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = func(*args)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    avg_time = sum(times) / len(times)
    min_time = min(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    return avg_time, min_time, std_time


def run_microbenchmark(dtype: torch.dtype = torch.bfloat16) -> Dict:
    """Run comprehensive RMSNorm benchmarks for Qwen3-8B dimensions."""
    print("=" * 80)
    print("RMSNorm Micro-Benchmark: Qwen3-8B Kernel vs PyTorch Baseline")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Dtype: {dtype}")
    print()

    # Qwen3-8B specific configurations
    # hidden_size = 4096
    QWEN3_HIDDEN = 4096
    QWEN3_EPS = 1e-6

    configs = [
        # (batch_size, seq_len, hidden_size) - typical Qwen3 inference scenarios
        (1, 128, QWEN3_HIDDEN),      # Short prompt
        (1, 512, QWEN3_HIDDEN),      # Medium prompt
        (1, 1024, QWEN3_HIDDEN),     # Long prompt
        (1, 2048, QWEN3_HIDDEN),     # Very long prompt
        (1, 4096, QWEN3_HIDDEN),     # Max context chunks
        (4, 512, QWEN3_HIDDEN),      # Batched inference
        (8, 256, QWEN3_HIDDEN),      # Larger batch, shorter seq
        (1, 8192, QWEN3_HIDDEN),     # Extended context
    ]

    print(f"{'Config':<25} {'Custom (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
    print("-" * 80)

    results = []
    total_speedup = 0
    num_configs = 0

    for batch, seq, hidden in configs:
        # Create input tensors
        x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
        weight = torch.ones(hidden, dtype=dtype, device="cuda")

        # Benchmark custom kernel
        if KERNELS_AVAILABLE:
            custom_avg, custom_min, custom_std = benchmark_kernel(
                rmsnorm, (x, weight, QWEN3_EPS),
                warmup=20, iterations=100, name="custom"
            )
        else:
            custom_avg, custom_min, custom_std = 0, 0, 0

        # Benchmark PyTorch baseline
        pytorch_avg, pytorch_min, pytorch_std = benchmark_kernel(
            pytorch_rmsnorm, (x, weight, QWEN3_EPS),
            warmup=20, iterations=100, name="pytorch"
        )

        # Calculate speedup
        speedup = pytorch_avg / custom_avg if custom_avg > 0 else 0
        total_speedup += speedup
        num_configs += 1

        config_str = f"[{batch}x{seq}x{hidden}]"
        if KERNELS_AVAILABLE:
            print(f"{config_str:<25} {custom_avg:>12.3f}   {pytorch_avg:>12.3f}   {speedup:>8.2f}x")
        else:
            print(f"{config_str:<25} {'N/A':>12}   {pytorch_avg:>12.3f}   {'N/A':>8}")

        results.append({
            "config": f"{batch}x{seq}x{hidden}",
            "batch": batch,
            "seq_len": seq,
            "hidden_size": hidden,
            "custom_ms": custom_avg,
            "pytorch_ms": pytorch_avg,
            "speedup": speedup,
        })

    avg_speedup = total_speedup / num_configs if num_configs > 0 else 0
    print("-" * 80)
    if KERNELS_AVAILABLE:
        print(f"{'Average Speedup:':<55} {avg_speedup:.2f}x")
    print()

    return {"results": results, "avg_speedup": avg_speedup}


def verify_correctness(dtype: torch.dtype = torch.bfloat16) -> bool:
    """Verify kernel correctness against PyTorch reference."""
    if not KERNELS_AVAILABLE:
        print("Correctness Check: SKIPPED (kernels not available)")
        return False

    print("Correctness Check:")
    QWEN3_HIDDEN = 4096
    QWEN3_EPS = 1e-6

    x = torch.randn(2, 1024, QWEN3_HIDDEN, dtype=dtype, device="cuda")
    weight = torch.ones(QWEN3_HIDDEN, dtype=dtype, device="cuda")

    custom_out = rmsnorm(x, weight, QWEN3_EPS)
    pytorch_out = pytorch_rmsnorm(x, weight, QWEN3_EPS)

    max_diff = (custom_out - pytorch_out).abs().max().item()
    rel_diff = ((custom_out - pytorch_out).abs() / (pytorch_out.abs() + 1e-8)).max().item()

    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Max relative difference: {rel_diff:.6e}")

    # BFloat16 has 7 bits mantissa, FP16 has 10 bits
    tolerance = 0.05 if dtype == torch.bfloat16 else 0.01
    passed = max_diff < tolerance
    print(f"  Correctness: {'PASS' if passed else 'FAIL'}")
    print()

    return passed


def analyze_bandwidth(dtype: torch.dtype = torch.bfloat16) -> Dict:
    """Analyze memory bandwidth utilization."""
    if not KERNELS_AVAILABLE:
        print("Bandwidth Analysis: SKIPPED (kernels not available)")
        return {}

    print("Memory Bandwidth Analysis (Qwen3-8B typical workload):")
    QWEN3_HIDDEN = 4096

    # Typical Qwen3 inference workload
    batch, seq, hidden = 1, 2048, QWEN3_HIDDEN
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    weight = torch.ones(hidden, dtype=dtype, device="cuda")

    # Bytes moved: read input + read weight + write output
    bytes_per_elem = 2 if dtype in [torch.float16, torch.bfloat16] else 4
    input_bytes = batch * seq * hidden * bytes_per_elem
    weight_bytes = hidden * bytes_per_elem
    output_bytes = batch * seq * hidden * bytes_per_elem
    total_bytes = input_bytes + weight_bytes + output_bytes

    custom_avg, _, _ = benchmark_kernel(rmsnorm, (x, weight, 1e-6), warmup=20, iterations=100)

    bandwidth_gbps = (total_bytes / 1e9) / (custom_avg / 1000)
    theoretical_bandwidth = 3350  # H100 theoretical 3.35 TB/s
    bandwidth_efficiency = (bandwidth_gbps / theoretical_bandwidth) * 100

    print(f"  Config: [{batch}x{seq}x{hidden}]")
    print(f"  Total data moved: {total_bytes / 1e6:.2f} MB")
    print(f"  Kernel latency: {custom_avg:.3f} ms")
    print(f"  Achieved bandwidth: {bandwidth_gbps:.1f} GB/s")
    print(f"  H100 theoretical: {theoretical_bandwidth} GB/s")
    print(f"  Bandwidth efficiency: {bandwidth_efficiency:.1f}%")
    print()

    return {
        "total_bytes_mb": total_bytes / 1e6,
        "latency_ms": custom_avg,
        "bandwidth_gbps": bandwidth_gbps,
        "efficiency_pct": bandwidth_efficiency,
    }


def estimate_model_impact() -> None:
    """Estimate impact on full Qwen3-8B model inference."""
    print("Estimated Model Impact (Qwen3-8B):")
    print("  RMSNorm modules: 65 (32 layers * 2 + 1 final)")
    print("  Typical RMSNorm % of compute: ~3-5%")
    print("  Expected end-to-end speedup: ~2-3% (bounded by attention/linear ops)")
    print()
    print("  Note: For larger speedups, combine with:")
    print("    - Flash Attention 2 (attn_implementation='flash_attention_2')")
    print("    - torch.compile (for linear/other ops)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-8B RMSNorm kernel")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                       help="Data type for benchmarking")
    parser.add_argument("--skip-correctness", action="store_true",
                       help="Skip correctness verification")
    parser.add_argument("--skip-bandwidth", action="store_true",
                       help="Skip bandwidth analysis")
    args = parser.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Run benchmarks
    benchmark_results = run_microbenchmark(dtype)

    # Verify correctness
    if not args.skip_correctness:
        verify_correctness(dtype)

    # Analyze bandwidth
    if not args.skip_bandwidth:
        analyze_bandwidth(dtype)

    # Estimate model impact
    estimate_model_impact()

    return benchmark_results


if __name__ == "__main__":
    main()
