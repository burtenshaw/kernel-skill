# LTX-Video Kernel Benchmarks

Benchmark results for custom CUDA kernels vs PyTorch baseline implementations.

## Hardware

- **GPU**: NVIDIA H100 80GB HBM3
- **Theoretical Memory Bandwidth**: 3,350 GB/s

## RMSNorm Kernel

The RMSNorm kernel uses warp shuffle reductions optimized for H100's 132 SMs.

### Performance Results

| Configuration | Custom Kernel (ms) | PyTorch (ms) | Speedup |
|--------------|-------------------|--------------|---------|
| [1×1024×2048] | 0.039 | 0.064 | 1.64× |
| [2×1024×2048] | 0.040 | 0.073 | 1.82× |
| [4×1024×2048] | 0.052 | 0.093 | 1.78× |
| [1×4096×2048] | 0.052 | 0.093 | 1.79× |
| [2×4096×3072] | 0.102 | 0.209 | 2.04× |
| [1×8192×2048] | 0.083 | 0.150 | 1.81× |
| [4×4096×3072] | 0.173 | 0.393 | 2.26× |

**Average Speedup: 1.88×**

### Correctness Verification

- Max absolute difference: 3.125e-02
- Max relative difference: 8.789e-03
- Status: **PASS** ✓

The small numerical differences are expected with BFloat16 precision (7-bit mantissa).

### Memory Bandwidth Analysis

For the largest workload ([4×4096×3072]):

| Metric | Value |
|--------|-------|
| Data moved | 201.33 MB |
| Achieved bandwidth | 1,160.9 GB/s |
| H100 theoretical | 3,350 GB/s |
| Bandwidth efficiency | 34.7% |

## Benchmark Methodology

- **Warmup iterations**: 20
- **Benchmark iterations**: 100
- **Data type**: BFloat16 (torch.bfloat16)
- **Synchronization**: `torch.cuda.synchronize()` before and after each iteration

## Reproducing Results

```bash
cd examples/ltx_video
python benchmark_rmsnorm.py
```

## Notes

- Speedup increases with larger workloads (up to 2.26× for [4×4096×3072])
- The kernel is optimized for LTX-Video's typical hidden sizes (2048, 3072)
- Bandwidth efficiency of ~35% is reasonable for a memory-bound kernel with reduction operations
