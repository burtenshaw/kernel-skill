# Troubleshooting Guide

Common issues and solutions when working with H100 CUDA kernels for diffusers.

## Build Issues

### 1. Type Conversion Errors with FP16/BF16

**Problem:** PyTorch compiles with `-D__CUDA_NO_HALF_OPERATORS__` which disables implicit type conversions:
```
error: no suitable conversion function from "__half" to "float" exists
```

**Solution:** Add explicit type conversion helper functions in your .cu files:
```cuda
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Type conversion helpers (required for PyTorch compatibility)
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float from_float(float x, float*) { return x; }
__device__ __forceinline__ __half from_float(float x, __half*) { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat16 from_float(float x, __nv_bfloat16*) { return __float2bfloat16(x); }

// Usage in kernels:
float val = to_float(input[idx]);
output[idx] = from_float(result, (scalar_t*)nullptr);
```

### 2. Missing CUDA Headers in torch_binding.cpp

**Problem:** Undeclared types `__half`, `__nv_bfloat16`

**Solution:** Include required headers:
```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAGuard.h>
```

### 3. Build Fails with "No module named torch"

**Solution:** Add torch to build dependencies in pyproject.toml:
```toml
[build-system]
requires = ["setuptools", "wheel", "torch>=2.0"]
```

## Performance Issues

### 4. Bank Conflicts in Shared Memory

**Problem:** Poor performance due to shared memory bank conflicts.

**Solution:** Add padding for 32-bank conflict avoidance:
```cuda
__shared__ float data[32][33];  // 33 instead of 32
```

### 5. Poor Occupancy

**Problem:** Low SM utilization.

**Solution:** Check register usage:
```bash
nvcc --ptxas-options=-v your_kernel.cu
```

### 6. Memory Coalescing

**Problem:** Poor memory bandwidth utilization.

**Solution:** Ensure 128-byte aligned accesses for optimal bandwidth.

## Integration Issues

### 7. AttributeError: 'NoneType' has no attribute 'contiguous' (RMSNorm weight is None)

**Problem:** Model uses `elementwise_affine=False`, so `module.weight` is `None`:
```
AttributeError: 'NoneType' object has no attribute 'contiguous'
```

**Root Cause:** LTX-Video transformer blocks use `RMSNorm(dim, elementwise_affine=False)` which has no learnable weight parameter.

**Solution:** Check if weight exists before using it:
```python
has_weight = hasattr(module, 'weight') and module.weight is not None
if has_weight:
    output = rmsnorm(x, module.weight, eps=eps)
else:
    # Create weight of ones
    weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    output = rmsnorm(x, weight, eps=eps)
```

### 8. GEGLU Kernel Not Being Used

**Problem:** You patched GEGLU modules but the kernel isn't being called.

**Diagnosis:** Check what activation the model actually uses:
```python
for name, module in model.named_modules():
    if 'GEGLU' in type(module).__name__:
        print(f"Found GEGLU: {name}")
    if 'GELU' in type(module).__name__:
        print(f"Found GELU: {name}")
```

**Solution:** LTX-Video uses `GELU`, not `GEGLU`. Only patch GEGLU for models that actually use it (e.g., SD3, FLUX).

### 9. Kernel Patching Doesn't Persist Through CPU Offloading

**Problem:** After `enable_model_cpu_offload()`, patched modules don't work correctly.

**Solution:** Inject kernels AFTER loading model to CUDA, BEFORE enabling offloading:
```python
pipe = LTXPipeline.from_pretrained(...)
pipe.to("cuda")  # Move to CUDA first
inject_optimized_kernels(pipe)  # Patch modules
pipe.enable_model_cpu_offload()  # Now enable offloading
```

### 10. isinstance() Check Misses Diffusers Modules

**Problem:** `isinstance(module, torch.nn.RMSNorm)` returns `False` for diffusers modules.

**Root Cause:** Diffusers has its own `RMSNorm` class that is NOT a subclass of `torch.nn.RMSNorm`:
```python
from diffusers.models.normalization import RMSNorm
# This is a DIFFERENT class from torch.nn.RMSNorm!
```

**Solution:** Check by class name instead:
```python
# WRONG - misses diffusers RMSNorm
if isinstance(module, torch.nn.RMSNorm):

# CORRECT - catches all RMSNorm variants
if type(module).__name__ == 'RMSNorm':
```

## Debugging Tips

### Profile Your Kernels

```bash
# NVIDIA Nsight Systems (system-wide overview)
nsys profile -o kernel_profile python your_script.py

# NVIDIA Nsight Compute (detailed kernel analysis)
ncu --set full --csv -o metrics.csv python your_script.py
```

### Verify Kernel Injection

```python
# Check if attention processors were replaced
for name, module in pipe.transformer.named_modules():
    if hasattr(module, 'processor'):
        print(f"{name}: {type(module.processor).__name__}")
        break

# Test a forward pass through patched modules
with torch.inference_mode():
    x = torch.randn(1, 100, 2048, device='cuda', dtype=torch.bfloat16)
    for name, module in pipe.transformer.named_modules():
        if type(module).__name__ == 'RMSNorm':
            out = module(x)
            print(f"RMSNorm forward pass: {x.shape} -> {out.shape}")
            break
```

### Check CUDA Architecture

```bash
# Verify H100 is detected
python -c "import torch; print(torch.cuda.get_device_capability())"
# Should print (9, 0) for H100
```

### Verify Kernels Are Built

```bash
# Check for compiled .so files
ls torch-ext/ltx_kernels/_ops*.so

# Try importing
python -c "from ltx_kernels import rmsnorm; print('OK')"
```
