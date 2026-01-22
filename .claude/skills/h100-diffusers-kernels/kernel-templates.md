# CUDA Kernel Templates for H100 Diffusers

Complete, copy-paste ready templates for implementing new kernels.

## Template 1: Element-wise Operation (RoPE style)

Use this pattern for operations that process elements independently.

```cuda
/*
 * Element-wise kernel template for H100 (sm_90)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

constexpr int BLOCK_SIZE = 256;

template <typename scalar_t>
__global__ void your_elementwise_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        float val = float(input[idx]);

        // Your computation here
        float result = val;  // Replace with actual operation

        output[idx] = scalar_t(result);
    }
}

// C++ entry points
extern "C" {

void your_kernel_forward_fp16(
    __half* output,
    const __half* input,
    int total_elements,
    cudaStream_t stream
) {
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    your_elementwise_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        output, input, total_elements
    );
}

void your_kernel_forward_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    int total_elements,
    cudaStream_t stream
) {
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    your_elementwise_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        output, input, total_elements
    );
}

void your_kernel_forward_fp32(
    float* output,
    const float* input,
    int total_elements,
    cudaStream_t stream
) {
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    your_elementwise_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        output, input, total_elements
    );
}

}
```

## Template 2: Row-wise Reduction (LayerNorm style)

Use for operations requiring reduction across a dimension (normalization, softmax).

```cuda
/*
 * Row-wise reduction kernel template for H100 (sm_90)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS = 1024;

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T block_reduce_sum(T val) {
    __shared__ T shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : T(0);
    if (wid == 0) val = warp_reduce_sum(val);

    return val;
}

template <typename scalar_t>
__global__ void your_reduction_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int hidden_size,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const scalar_t* row_input = input + row * hidden_size;
    scalar_t* row_output = output + row * hidden_size;

    // Step 1: Compute reduction (e.g., sum of squares for RMSNorm)
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = float(row_input[i]);
        sum_sq += val * val;
    }
    sum_sq = block_reduce_sum(sum_sq);

    // Step 2: Compute normalization factor
    __shared__ float s_factor;
    if (tid == 0) {
        s_factor = rsqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads();
    float factor = s_factor;

    // Step 3: Apply normalization
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = float(row_input[i]) * factor;
        row_output[i] = scalar_t(normalized * float(weight[i]));
    }
}

// C++ entry points
extern "C" {

void your_reduction_forward_fp16(
    const __half* input,
    const __half* weight,
    __half* output,
    int batch_size,
    int hidden_size,
    float eps,
    cudaStream_t stream
) {
    int threads = min(hidden_size, MAX_THREADS);
    threads = (threads + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;

    your_reduction_kernel<__half><<<batch_size, threads, 0, stream>>>(
        input, weight, output, hidden_size, eps
    );
}

void your_reduction_forward_bf16(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int batch_size,
    int hidden_size,
    float eps,
    cudaStream_t stream
) {
    int threads = min(hidden_size, MAX_THREADS);
    threads = (threads + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;

    your_reduction_kernel<__nv_bfloat16><<<batch_size, threads, 0, stream>>>(
        input, weight, output, hidden_size, eps
    );
}

void your_reduction_forward_fp32(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int hidden_size,
    float eps,
    cudaStream_t stream
) {
    int threads = min(hidden_size, MAX_THREADS);
    threads = (threads + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;

    your_reduction_kernel<float><<<batch_size, threads, 0, stream>>>(
        input, weight, output, hidden_size, eps
    );
}

}
```

## Template 3: Tiled Matrix Operation (Attention style)

Use for operations requiring shared memory tiling (matmul, attention).

```cuda
/*
 * Tiled matrix operation template for H100 (sm_90)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

// Block sizes optimized for H100 L2 cache
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 64;
constexpr int NUM_WARPS = 8;

template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void your_tiled_kernel(
    const scalar_t* __restrict__ A,  // [batch, M, K]
    const scalar_t* __restrict__ B,  // [batch, K, N]
    scalar_t* __restrict__ C,        // [batch, M, N]
    const int batch_size,
    const int M,
    const int N,
    const int K
) {
    // Shared memory for tiles
    extern __shared__ char shared_mem[];
    scalar_t* tile_A = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* tile_B = tile_A + BLOCK_M * BLOCK_K;

    const int batch_idx = blockIdx.z;
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    const int tid = threadIdx.x;

    // Base offsets for this batch
    const scalar_t* batch_A = A + batch_idx * M * K;
    const scalar_t* batch_B = B + batch_idx * K * N;
    scalar_t* batch_C = C + batch_idx * M * N;

    // Initialize accumulator
    float acc[BLOCK_M / (NUM_WARPS * 32)][BLOCK_N / 32] = {0};

    // Iterate over K dimension tiles
    for (int k_tile = 0; k_tile < (K + BLOCK_K - 1) / BLOCK_K; k_tile++) {
        // Cooperative loading of tiles to shared memory
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += blockDim.x) {
            int row = i / BLOCK_K;
            int col = i % BLOCK_K;
            int global_row = block_row * BLOCK_M + row;
            int global_col = k_tile * BLOCK_K + col;

            if (global_row < M && global_col < K) {
                tile_A[i] = batch_A[global_row * K + global_col];
            } else {
                tile_A[i] = scalar_t(0);
            }
        }

        for (int i = tid; i < BLOCK_K * BLOCK_N; i += blockDim.x) {
            int row = i / BLOCK_N;
            int col = i % BLOCK_N;
            int global_row = k_tile * BLOCK_K + row;
            int global_col = block_col * BLOCK_N + col;

            if (global_row < K && global_col < N) {
                tile_B[i] = batch_B[global_row * N + global_col];
            } else {
                tile_B[i] = scalar_t(0);
            }
        }
        __syncthreads();

        // Compute partial results
        // (Simplified - real implementation would use register tiling)
        #pragma unroll
        for (int k = 0; k < BLOCK_K; k++) {
            // Your tiled computation here
        }
        __syncthreads();
    }

    // Write results
    // (Implementation depends on your specific needs)
}

// C++ entry points follow same pattern as above
```

## Template 4: PyTorch Binding

```cpp
// torch_binding.cpp addition

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

extern "C" {
void your_kernel_forward_fp16(const void*, void*, int, cudaStream_t);
void your_kernel_forward_bf16(const void*, void*, int, cudaStream_t);
void your_kernel_forward_fp32(const float*, float*, int, cudaStream_t);
}

void your_kernel_forward(
    torch::Tensor& output,
    const torch::Tensor& input
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");

    const int total_elements = input.numel();

    const at::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (input.scalar_type() == at::kHalf) {
        your_kernel_forward_fp16(
            input.data_ptr(), output.data_ptr(),
            total_elements, stream
        );
    } else if (input.scalar_type() == at::kBFloat16) {
        your_kernel_forward_bf16(
            input.data_ptr(), output.data_ptr(),
            total_elements, stream
        );
    } else if (input.scalar_type() == at::kFloat) {
        your_kernel_forward_fp32(
            static_cast<const float*>(input.data_ptr()),
            static_cast<float*>(output.data_ptr()),
            total_elements, stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }
}

// In TORCH_LIBRARY_EXPAND:
// ops.def("your_kernel_forward(Tensor! out, Tensor input) -> ()");
// ops.impl("your_kernel_forward", torch::kCUDA, &your_kernel_forward);
```

## Template 5: Python API

```python
# In ltx_kernels/__init__.py

def your_kernel(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Your kernel description.

    Args:
        input: Input tensor [batch, seq, hidden]
        out: Optional pre-allocated output tensor

    Returns:
        Output tensor [batch, seq, hidden]
    """
    if out is None:
        out = torch.empty_like(input)

    ops.your_kernel_forward(out, input.contiguous())
    return out
```

## Template 6: build.toml Entry

```toml
[kernel.your_kernel]
backend = "cuda"
depends = []
src = ["kernel_src/your_kernel.cu"]
cuda-capabilities = ["9.0"]
```

## Template 7: Test Case

```python
# In tests/test_kernels.py

import torch
import pytest
from ltx_kernels import your_kernel

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 1024, 2048), (1, 4096, 4096)])
def test_your_kernel(dtype, shape):
    device = "cuda"
    input = torch.randn(shape, dtype=dtype, device=device)

    # Reference implementation
    expected = your_reference_implementation(input)

    # Kernel implementation
    output = your_kernel(input)

    # Compare
    rtol = 1e-2 if dtype == torch.float16 else 1e-4
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)

def test_your_kernel_with_preallocated():
    device = "cuda"
    dtype = torch.bfloat16
    shape = (2, 1024, 2048)

    input = torch.randn(shape, dtype=dtype, device=device)
    output = torch.empty_like(input)

    result = your_kernel(input, out=output)

    assert result is output  # Verify in-place
```
