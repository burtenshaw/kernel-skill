/*
 * RMSNorm CUDA Kernel for LTX-Video
 * Optimized for NVIDIA H100 (sm_90)
 *
 * RMSNorm formula: output = x * weight / sqrt(mean(x²) + eps)
 * Uses warp-level shuffle reductions for efficient parallel reduction.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

// Warp-level reduction using shuffle operations
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
template <typename T>
__device__ __forceinline__ T block_reduce_sum(T val, T* shared) {
    const int lane = threadIdx.x % 32;
    const int wid = threadIdx.x / 32;

    // Warp-level reduction
    val = warp_reduce_sum(val);

    // Write warp results to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Final reduction in first warp
    const int num_warps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : T(0);
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

// RMSNorm kernel - processes one row per block
template <typename scalar_t, typename acc_t = float>
__global__ void rmsnorm_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    extern __shared__ char smem[];
    acc_t* shared = reinterpret_cast<acc_t*>(smem);

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const scalar_t* row_input = input + row * hidden_size;
    scalar_t* row_output = output + row * hidden_size;

    // Compute sum of squares
    acc_t sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        acc_t val = static_cast<acc_t>(row_input[i]);
        sum_sq += val * val;
    }

    // Reduce across block
    sum_sq = block_reduce_sum(sum_sq, shared);

    // Compute RMS
    __shared__ acc_t rms_inv;
    if (tid == 0) {
        acc_t mean_sq = sum_sq / static_cast<acc_t>(hidden_size);
        rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    // Apply normalization and weight
    for (int i = tid; i < hidden_size; i += stride) {
        acc_t val = static_cast<acc_t>(row_input[i]);
        acc_t w = static_cast<acc_t>(weight[i]);
        row_output[i] = static_cast<scalar_t>(val * rms_inv * w);
    }
}

// Fused RMSNorm + residual add kernel
template <typename scalar_t, typename acc_t = float>
__global__ void rmsnorm_residual_kernel(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ residual_out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    extern __shared__ char smem[];
    acc_t* shared = reinterpret_cast<acc_t*>(smem);

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const scalar_t* row_input = input + row * hidden_size;
    const scalar_t* row_residual = residual + row * hidden_size;
    scalar_t* row_output = output + row * hidden_size;
    scalar_t* row_residual_out = residual_out + row * hidden_size;

    // Compute sum of squares of (input + residual)
    acc_t sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        acc_t val = static_cast<acc_t>(row_input[i]) + static_cast<acc_t>(row_residual[i]);
        sum_sq += val * val;
    }

    // Reduce across block
    sum_sq = block_reduce_sum(sum_sq, shared);

    // Compute RMS
    __shared__ acc_t rms_inv;
    if (tid == 0) {
        acc_t mean_sq = sum_sq / static_cast<acc_t>(hidden_size);
        rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    // Apply normalization, weight, and store residual
    for (int i = tid; i < hidden_size; i += stride) {
        acc_t in_val = static_cast<acc_t>(row_input[i]);
        acc_t res_val = static_cast<acc_t>(row_residual[i]);
        acc_t combined = in_val + res_val;
        acc_t w = static_cast<acc_t>(weight[i]);

        row_residual_out[i] = static_cast<scalar_t>(combined);
        row_output[i] = static_cast<scalar_t>(combined * rms_inv * w);
    }
}

// Launch functions for different data types
extern "C" {

void rmsnorm_forward_fp16(
    __half* output,
    const __half* input,
    const __half* weight,
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const float eps,
    cudaStream_t stream
) {
    const int num_rows = batch_size * seq_len;
    int threads = min(hidden_size, 1024);
    threads = ((threads + 31) / 32) * 32;  // Round to warp boundary

    size_t smem_size = ((threads + 31) / 32) * sizeof(float);

    rmsnorm_kernel<__half><<<num_rows, threads, smem_size, stream>>>(
        output, input, weight, hidden_size, eps
    );
}

void rmsnorm_forward_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const float eps,
    cudaStream_t stream
) {
    const int num_rows = batch_size * seq_len;
    int threads = min(hidden_size, 1024);
    threads = ((threads + 31) / 32) * 32;

    size_t smem_size = ((threads + 31) / 32) * sizeof(float);

    rmsnorm_kernel<__nv_bfloat16><<<num_rows, threads, smem_size, stream>>>(
        output, input, weight, hidden_size, eps
    );
}

void rmsnorm_forward_fp32(
    float* output,
    const float* input,
    const float* weight,
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const float eps,
    cudaStream_t stream
) {
    const int num_rows = batch_size * seq_len;
    int threads = min(hidden_size, 1024);
    threads = ((threads + 31) / 32) * 32;

    size_t smem_size = ((threads + 31) / 32) * sizeof(float);

    rmsnorm_kernel<float><<<num_rows, threads, smem_size, stream>>>(
        output, input, weight, hidden_size, eps
    );
}

// Fused RMSNorm + residual variants
void rmsnorm_residual_forward_fp16(
    __half* output,
    __half* residual_out,
    const __half* input,
    const __half* residual,
    const __half* weight,
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const float eps,
    cudaStream_t stream
) {
    const int num_rows = batch_size * seq_len;
    int threads = min(hidden_size, 1024);
    threads = ((threads + 31) / 32) * 32;

    size_t smem_size = ((threads + 31) / 32) * sizeof(float);

    rmsnorm_residual_kernel<__half><<<num_rows, threads, smem_size, stream>>>(
        output, residual_out, input, residual, weight, hidden_size, eps
    );
}

void rmsnorm_residual_forward_bf16(
    __nv_bfloat16* output,
    __nv_bfloat16* residual_out,
    const __nv_bfloat16* input,
    const __nv_bfloat16* residual,
    const __nv_bfloat16* weight,
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const float eps,
    cudaStream_t stream
) {
    const int num_rows = batch_size * seq_len;
    int threads = min(hidden_size, 1024);
    threads = ((threads + 31) / 32) * 32;

    size_t smem_size = ((threads + 31) / 32) * sizeof(float);

    rmsnorm_residual_kernel<__nv_bfloat16><<<num_rows, threads, smem_size, stream>>>(
        output, residual_out, input, residual, weight, hidden_size, eps
    );
}

void rmsnorm_residual_forward_fp32(
    float* output,
    float* residual_out,
    const float* input,
    const float* residual,
    const float* weight,
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const float eps,
    cudaStream_t stream
) {
    const int num_rows = batch_size * seq_len;
    int threads = min(hidden_size, 1024);
    threads = ((threads + 31) / 32) * 32;

    size_t smem_size = ((threads + 31) / 32) * sizeof(float);

    rmsnorm_residual_kernel<float><<<num_rows, threads, smem_size, stream>>>(
        output, residual_out, input, residual, weight, hidden_size, eps
    );
}

} // extern "C"
