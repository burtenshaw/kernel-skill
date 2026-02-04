/*
 * Optimized RMSNorm CUDA Kernel for LTX-Video
 * Optimized for NVIDIA H100 (sm_90)
 *
 * RMSNorm formula: output = x * weight / sqrt(mean(x²) + eps)
 *
 * H100 Optimizations:
 * - Vectorized loads/stores (float4/bfloat162) for maximum memory bandwidth
 * - Warp shuffle reductions (no shared memory bank conflicts)
 * - Coalesced memory access patterns
 * - Register-level accumulation to reduce memory traffic
 * - Block size tuned for 132 SMs
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS = 1024;

// Number of elements each thread processes (for vectorization)
constexpr int ELEMENTS_PER_THREAD = 8;

// Warp-level reduction using shuffle operations
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
template <typename T>
__device__ __forceinline__ T block_reduce_sum(T val, T* shared) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;

    // Warp-level reduction
    val = warp_reduce_sum(val);

    // Write warp results to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Final reduction in first warp
    const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : T(0);
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

// Helper functions for type conversion
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float from_float(float x, float*) { return x; }
__device__ __forceinline__ __half from_float(float x, __half*) { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat16 from_float(float x, __nv_bfloat16*) { return __float2bfloat16(x); }

// ============================================================================
// Vectorized RMSNorm kernel - OPTIMIZED VERSION
// Uses float4 loads for better memory bandwidth utilization
// ============================================================================
template <typename scalar_t, typename acc_t = float>
__global__ void rmsnorm_kernel_vectorized(
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

    // Phase 1: Compute sum of squares with vectorized loads
    // Process 4 elements at a time for better memory bandwidth
    acc_t sum_sq = 0.0f;

    // Vector processing for aligned portions (4 elements at a time)
    const int vec_size = 4;
    const int vec_hidden = hidden_size / vec_size;

    // Use float4 for vectorized loads when hidden_size is divisible by 4
    if (hidden_size % vec_size == 0) {
        const float4* vec_input = reinterpret_cast<const float4*>(row_input);

        for (int i = tid; i < vec_hidden; i += stride) {
            float4 v;
            // Load 4 elements at once
            if constexpr (sizeof(scalar_t) == 4) {
                v = vec_input[i];
            } else {
                // For FP16/BF16, load as bytes and convert
                const scalar_t* ptr = row_input + i * vec_size;
                v.x = to_float(ptr[0]);
                v.y = to_float(ptr[1]);
                v.z = to_float(ptr[2]);
                v.w = to_float(ptr[3]);
            }

            sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        // Fallback for non-aligned sizes
        for (int i = tid; i < hidden_size; i += stride) {
            acc_t val = to_float(row_input[i]);
            sum_sq += val * val;
        }
    }

    // Reduce across block
    sum_sq = block_reduce_sum(sum_sq, shared);

    // Compute RMS inverse
    __shared__ acc_t rms_inv;
    if (tid == 0) {
        acc_t mean_sq = sum_sq / static_cast<acc_t>(hidden_size);
        rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    const acc_t factor = rms_inv;

    // Phase 2: Apply normalization and weight with vectorized stores
    if (hidden_size % vec_size == 0) {
        for (int i = tid; i < vec_hidden; i += stride) {
            const int base_idx = i * vec_size;

            // Load input values
            acc_t v0 = to_float(row_input[base_idx]);
            acc_t v1 = to_float(row_input[base_idx + 1]);
            acc_t v2 = to_float(row_input[base_idx + 2]);
            acc_t v3 = to_float(row_input[base_idx + 3]);

            // Load weights
            acc_t w0 = to_float(weight[base_idx]);
            acc_t w1 = to_float(weight[base_idx + 1]);
            acc_t w2 = to_float(weight[base_idx + 2]);
            acc_t w3 = to_float(weight[base_idx + 3]);

            // Apply normalization and weight
            row_output[base_idx] = from_float(v0 * factor * w0, (scalar_t*)nullptr);
            row_output[base_idx + 1] = from_float(v1 * factor * w1, (scalar_t*)nullptr);
            row_output[base_idx + 2] = from_float(v2 * factor * w2, (scalar_t*)nullptr);
            row_output[base_idx + 3] = from_float(v3 * factor * w3, (scalar_t*)nullptr);
        }
    } else {
        for (int i = tid; i < hidden_size; i += stride) {
            acc_t val = to_float(row_input[i]);
            acc_t w = to_float(weight[i]);
            row_output[i] = from_float(val * factor * w, (scalar_t*)nullptr);
        }
    }
}

// ============================================================================
// BF16-specific optimized kernel using __nv_bfloat162 for 2-element vectorization
// ============================================================================
__global__ void rmsnorm_kernel_bf16_vectorized(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    extern __shared__ char smem[];
    float* shared = reinterpret_cast<float*>(smem);

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const __nv_bfloat16* row_input = input + row * hidden_size;
    __nv_bfloat16* row_output = output + row * hidden_size;

    // Phase 1: Compute sum of squares with bf16x2 vectorized loads
    float sum_sq = 0.0f;

    // Use __nv_bfloat162 for 2-element vectorized loads
    const int vec_hidden = hidden_size / 2;
    const __nv_bfloat162* vec_input = reinterpret_cast<const __nv_bfloat162*>(row_input);

    #pragma unroll 4
    for (int i = tid; i < vec_hidden; i += stride) {
        __nv_bfloat162 v = vec_input[i];
        float v0 = __bfloat162float(v.x);
        float v1 = __bfloat162float(v.y);
        sum_sq += v0 * v0 + v1 * v1;
    }

    // Handle odd element if hidden_size is odd
    if (hidden_size % 2 == 1 && tid == 0) {
        float v = __bfloat162float(row_input[hidden_size - 1]);
        sum_sq += v * v;
    }

    // Reduce across block
    sum_sq = block_reduce_sum(sum_sq, shared);

    // Compute RMS inverse
    __shared__ float rms_inv;
    if (tid == 0) {
        float mean_sq = sum_sq / static_cast<float>(hidden_size);
        rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    const float factor = rms_inv;

    // Phase 2: Apply normalization and weight with bf16x2 vectorized stores
    const __nv_bfloat162* vec_weight = reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* vec_output = reinterpret_cast<__nv_bfloat162*>(row_output);

    #pragma unroll 4
    for (int i = tid; i < vec_hidden; i += stride) {
        __nv_bfloat162 v_in = vec_input[i];
        __nv_bfloat162 v_w = vec_weight[i];

        float v0 = __bfloat162float(v_in.x);
        float v1 = __bfloat162float(v_in.y);
        float w0 = __bfloat162float(v_w.x);
        float w1 = __bfloat162float(v_w.y);

        __nv_bfloat162 result;
        result.x = __float2bfloat16(v0 * factor * w0);
        result.y = __float2bfloat16(v1 * factor * w1);
        vec_output[i] = result;
    }

    // Handle odd element
    if (hidden_size % 2 == 1 && tid == 0) {
        float v = __bfloat162float(row_input[hidden_size - 1]);
        float w = __bfloat162float(weight[hidden_size - 1]);
        row_output[hidden_size - 1] = __float2bfloat16(v * factor * w);
    }
}

// ============================================================================
// FP16-specific optimized kernel using __half2 for 2-element vectorization
// ============================================================================
__global__ void rmsnorm_kernel_fp16_vectorized(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    extern __shared__ char smem[];
    float* shared = reinterpret_cast<float*>(smem);

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const __half* row_input = input + row * hidden_size;
    __half* row_output = output + row * hidden_size;

    // Phase 1: Compute sum of squares with half2 vectorized loads
    float sum_sq = 0.0f;

    // Use __half2 for 2-element vectorized loads
    const int vec_hidden = hidden_size / 2;
    const __half2* vec_input = reinterpret_cast<const __half2*>(row_input);

    #pragma unroll 4
    for (int i = tid; i < vec_hidden; i += stride) {
        __half2 v = vec_input[i];
        float v0 = __half2float(v.x);
        float v1 = __half2float(v.y);
        sum_sq += v0 * v0 + v1 * v1;
    }

    // Handle odd element if hidden_size is odd
    if (hidden_size % 2 == 1 && tid == 0) {
        float v = __half2float(row_input[hidden_size - 1]);
        sum_sq += v * v;
    }

    // Reduce across block
    sum_sq = block_reduce_sum(sum_sq, shared);

    // Compute RMS inverse
    __shared__ float rms_inv;
    if (tid == 0) {
        float mean_sq = sum_sq / static_cast<float>(hidden_size);
        rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    const float factor = rms_inv;

    // Phase 2: Apply normalization and weight with half2 vectorized stores
    const __half2* vec_weight = reinterpret_cast<const __half2*>(weight);
    __half2* vec_output = reinterpret_cast<__half2*>(row_output);

    #pragma unroll 4
    for (int i = tid; i < vec_hidden; i += stride) {
        __half2 v_in = vec_input[i];
        __half2 v_w = vec_weight[i];

        float v0 = __half2float(v_in.x);
        float v1 = __half2float(v_in.y);
        float w0 = __half2float(v_w.x);
        float w1 = __half2float(v_w.y);

        __half2 result;
        result.x = __float2half(v0 * factor * w0);
        result.y = __float2half(v1 * factor * w1);
        vec_output[i] = result;
    }

    // Handle odd element
    if (hidden_size % 2 == 1 && tid == 0) {
        float v = __half2float(row_input[hidden_size - 1]);
        float w = __half2float(weight[hidden_size - 1]);
        row_output[hidden_size - 1] = __float2half(v * factor * w);
    }
}

// ============================================================================
// Fused RMSNorm + residual add kernel (vectorized BF16 version)
// ============================================================================
__global__ void rmsnorm_residual_kernel_bf16_vectorized(
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ residual_out,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    const int hidden_size,
    const float eps
) {
    extern __shared__ char smem[];
    float* shared = reinterpret_cast<float*>(smem);

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const __nv_bfloat16* row_input = input + row * hidden_size;
    const __nv_bfloat16* row_residual = residual + row * hidden_size;
    __nv_bfloat16* row_output = output + row * hidden_size;
    __nv_bfloat16* row_residual_out = residual_out + row * hidden_size;

    // Phase 1: Compute sum of squares of (input + residual)
    float sum_sq = 0.0f;

    const int vec_hidden = hidden_size / 2;
    const __nv_bfloat162* vec_input = reinterpret_cast<const __nv_bfloat162*>(row_input);
    const __nv_bfloat162* vec_residual = reinterpret_cast<const __nv_bfloat162*>(row_residual);

    #pragma unroll 4
    for (int i = tid; i < vec_hidden; i += stride) {
        __nv_bfloat162 v_in = vec_input[i];
        __nv_bfloat162 v_res = vec_residual[i];

        float combined0 = __bfloat162float(v_in.x) + __bfloat162float(v_res.x);
        float combined1 = __bfloat162float(v_in.y) + __bfloat162float(v_res.y);
        sum_sq += combined0 * combined0 + combined1 * combined1;
    }

    // Reduce across block
    sum_sq = block_reduce_sum(sum_sq, shared);

    // Compute RMS inverse
    __shared__ float rms_inv;
    if (tid == 0) {
        float mean_sq = sum_sq / static_cast<float>(hidden_size);
        rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    const float factor = rms_inv;

    // Phase 2: Apply normalization, weight, and store residual
    const __nv_bfloat162* vec_weight = reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* vec_output = reinterpret_cast<__nv_bfloat162*>(row_output);
    __nv_bfloat162* vec_residual_out = reinterpret_cast<__nv_bfloat162*>(row_residual_out);

    #pragma unroll 4
    for (int i = tid; i < vec_hidden; i += stride) {
        __nv_bfloat162 v_in = vec_input[i];
        __nv_bfloat162 v_res = vec_residual[i];
        __nv_bfloat162 v_w = vec_weight[i];

        float combined0 = __bfloat162float(v_in.x) + __bfloat162float(v_res.x);
        float combined1 = __bfloat162float(v_in.y) + __bfloat162float(v_res.y);
        float w0 = __bfloat162float(v_w.x);
        float w1 = __bfloat162float(v_w.y);

        __nv_bfloat162 res_out;
        res_out.x = __float2bfloat16(combined0);
        res_out.y = __float2bfloat16(combined1);
        vec_residual_out[i] = res_out;

        __nv_bfloat162 out;
        out.x = __float2bfloat16(combined0 * factor * w0);
        out.y = __float2bfloat16(combined1 * factor * w1);
        vec_output[i] = out;
    }
}

// ============================================================================
// Legacy kernel - kept for compatibility
// ============================================================================
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
        acc_t val = to_float(row_input[i]);
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
        acc_t val = to_float(row_input[i]);
        acc_t w = to_float(weight[i]);
        row_output[i] = from_float(val * rms_inv * w, (scalar_t*)nullptr);
    }
}

// Fused RMSNorm + residual add kernel (legacy)
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
        acc_t val = to_float(row_input[i]) + to_float(row_residual[i]);
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
        acc_t in_val = to_float(row_input[i]);
        acc_t res_val = to_float(row_residual[i]);
        acc_t combined = in_val + res_val;
        acc_t w = to_float(weight[i]);

        row_residual_out[i] = from_float(combined, (scalar_t*)nullptr);
        row_output[i] = from_float(combined * rms_inv * w, (scalar_t*)nullptr);
    }
}

// ============================================================================
// Launch functions - OPTIMIZED VERSIONS
// ============================================================================
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
    int threads = min(hidden_size / 2, MAX_THREADS);  // Divide by 2 for vectorized access
    threads = max(threads, WARP_SIZE);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;  // Round to warp boundary

    size_t smem_size = ((threads + WARP_SIZE - 1) / WARP_SIZE) * sizeof(float);

    // Use vectorized kernel for even hidden sizes >= 64
    if (hidden_size % 2 == 0 && hidden_size >= 64) {
        rmsnorm_kernel_fp16_vectorized<<<num_rows, threads, smem_size, stream>>>(
            output, input, weight, hidden_size, eps
        );
    } else {
        threads = min(hidden_size, MAX_THREADS);
        threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        rmsnorm_kernel<__half><<<num_rows, threads, smem_size, stream>>>(
            output, input, weight, hidden_size, eps
        );
    }
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
    int threads = min(hidden_size / 2, MAX_THREADS);  // Divide by 2 for vectorized access
    threads = max(threads, WARP_SIZE);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    size_t smem_size = ((threads + WARP_SIZE - 1) / WARP_SIZE) * sizeof(float);

    // Use vectorized kernel for even hidden sizes >= 64
    if (hidden_size % 2 == 0 && hidden_size >= 64) {
        rmsnorm_kernel_bf16_vectorized<<<num_rows, threads, smem_size, stream>>>(
            output, input, weight, hidden_size, eps
        );
    } else {
        threads = min(hidden_size, MAX_THREADS);
        threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        rmsnorm_kernel<__nv_bfloat16><<<num_rows, threads, smem_size, stream>>>(
            output, input, weight, hidden_size, eps
        );
    }
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
    int threads = min(hidden_size, MAX_THREADS);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    size_t smem_size = ((threads + WARP_SIZE - 1) / WARP_SIZE) * sizeof(float);

    // Use vectorized kernel for sizes divisible by 4
    if (hidden_size % 4 == 0 && hidden_size >= 64) {
        rmsnorm_kernel_vectorized<float><<<num_rows, threads, smem_size, stream>>>(
            output, input, weight, hidden_size, eps
        );
    } else {
        rmsnorm_kernel<float><<<num_rows, threads, smem_size, stream>>>(
            output, input, weight, hidden_size, eps
        );
    }
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
    int threads = min(hidden_size, MAX_THREADS);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    size_t smem_size = ((threads + WARP_SIZE - 1) / WARP_SIZE) * sizeof(float);

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
    int threads = min(hidden_size / 2, MAX_THREADS);
    threads = max(threads, WARP_SIZE);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    size_t smem_size = ((threads + WARP_SIZE - 1) / WARP_SIZE) * sizeof(float);

    // Use vectorized kernel for even hidden sizes
    if (hidden_size % 2 == 0 && hidden_size >= 64) {
        rmsnorm_residual_kernel_bf16_vectorized<<<num_rows, threads, smem_size, stream>>>(
            output, residual_out, input, residual, weight, hidden_size, eps
        );
    } else {
        threads = min(hidden_size, MAX_THREADS);
        threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        rmsnorm_residual_kernel<__nv_bfloat16><<<num_rows, threads, smem_size, stream>>>(
            output, residual_out, input, residual, weight, hidden_size, eps
        );
    }
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
    int threads = min(hidden_size, MAX_THREADS);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    size_t smem_size = ((threads + WARP_SIZE - 1) / WARP_SIZE) * sizeof(float);

    rmsnorm_residual_kernel<float><<<num_rows, threads, smem_size, stream>>>(
        output, residual_out, input, residual, weight, hidden_size, eps
    );
}

} // extern "C"
