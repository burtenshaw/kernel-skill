/*
 * Adaptive Layer Normalization (AdaLN) CUDA Kernel for LTX-Video
 * Optimized for NVIDIA H100 (sm_90)
 *
 * AdaLN is the key conditioning mechanism for Diffusion Transformers (DiT).
 * Formula: output = norm(x) * weight * (1 + scale) + shift
 * where scale and shift come from timestep/conditioning MLP.
 *
 * Variants:
 * - adaln_rmsnorm: Uses RMSNorm as base (faster, used in LTX-Video)
 * - adaln_layernorm: Uses LayerNorm as base
 * - adaln_zero: AdaLN with zero-initialized gating
 *
 * H100 Optimizations:
 * - Fused normalization + conditioning in single kernel
 * - Warp shuffle reductions
 * - Coalesced memory access
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS = 1024;

// Helper functions for type conversion
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float from_float_adaln(float x, float*) { return x; }
__device__ __forceinline__ __half from_float_adaln(float x, __half*) { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat16 from_float_adaln(float x, __nv_bfloat16*) { return __float2bfloat16(x); }

// Warp-level reduction
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction
template <typename T>
__device__ __forceinline__ T block_reduce_sum(T val, T* shared) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : T(0);
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

// AdaLN with RMSNorm base (used in LTX-Video)
// Formula: rms_norm(x) * weight * (1 + scale) + shift
template <typename scalar_t, typename acc_t = float>
__global__ void adaln_rmsnorm_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ scale,   // [batch, hidden]
    const scalar_t* __restrict__ shift,   // [batch, hidden]
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const float eps
) {
    extern __shared__ char smem[];
    acc_t* shared = reinterpret_cast<acc_t*>(smem);

    // Each block processes one row
    const int row = blockIdx.x;
    const int batch_idx = row / seq_len;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const scalar_t* row_input = input + row * hidden_size;
    scalar_t* row_output = output + row * hidden_size;

    // Scale and shift are per-batch (broadcast across sequence)
    const scalar_t* batch_scale = scale + batch_idx * hidden_size;
    const scalar_t* batch_shift = shift + batch_idx * hidden_size;

    // Compute sum of squares for RMSNorm
    acc_t sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        acc_t val = to_float(row_input[i]);
        sum_sq += val * val;
    }

    sum_sq = block_reduce_sum(sum_sq, shared);

    // Compute RMS inverse
    __shared__ acc_t rms_inv;
    if (tid == 0) {
        acc_t mean_sq = sum_sq / static_cast<acc_t>(hidden_size);
        rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    // Apply: norm(x) * weight * (1 + scale) + shift
    for (int i = tid; i < hidden_size; i += stride) {
        acc_t x = to_float(row_input[i]);
        acc_t w = to_float(weight[i]);
        acc_t s = to_float(batch_scale[i]);
        acc_t sh = to_float(batch_shift[i]);

        acc_t normalized = x * rms_inv;
        acc_t result = normalized * w * (1.0f + s) + sh;

        row_output[i] = from_float_adaln(result, (scalar_t*)nullptr);
    }
}

// AdaLN with LayerNorm base
// Formula: layer_norm(x) * weight * (1 + scale) + shift
template <typename scalar_t, typename acc_t = float>
__global__ void adaln_layernorm_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,    // Optional (can be nullptr)
    const scalar_t* __restrict__ scale,
    const scalar_t* __restrict__ shift,
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const float eps
) {
    extern __shared__ char smem[];
    acc_t* shared = reinterpret_cast<acc_t*>(smem);

    const int row = blockIdx.x;
    const int batch_idx = row / seq_len;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const scalar_t* row_input = input + row * hidden_size;
    scalar_t* row_output = output + row * hidden_size;
    const scalar_t* batch_scale = scale + batch_idx * hidden_size;
    const scalar_t* batch_shift = shift + batch_idx * hidden_size;

    // Compute mean
    acc_t sum = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        sum += to_float(row_input[i]);
    }
    sum = block_reduce_sum(sum, shared);

    __shared__ acc_t mean_val;
    if (tid == 0) {
        mean_val = sum / static_cast<acc_t>(hidden_size);
    }
    __syncthreads();
    acc_t mean = mean_val;

    // Compute variance
    acc_t var_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        acc_t diff = to_float(row_input[i]) - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum, shared);

    __shared__ acc_t inv_std;
    if (tid == 0) {
        acc_t variance = var_sum / static_cast<acc_t>(hidden_size);
        inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();

    // Apply: (x - mean) * inv_std * weight + bias, then adaptive modulation
    for (int i = tid; i < hidden_size; i += stride) {
        acc_t x = to_float(row_input[i]);
        acc_t w = to_float(weight[i]);
        acc_t b = bias ? to_float(bias[i]) : 0.0f;
        acc_t s = to_float(batch_scale[i]);
        acc_t sh = to_float(batch_shift[i]);

        acc_t normalized = (x - mean) * inv_std * w + b;
        acc_t result = normalized * (1.0f + s) + sh;

        row_output[i] = from_float_adaln(result, (scalar_t*)nullptr);
    }
}

// AdaLN-Zero: with zero-initialized gating
// Formula: (norm(x) * weight * (1 + scale) + shift) * gate
template <typename scalar_t, typename acc_t = float>
__global__ void adaln_zero_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ scale,
    const scalar_t* __restrict__ shift,
    const scalar_t* __restrict__ gate,    // [batch, hidden]
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const float eps
) {
    extern __shared__ char smem[];
    acc_t* shared = reinterpret_cast<acc_t*>(smem);

    const int row = blockIdx.x;
    const int batch_idx = row / seq_len;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const scalar_t* row_input = input + row * hidden_size;
    scalar_t* row_output = output + row * hidden_size;
    const scalar_t* batch_scale = scale + batch_idx * hidden_size;
    const scalar_t* batch_shift = shift + batch_idx * hidden_size;
    const scalar_t* batch_gate = gate + batch_idx * hidden_size;

    // RMSNorm
    acc_t sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        acc_t val = to_float(row_input[i]);
        sum_sq += val * val;
    }
    sum_sq = block_reduce_sum(sum_sq, shared);

    __shared__ acc_t rms_inv;
    if (tid == 0) {
        acc_t mean_sq = sum_sq / static_cast<acc_t>(hidden_size);
        rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    // Apply: (norm(x) * weight * (1 + scale) + shift) * gate
    for (int i = tid; i < hidden_size; i += stride) {
        acc_t x = to_float(row_input[i]);
        acc_t w = to_float(weight[i]);
        acc_t s = to_float(batch_scale[i]);
        acc_t sh = to_float(batch_shift[i]);
        acc_t g = to_float(batch_gate[i]);

        acc_t normalized = x * rms_inv;
        acc_t modulated = normalized * w * (1.0f + s) + sh;
        acc_t result = modulated * g;

        row_output[i] = from_float_adaln(result, (scalar_t*)nullptr);
    }
}

// Launch functions
extern "C" {

// AdaLN with RMSNorm
void adaln_rmsnorm_forward_fp16(
    __half* output,
    const __half* input,
    const __half* weight,
    const __half* scale,
    const __half* shift,
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

    adaln_rmsnorm_kernel<__half><<<num_rows, threads, smem_size, stream>>>(
        output, input, weight, scale, shift, batch_size, seq_len, hidden_size, eps
    );
}

void adaln_rmsnorm_forward_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* scale,
    const __nv_bfloat16* shift,
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

    adaln_rmsnorm_kernel<__nv_bfloat16><<<num_rows, threads, smem_size, stream>>>(
        output, input, weight, scale, shift, batch_size, seq_len, hidden_size, eps
    );
}

void adaln_rmsnorm_forward_fp32(
    float* output,
    const float* input,
    const float* weight,
    const float* scale,
    const float* shift,
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

    adaln_rmsnorm_kernel<float><<<num_rows, threads, smem_size, stream>>>(
        output, input, weight, scale, shift, batch_size, seq_len, hidden_size, eps
    );
}

// AdaLN with LayerNorm
void adaln_layernorm_forward_fp16(
    __half* output,
    const __half* input,
    const __half* weight,
    const __half* bias,
    const __half* scale,
    const __half* shift,
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

    adaln_layernorm_kernel<__half><<<num_rows, threads, smem_size, stream>>>(
        output, input, weight, bias, scale, shift, batch_size, seq_len, hidden_size, eps
    );
}

void adaln_layernorm_forward_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    const __nv_bfloat16* scale,
    const __nv_bfloat16* shift,
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

    adaln_layernorm_kernel<__nv_bfloat16><<<num_rows, threads, smem_size, stream>>>(
        output, input, weight, bias, scale, shift, batch_size, seq_len, hidden_size, eps
    );
}

void adaln_layernorm_forward_fp32(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* scale,
    const float* shift,
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

    adaln_layernorm_kernel<float><<<num_rows, threads, smem_size, stream>>>(
        output, input, weight, bias, scale, shift, batch_size, seq_len, hidden_size, eps
    );
}

// AdaLN-Zero
void adaln_zero_forward_fp16(
    __half* output,
    const __half* input,
    const __half* weight,
    const __half* scale,
    const __half* shift,
    const __half* gate,
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

    adaln_zero_kernel<__half><<<num_rows, threads, smem_size, stream>>>(
        output, input, weight, scale, shift, gate, batch_size, seq_len, hidden_size, eps
    );
}

void adaln_zero_forward_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* scale,
    const __nv_bfloat16* shift,
    const __nv_bfloat16* gate,
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

    adaln_zero_kernel<__nv_bfloat16><<<num_rows, threads, smem_size, stream>>>(
        output, input, weight, scale, shift, gate, batch_size, seq_len, hidden_size, eps
    );
}

void adaln_zero_forward_fp32(
    float* output,
    const float* input,
    const float* weight,
    const float* scale,
    const float* shift,
    const float* gate,
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

    adaln_zero_kernel<float><<<num_rows, threads, smem_size, stream>>>(
        output, input, weight, scale, shift, gate, batch_size, seq_len, hidden_size, eps
    );
}

} // extern "C"
