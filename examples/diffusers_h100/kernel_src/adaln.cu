/*
 * Adaptive Layer Normalization (AdaLN) CUDA Kernel for LTX-Video
 * Optimized for NVIDIA H100 (sm_90)
 *
 * AdaLN formula: output = norm(x) * weight * (1 + scale) + shift
 * where scale and shift come from timestep/conditioning MLP.
 *
 * This is critical for DiT (Diffusion Transformer) conditioning.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

// Warp-level reduction
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction
template <typename T>
__device__ __forceinline__ T block_reduce_sum(T val, T* shared) {
    const int lane = threadIdx.x % 32;
    const int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    const int num_warps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : T(0);
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

// AdaLN with LayerNorm base
// Formula: output = ((x - mean) / sqrt(var + eps)) * weight * (1 + scale) + shift
template <typename scalar_t, typename acc_t = float>
__global__ void adaln_layernorm_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,      // Can be nullptr
    const scalar_t* __restrict__ scale,     // [batch, hidden_size] from timestep MLP
    const scalar_t* __restrict__ shift,     // [batch, hidden_size] from timestep MLP
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const float eps
) {
    extern __shared__ char smem[];
    acc_t* shared = reinterpret_cast<acc_t*>(smem);

    // Each block handles one row (one token)
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
        sum += static_cast<acc_t>(row_input[i]);
    }
    sum = block_reduce_sum(sum, shared);

    __shared__ acc_t mean;
    if (tid == 0) {
        mean = sum / static_cast<acc_t>(hidden_size);
    }
    __syncthreads();

    // Compute variance
    acc_t var_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        acc_t diff = static_cast<acc_t>(row_input[i]) - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum, shared);

    __shared__ acc_t inv_std;
    if (tid == 0) {
        acc_t variance = var_sum / static_cast<acc_t>(hidden_size);
        inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();

    // Apply normalization with adaptive scale and shift
    for (int i = tid; i < hidden_size; i += stride) {
        acc_t val = static_cast<acc_t>(row_input[i]);
        acc_t normalized = (val - mean) * inv_std;

        acc_t w = static_cast<acc_t>(weight[i]);
        acc_t s = static_cast<acc_t>(batch_scale[i]);
        acc_t sh = static_cast<acc_t>(batch_shift[i]);

        // AdaLN: norm(x) * weight * (1 + scale) + shift
        acc_t result = normalized * w * (1.0f + s) + sh;

        if (bias != nullptr) {
            result += static_cast<acc_t>(bias[i]);
        }

        row_output[i] = static_cast<scalar_t>(result);
    }
}

// AdaLN with RMSNorm base (ada_rmsnorm)
// Formula: output = (x / sqrt(mean(x²) + eps)) * weight * (1 + scale) + shift
template <typename scalar_t, typename acc_t = float>
__global__ void adaln_rmsnorm_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ scale,     // [batch, hidden_size]
    const scalar_t* __restrict__ shift,     // [batch, hidden_size]
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

    // Compute sum of squares
    acc_t sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        acc_t val = static_cast<acc_t>(row_input[i]);
        sum_sq += val * val;
    }
    sum_sq = block_reduce_sum(sum_sq, shared);

    __shared__ acc_t rms_inv;
    if (tid == 0) {
        acc_t mean_sq = sum_sq / static_cast<acc_t>(hidden_size);
        rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    // Apply RMS normalization with adaptive scale and shift
    for (int i = tid; i < hidden_size; i += stride) {
        acc_t val = static_cast<acc_t>(row_input[i]);
        acc_t normalized = val * rms_inv;

        acc_t w = static_cast<acc_t>(weight[i]);
        acc_t s = static_cast<acc_t>(batch_scale[i]);
        acc_t sh = static_cast<acc_t>(batch_shift[i]);

        acc_t result = normalized * w * (1.0f + s) + sh;

        row_output[i] = static_cast<scalar_t>(result);
    }
}

// AdaLN-Zero: variant used in some DiT implementations
// Formula: output = norm(x) * (1 + scale) + shift, then gated by alpha
template <typename scalar_t, typename acc_t = float>
__global__ void adaln_zero_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ scale,     // [batch, hidden_size]
    const scalar_t* __restrict__ shift,     // [batch, hidden_size]
    const scalar_t* __restrict__ gate,      // [batch, hidden_size] - zero-init gate
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

    // Compute sum of squares for RMSNorm
    acc_t sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        acc_t val = static_cast<acc_t>(row_input[i]);
        sum_sq += val * val;
    }
    sum_sq = block_reduce_sum(sum_sq, shared);

    __shared__ acc_t rms_inv;
    if (tid == 0) {
        acc_t mean_sq = sum_sq / static_cast<acc_t>(hidden_size);
        rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    // Apply AdaLN-Zero
    for (int i = tid; i < hidden_size; i += stride) {
        acc_t val = static_cast<acc_t>(row_input[i]);
        acc_t normalized = val * rms_inv;

        acc_t w = static_cast<acc_t>(weight[i]);
        acc_t s = static_cast<acc_t>(batch_scale[i]);
        acc_t sh = static_cast<acc_t>(batch_shift[i]);
        acc_t g = static_cast<acc_t>(batch_gate[i]);

        // AdaLN-Zero: (norm(x) * weight * (1 + scale) + shift) * gate
        acc_t result = (normalized * w * (1.0f + s) + sh) * g;

        row_output[i] = static_cast<scalar_t>(result);
    }
}

// Fused AdaLN + residual
template <typename scalar_t, typename acc_t = float>
__global__ void adaln_residual_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
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
    const scalar_t* row_residual = residual + row * hidden_size;
    scalar_t* row_output = output + row * hidden_size;
    const scalar_t* batch_scale = scale + batch_idx * hidden_size;
    const scalar_t* batch_shift = shift + batch_idx * hidden_size;

    // Compute sum of squares of (input + residual)
    acc_t sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        acc_t combined = static_cast<acc_t>(row_input[i]) + static_cast<acc_t>(row_residual[i]);
        sum_sq += combined * combined;
    }
    sum_sq = block_reduce_sum(sum_sq, shared);

    __shared__ acc_t rms_inv;
    if (tid == 0) {
        acc_t mean_sq = sum_sq / static_cast<acc_t>(hidden_size);
        rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    for (int i = tid; i < hidden_size; i += stride) {
        acc_t combined = static_cast<acc_t>(row_input[i]) + static_cast<acc_t>(row_residual[i]);
        acc_t normalized = combined * rms_inv;

        acc_t w = static_cast<acc_t>(weight[i]);
        acc_t s = static_cast<acc_t>(batch_scale[i]);
        acc_t sh = static_cast<acc_t>(batch_shift[i]);

        acc_t result = normalized * w * (1.0f + s) + sh;

        row_output[i] = static_cast<scalar_t>(result);
    }
}

// Launch functions
extern "C" {

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
    int threads = min(hidden_size, 1024);
    threads = ((threads + 31) / 32) * 32;

    size_t smem_size = ((threads + 31) / 32) * sizeof(float);

    adaln_layernorm_kernel<__half><<<num_rows, threads, smem_size, stream>>>(
        output, input, weight, bias, scale, shift,
        batch_size, seq_len, hidden_size, eps
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
    int threads = min(hidden_size, 1024);
    threads = ((threads + 31) / 32) * 32;

    size_t smem_size = ((threads + 31) / 32) * sizeof(float);

    adaln_layernorm_kernel<__nv_bfloat16><<<num_rows, threads, smem_size, stream>>>(
        output, input, weight, bias, scale, shift,
        batch_size, seq_len, hidden_size, eps
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
    int threads = min(hidden_size, 1024);
    threads = ((threads + 31) / 32) * 32;

    size_t smem_size = ((threads + 31) / 32) * sizeof(float);

    adaln_layernorm_kernel<float><<<num_rows, threads, smem_size, stream>>>(
        output, input, weight, bias, scale, shift,
        batch_size, seq_len, hidden_size, eps
    );
}

// AdaLN with RMSNorm (ada_rmsnorm)
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
    int threads = min(hidden_size, 1024);
    threads = ((threads + 31) / 32) * 32;

    size_t smem_size = ((threads + 31) / 32) * sizeof(float);

    adaln_rmsnorm_kernel<__half><<<num_rows, threads, smem_size, stream>>>(
        output, input, weight, scale, shift,
        batch_size, seq_len, hidden_size, eps
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
    int threads = min(hidden_size, 1024);
    threads = ((threads + 31) / 32) * 32;

    size_t smem_size = ((threads + 31) / 32) * sizeof(float);

    adaln_rmsnorm_kernel<__nv_bfloat16><<<num_rows, threads, smem_size, stream>>>(
        output, input, weight, scale, shift,
        batch_size, seq_len, hidden_size, eps
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
    int threads = min(hidden_size, 1024);
    threads = ((threads + 31) / 32) * 32;

    size_t smem_size = ((threads + 31) / 32) * sizeof(float);

    adaln_rmsnorm_kernel<float><<<num_rows, threads, smem_size, stream>>>(
        output, input, weight, scale, shift,
        batch_size, seq_len, hidden_size, eps
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
    int threads = min(hidden_size, 1024);
    threads = ((threads + 31) / 32) * 32;

    size_t smem_size = ((threads + 31) / 32) * sizeof(float);

    adaln_zero_kernel<__half><<<num_rows, threads, smem_size, stream>>>(
        output, input, weight, scale, shift, gate,
        batch_size, seq_len, hidden_size, eps
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
    int threads = min(hidden_size, 1024);
    threads = ((threads + 31) / 32) * 32;

    size_t smem_size = ((threads + 31) / 32) * sizeof(float);

    adaln_zero_kernel<__nv_bfloat16><<<num_rows, threads, smem_size, stream>>>(
        output, input, weight, scale, shift, gate,
        batch_size, seq_len, hidden_size, eps
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
    int threads = min(hidden_size, 1024);
    threads = ((threads + 31) / 32) * 32;

    size_t smem_size = ((threads + 31) / 32) * sizeof(float);

    adaln_zero_kernel<float><<<num_rows, threads, smem_size, stream>>>(
        output, input, weight, scale, shift, gate,
        batch_size, seq_len, hidden_size, eps
    );
}

// AdaLN with residual (fused)
void adaln_residual_forward_fp16(
    __half* output,
    const __half* input,
    const __half* residual,
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
    int threads = min(hidden_size, 1024);
    threads = ((threads + 31) / 32) * 32;

    size_t smem_size = ((threads + 31) / 32) * sizeof(float);

    adaln_residual_kernel<__half><<<num_rows, threads, smem_size, stream>>>(
        output, input, residual, weight, scale, shift,
        batch_size, seq_len, hidden_size, eps
    );
}

void adaln_residual_forward_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* residual,
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
    int threads = min(hidden_size, 1024);
    threads = ((threads + 31) / 32) * 32;

    size_t smem_size = ((threads + 31) / 32) * sizeof(float);

    adaln_residual_kernel<__nv_bfloat16><<<num_rows, threads, smem_size, stream>>>(
        output, input, residual, weight, scale, shift,
        batch_size, seq_len, hidden_size, eps
    );
}

void adaln_residual_forward_fp32(
    float* output,
    const float* input,
    const float* residual,
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
    int threads = min(hidden_size, 1024);
    threads = ((threads + 31) / 32) * 32;

    size_t smem_size = ((threads + 31) / 32) * sizeof(float);

    adaln_residual_kernel<float><<<num_rows, threads, smem_size, stream>>>(
        output, input, residual, weight, scale, shift,
        batch_size, seq_len, hidden_size, eps
    );
}

} // extern "C"
