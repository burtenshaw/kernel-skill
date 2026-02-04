/*
 * Rotary Position Embedding (RoPE) CUDA Kernel for LTX-Video
 * Optimized for NVIDIA H100 (sm_90)
 *
 * Supports both 1D (text sequences) and 3D (video: time × height × width) RoPE.
 * 3D RoPE splits head_dim into three parts for temporal and spatial dimensions.
 *
 * H100 Optimizations:
 * - Element-wise parallelism with BLOCK_SIZE=256
 * - Coalesced memory access
 * - Fast trigonometric computations
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

constexpr int ROPE_BLOCK_SIZE = 256;

// Helper functions for type conversion
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float from_float_rope(float x, float*) { return x; }
__device__ __forceinline__ __half from_float_rope(float x, __half*) { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat16 from_float_rope(float x, __nv_bfloat16*) { return __float2bfloat16(x); }

// Compute rotary embedding frequencies
__device__ __forceinline__ float compute_freq(int dim_idx, int head_dim, float theta_base) {
    float freq_exp = -2.0f * static_cast<float>(dim_idx) / static_cast<float>(head_dim);
    return powf(theta_base, freq_exp);
}

// Apply rotation to a pair of values
template <typename scalar_t, typename acc_t = float>
__device__ __forceinline__ void apply_rotation(
    scalar_t& x0, scalar_t& x1,
    acc_t cos_val, acc_t sin_val
) {
    acc_t v0 = to_float(x0);
    acc_t v1 = to_float(x1);
    x0 = from_float_rope(v0 * cos_val - v1 * sin_val, (scalar_t*)nullptr);
    x1 = from_float_rope(v0 * sin_val + v1 * cos_val, (scalar_t*)nullptr);
}

// 1D RoPE kernel for text sequences
// Input shape: [batch, seq_len, num_heads, head_dim]
template <typename scalar_t>
__global__ void rope_1d_kernel(
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const float theta_base
) {
    const int total_elements = batch_size * seq_len * num_heads * (head_dim / 2);
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    // Decode indices
    const int half_head_dim = head_dim / 2;
    const int pair_idx = idx % half_head_dim;
    const int head_idx = (idx / half_head_dim) % num_heads;
    const int seq_idx = (idx / (half_head_dim * num_heads)) % seq_len;
    const int batch_idx = idx / (half_head_dim * num_heads * seq_len);

    // Compute position in flattened tensor
    const int base_offset = batch_idx * (seq_len * num_heads * head_dim)
                          + seq_idx * (num_heads * head_dim)
                          + head_idx * head_dim;
    const int idx0 = base_offset + pair_idx * 2;
    const int idx1 = idx0 + 1;

    // Compute frequency and angle
    float freq = compute_freq(pair_idx, half_head_dim, theta_base);
    float angle = static_cast<float>(seq_idx) * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // Apply rotation to query
    apply_rotation(query[idx0], query[idx1], cos_val, sin_val);

    // Apply rotation to key
    apply_rotation(key[idx0], key[idx1], cos_val, sin_val);
}

// 3D RoPE kernel for video (time × height × width)
// Input shape: [batch, num_frames * height * width, num_heads, head_dim]
// Head dim split: [rope_dim_t, rope_dim_h, rope_dim_w]
template <typename scalar_t>
__global__ void rope_3d_kernel(
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const int batch_size,
    const int num_frames,
    const int height,
    const int width,
    const int num_heads,
    const int head_dim,
    const int rope_dim_t,
    const int rope_dim_h,
    const int rope_dim_w,
    const float theta_base_t,
    const float theta_base_h,
    const float theta_base_w
) {
    const int seq_len = num_frames * height * width;
    const int total_pairs = batch_size * seq_len * num_heads * (head_dim / 2);
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_pairs) return;

    // Decode indices
    const int half_head_dim = head_dim / 2;
    const int pair_idx = idx % half_head_dim;
    const int head_idx = (idx / half_head_dim) % num_heads;
    const int flat_seq_idx = (idx / (half_head_dim * num_heads)) % seq_len;
    const int batch_idx = idx / (half_head_dim * num_heads * seq_len);

    // Decode 3D position from flat sequence index
    const int hw = height * width;
    const int t_idx = flat_seq_idx / hw;
    const int hw_idx = flat_seq_idx % hw;
    const int h_idx = hw_idx / width;
    const int w_idx = hw_idx % width;

    // Determine which dimension this pair belongs to
    const int dim_offset = pair_idx * 2;
    float angle = 0.0f;
    float theta_base;
    int local_pair_idx;
    int rope_dim;

    if (dim_offset < rope_dim_t) {
        // Temporal dimension
        local_pair_idx = pair_idx;
        rope_dim = rope_dim_t / 2;
        theta_base = theta_base_t;
        float freq = compute_freq(local_pair_idx, rope_dim, theta_base);
        angle = static_cast<float>(t_idx) * freq;
    } else if (dim_offset < rope_dim_t + rope_dim_h) {
        // Height dimension
        local_pair_idx = pair_idx - (rope_dim_t / 2);
        rope_dim = rope_dim_h / 2;
        theta_base = theta_base_h;
        float freq = compute_freq(local_pair_idx, rope_dim, theta_base);
        angle = static_cast<float>(h_idx) * freq;
    } else {
        // Width dimension
        local_pair_idx = pair_idx - (rope_dim_t / 2) - (rope_dim_h / 2);
        rope_dim = rope_dim_w / 2;
        theta_base = theta_base_w;
        float freq = compute_freq(local_pair_idx, rope_dim, theta_base);
        angle = static_cast<float>(w_idx) * freq;
    }

    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // Compute position in flattened tensor
    const int base_offset = batch_idx * (seq_len * num_heads * head_dim)
                          + flat_seq_idx * (num_heads * head_dim)
                          + head_idx * head_dim;
    const int idx0 = base_offset + pair_idx * 2;
    const int idx1 = idx0 + 1;

    // Apply rotation to query
    apply_rotation(query[idx0], query[idx1], cos_val, sin_val);

    // Apply rotation to key
    apply_rotation(key[idx0], key[idx1], cos_val, sin_val);
}

// Launch functions
extern "C" {

// 1D RoPE
void rope_1d_forward_fp16(
    __half* query,
    __half* key,
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const float theta_base,
    cudaStream_t stream
) {
    const int total_pairs = batch_size * seq_len * num_heads * (head_dim / 2);
    const int num_blocks = (total_pairs + ROPE_BLOCK_SIZE - 1) / ROPE_BLOCK_SIZE;

    rope_1d_kernel<__half><<<num_blocks, ROPE_BLOCK_SIZE, 0, stream>>>(
        query, key, batch_size, seq_len, num_heads, head_dim, theta_base
    );
}

void rope_1d_forward_bf16(
    __nv_bfloat16* query,
    __nv_bfloat16* key,
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const float theta_base,
    cudaStream_t stream
) {
    const int total_pairs = batch_size * seq_len * num_heads * (head_dim / 2);
    const int num_blocks = (total_pairs + ROPE_BLOCK_SIZE - 1) / ROPE_BLOCK_SIZE;

    rope_1d_kernel<__nv_bfloat16><<<num_blocks, ROPE_BLOCK_SIZE, 0, stream>>>(
        query, key, batch_size, seq_len, num_heads, head_dim, theta_base
    );
}

void rope_1d_forward_fp32(
    float* query,
    float* key,
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const float theta_base,
    cudaStream_t stream
) {
    const int total_pairs = batch_size * seq_len * num_heads * (head_dim / 2);
    const int num_blocks = (total_pairs + ROPE_BLOCK_SIZE - 1) / ROPE_BLOCK_SIZE;

    rope_1d_kernel<float><<<num_blocks, ROPE_BLOCK_SIZE, 0, stream>>>(
        query, key, batch_size, seq_len, num_heads, head_dim, theta_base
    );
}

// 3D RoPE for video
void rope_3d_forward_fp16(
    __half* query,
    __half* key,
    const int batch_size,
    const int num_frames,
    const int height,
    const int width,
    const int num_heads,
    const int head_dim,
    const float theta_base,
    cudaStream_t stream
) {
    // Default split: equal thirds
    const int rope_dim_t = head_dim / 3;
    const int rope_dim_h = head_dim / 3;
    const int rope_dim_w = head_dim - rope_dim_t - rope_dim_h;

    const int seq_len = num_frames * height * width;
    const int total_pairs = batch_size * seq_len * num_heads * (head_dim / 2);
    const int num_blocks = (total_pairs + ROPE_BLOCK_SIZE - 1) / ROPE_BLOCK_SIZE;

    rope_3d_kernel<__half><<<num_blocks, ROPE_BLOCK_SIZE, 0, stream>>>(
        query, key, batch_size, num_frames, height, width, num_heads, head_dim,
        rope_dim_t, rope_dim_h, rope_dim_w, theta_base, theta_base, theta_base
    );
}

void rope_3d_forward_bf16(
    __nv_bfloat16* query,
    __nv_bfloat16* key,
    const int batch_size,
    const int num_frames,
    const int height,
    const int width,
    const int num_heads,
    const int head_dim,
    const float theta_base,
    cudaStream_t stream
) {
    const int rope_dim_t = head_dim / 3;
    const int rope_dim_h = head_dim / 3;
    const int rope_dim_w = head_dim - rope_dim_t - rope_dim_h;

    const int seq_len = num_frames * height * width;
    const int total_pairs = batch_size * seq_len * num_heads * (head_dim / 2);
    const int num_blocks = (total_pairs + ROPE_BLOCK_SIZE - 1) / ROPE_BLOCK_SIZE;

    rope_3d_kernel<__nv_bfloat16><<<num_blocks, ROPE_BLOCK_SIZE, 0, stream>>>(
        query, key, batch_size, num_frames, height, width, num_heads, head_dim,
        rope_dim_t, rope_dim_h, rope_dim_w, theta_base, theta_base, theta_base
    );
}

void rope_3d_forward_fp32(
    float* query,
    float* key,
    const int batch_size,
    const int num_frames,
    const int height,
    const int width,
    const int num_heads,
    const int head_dim,
    const float theta_base,
    cudaStream_t stream
) {
    const int rope_dim_t = head_dim / 3;
    const int rope_dim_h = head_dim / 3;
    const int rope_dim_w = head_dim - rope_dim_t - rope_dim_h;

    const int seq_len = num_frames * height * width;
    const int total_pairs = batch_size * seq_len * num_heads * (head_dim / 2);
    const int num_blocks = (total_pairs + ROPE_BLOCK_SIZE - 1) / ROPE_BLOCK_SIZE;

    rope_3d_kernel<float><<<num_blocks, ROPE_BLOCK_SIZE, 0, stream>>>(
        query, key, batch_size, num_frames, height, width, num_heads, head_dim,
        rope_dim_t, rope_dim_h, rope_dim_w, theta_base, theta_base, theta_base
    );
}

// Extended 3D RoPE with custom dimension splits and theta bases
void rope_3d_extended_forward_fp16(
    __half* query,
    __half* key,
    const int batch_size,
    const int num_frames,
    const int height,
    const int width,
    const int num_heads,
    const int head_dim,
    const int rope_dim_t,
    const int rope_dim_h,
    const int rope_dim_w,
    const float theta_base_t,
    const float theta_base_h,
    const float theta_base_w,
    cudaStream_t stream
) {
    const int seq_len = num_frames * height * width;
    const int total_pairs = batch_size * seq_len * num_heads * (head_dim / 2);
    const int num_blocks = (total_pairs + ROPE_BLOCK_SIZE - 1) / ROPE_BLOCK_SIZE;

    rope_3d_kernel<__half><<<num_blocks, ROPE_BLOCK_SIZE, 0, stream>>>(
        query, key, batch_size, num_frames, height, width, num_heads, head_dim,
        rope_dim_t, rope_dim_h, rope_dim_w, theta_base_t, theta_base_h, theta_base_w
    );
}

void rope_3d_extended_forward_bf16(
    __nv_bfloat16* query,
    __nv_bfloat16* key,
    const int batch_size,
    const int num_frames,
    const int height,
    const int width,
    const int num_heads,
    const int head_dim,
    const int rope_dim_t,
    const int rope_dim_h,
    const int rope_dim_w,
    const float theta_base_t,
    const float theta_base_h,
    const float theta_base_w,
    cudaStream_t stream
) {
    const int seq_len = num_frames * height * width;
    const int total_pairs = batch_size * seq_len * num_heads * (head_dim / 2);
    const int num_blocks = (total_pairs + ROPE_BLOCK_SIZE - 1) / ROPE_BLOCK_SIZE;

    rope_3d_kernel<__nv_bfloat16><<<num_blocks, ROPE_BLOCK_SIZE, 0, stream>>>(
        query, key, batch_size, num_frames, height, width, num_heads, head_dim,
        rope_dim_t, rope_dim_h, rope_dim_w, theta_base_t, theta_base_h, theta_base_w
    );
}

void rope_3d_extended_forward_fp32(
    float* query,
    float* key,
    const int batch_size,
    const int num_frames,
    const int height,
    const int width,
    const int num_heads,
    const int head_dim,
    const int rope_dim_t,
    const int rope_dim_h,
    const int rope_dim_w,
    const float theta_base_t,
    const float theta_base_h,
    const float theta_base_w,
    cudaStream_t stream
) {
    const int seq_len = num_frames * height * width;
    const int total_pairs = batch_size * seq_len * num_heads * (head_dim / 2);
    const int num_blocks = (total_pairs + ROPE_BLOCK_SIZE - 1) / ROPE_BLOCK_SIZE;

    rope_3d_kernel<float><<<num_blocks, ROPE_BLOCK_SIZE, 0, stream>>>(
        query, key, batch_size, num_frames, height, width, num_heads, head_dim,
        rope_dim_t, rope_dim_h, rope_dim_w, theta_base_t, theta_base_h, theta_base_w
    );
}

} // extern "C"
