/*
 * GEGLU (Gated Linear Unit with GELU) CUDA Kernel for LTX-Video
 * Optimized for NVIDIA H100 (sm_90)
 *
 * Formula: output = GELU(gate) * value
 * where input = [gate, value] split along last dimension
 *
 * LTX-Video uses the tanh approximation of GELU (gelu-approximate).
 *
 * H100 Optimizations:
 * - High occupancy element-wise parallelism
 * - Fast GELU tanh approximation
 * - Coalesced memory access patterns
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

constexpr int GEGLU_BLOCK_SIZE = 256;

// Helper functions for type conversion
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float from_float_geglu(float x, float*) { return x; }
__device__ __forceinline__ __half from_float_geglu(float x, __half*) { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat16 from_float_geglu(float x, __nv_bfloat16*) { return __float2bfloat16(x); }

// GELU constants for tanh approximation
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
constexpr float GELU_COEF_A = 0.7978845608028654f;  // sqrt(2/π)
constexpr float GELU_COEF_B = 0.044715f;

// Fast GELU with tanh approximation
__device__ __forceinline__ float gelu_tanh_approx(float x) {
    float x3 = x * x * x;
    float inner = GELU_COEF_A * (x + GELU_COEF_B * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Exact GELU using error function
__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));  // 1/sqrt(2)
}

// SiLU (Swish) activation: x * sigmoid(x)
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// GEGLU kernel with tanh approximation (default for LTX-Video)
template <typename scalar_t>
__global__ void geglu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int batch_size,
    const int seq_len,
    const int hidden_dim  // Output hidden dim (input has 2x this)
) {
    const int total_elements = batch_size * seq_len * hidden_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    // Decode position
    const int h_idx = idx % hidden_dim;
    const int seq_idx = (idx / hidden_dim) % seq_len;
    const int batch_idx = idx / (hidden_dim * seq_len);

    // Input layout: [..., 2*hidden_dim]
    const int input_offset = batch_idx * (seq_len * 2 * hidden_dim)
                           + seq_idx * (2 * hidden_dim);

    // Gate is first half, value is second half
    float gate = to_float(input[input_offset + h_idx]);
    float value = to_float(input[input_offset + hidden_dim + h_idx]);

    // GEGLU: GELU(gate) * value
    float result = gelu_tanh_approx(gate) * value;

    output[idx] = from_float_geglu(result, (scalar_t*)nullptr);
}

// GEGLU kernel with exact GELU
template <typename scalar_t>
__global__ void geglu_exact_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int batch_size,
    const int seq_len,
    const int hidden_dim
) {
    const int total_elements = batch_size * seq_len * hidden_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    const int h_idx = idx % hidden_dim;
    const int seq_idx = (idx / hidden_dim) % seq_len;
    const int batch_idx = idx / (hidden_dim * seq_len);

    const int input_offset = batch_idx * (seq_len * 2 * hidden_dim)
                           + seq_idx * (2 * hidden_dim);

    float gate = to_float(input[input_offset + h_idx]);
    float value = to_float(input[input_offset + hidden_dim + h_idx]);

    float result = gelu_exact(gate) * value;

    output[idx] = from_float_geglu(result, (scalar_t*)nullptr);
}

// SwiGLU kernel (alternative activation)
template <typename scalar_t>
__global__ void swiglu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int batch_size,
    const int seq_len,
    const int hidden_dim
) {
    const int total_elements = batch_size * seq_len * hidden_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    const int h_idx = idx % hidden_dim;
    const int seq_idx = (idx / hidden_dim) % seq_len;
    const int batch_idx = idx / (hidden_dim * seq_len);

    const int input_offset = batch_idx * (seq_len * 2 * hidden_dim)
                           + seq_idx * (2 * hidden_dim);

    float gate = to_float(input[input_offset + h_idx]);
    float value = to_float(input[input_offset + hidden_dim + h_idx]);

    // SwiGLU: SiLU(gate) * value
    float result = silu(gate) * value;

    output[idx] = from_float_geglu(result, (scalar_t*)nullptr);
}

// Launch functions
extern "C" {

// GEGLU with tanh approximation (default)
void geglu_forward_fp16(
    __half* output,
    const __half* input,
    const int batch_size,
    const int seq_len,
    const int hidden_dim,
    cudaStream_t stream
) {
    const int total_elements = batch_size * seq_len * hidden_dim;
    const int num_blocks = (total_elements + GEGLU_BLOCK_SIZE - 1) / GEGLU_BLOCK_SIZE;

    geglu_kernel<__half><<<num_blocks, GEGLU_BLOCK_SIZE, 0, stream>>>(
        output, input, batch_size, seq_len, hidden_dim
    );
}

void geglu_forward_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const int batch_size,
    const int seq_len,
    const int hidden_dim,
    cudaStream_t stream
) {
    const int total_elements = batch_size * seq_len * hidden_dim;
    const int num_blocks = (total_elements + GEGLU_BLOCK_SIZE - 1) / GEGLU_BLOCK_SIZE;

    geglu_kernel<__nv_bfloat16><<<num_blocks, GEGLU_BLOCK_SIZE, 0, stream>>>(
        output, input, batch_size, seq_len, hidden_dim
    );
}

void geglu_forward_fp32(
    float* output,
    const float* input,
    const int batch_size,
    const int seq_len,
    const int hidden_dim,
    cudaStream_t stream
) {
    const int total_elements = batch_size * seq_len * hidden_dim;
    const int num_blocks = (total_elements + GEGLU_BLOCK_SIZE - 1) / GEGLU_BLOCK_SIZE;

    geglu_kernel<float><<<num_blocks, GEGLU_BLOCK_SIZE, 0, stream>>>(
        output, input, batch_size, seq_len, hidden_dim
    );
}

// GEGLU with exact GELU
void geglu_exact_forward_fp16(
    __half* output,
    const __half* input,
    const int batch_size,
    const int seq_len,
    const int hidden_dim,
    cudaStream_t stream
) {
    const int total_elements = batch_size * seq_len * hidden_dim;
    const int num_blocks = (total_elements + GEGLU_BLOCK_SIZE - 1) / GEGLU_BLOCK_SIZE;

    geglu_exact_kernel<__half><<<num_blocks, GEGLU_BLOCK_SIZE, 0, stream>>>(
        output, input, batch_size, seq_len, hidden_dim
    );
}

void geglu_exact_forward_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const int batch_size,
    const int seq_len,
    const int hidden_dim,
    cudaStream_t stream
) {
    const int total_elements = batch_size * seq_len * hidden_dim;
    const int num_blocks = (total_elements + GEGLU_BLOCK_SIZE - 1) / GEGLU_BLOCK_SIZE;

    geglu_exact_kernel<__nv_bfloat16><<<num_blocks, GEGLU_BLOCK_SIZE, 0, stream>>>(
        output, input, batch_size, seq_len, hidden_dim
    );
}

void geglu_exact_forward_fp32(
    float* output,
    const float* input,
    const int batch_size,
    const int seq_len,
    const int hidden_dim,
    cudaStream_t stream
) {
    const int total_elements = batch_size * seq_len * hidden_dim;
    const int num_blocks = (total_elements + GEGLU_BLOCK_SIZE - 1) / GEGLU_BLOCK_SIZE;

    geglu_exact_kernel<float><<<num_blocks, GEGLU_BLOCK_SIZE, 0, stream>>>(
        output, input, batch_size, seq_len, hidden_dim
    );
}

// SwiGLU
void swiglu_forward_fp16(
    __half* output,
    const __half* input,
    const int batch_size,
    const int seq_len,
    const int hidden_dim,
    cudaStream_t stream
) {
    const int total_elements = batch_size * seq_len * hidden_dim;
    const int num_blocks = (total_elements + GEGLU_BLOCK_SIZE - 1) / GEGLU_BLOCK_SIZE;

    swiglu_kernel<__half><<<num_blocks, GEGLU_BLOCK_SIZE, 0, stream>>>(
        output, input, batch_size, seq_len, hidden_dim
    );
}

void swiglu_forward_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const int batch_size,
    const int seq_len,
    const int hidden_dim,
    cudaStream_t stream
) {
    const int total_elements = batch_size * seq_len * hidden_dim;
    const int num_blocks = (total_elements + GEGLU_BLOCK_SIZE - 1) / GEGLU_BLOCK_SIZE;

    swiglu_kernel<__nv_bfloat16><<<num_blocks, GEGLU_BLOCK_SIZE, 0, stream>>>(
        output, input, batch_size, seq_len, hidden_dim
    );
}

void swiglu_forward_fp32(
    float* output,
    const float* input,
    const int batch_size,
    const int seq_len,
    const int hidden_dim,
    cudaStream_t stream
) {
    const int total_elements = batch_size * seq_len * hidden_dim;
    const int num_blocks = (total_elements + GEGLU_BLOCK_SIZE - 1) / GEGLU_BLOCK_SIZE;

    swiglu_kernel<float><<<num_blocks, GEGLU_BLOCK_SIZE, 0, stream>>>(
        output, input, batch_size, seq_len, hidden_dim
    );
}

} // extern "C"
