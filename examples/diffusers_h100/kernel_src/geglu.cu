/*
 * GEGLU (GELU-Gated Linear Unit) CUDA Kernel for LTX-Video
 * Optimized for NVIDIA H100 (sm_90)
 *
 * GEGLU formula: output = GELU(gate) * value
 * where input is split in half: [gate, value]
 *
 * Uses tanh approximation for GELU (faster, matches LTX-Video default):
 * GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

constexpr int GEGLU_BLOCK_SIZE = 256;

// GELU constants for tanh approximation
constexpr float GELU_COEF_A = 0.7978845608028654f;  // sqrt(2/π)
constexpr float GELU_COEF_B = 0.044715f;

// Fast GELU with tanh approximation
__device__ __forceinline__ float gelu_tanh(float x) {
    float x3 = x * x * x;
    float inner = GELU_COEF_A * (x + GELU_COEF_B * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Exact GELU using erf (more precise but slower)
__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));  // 1/sqrt(2)
}

// SiLU/Swish activation (alternative)
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// GEGLU kernel with tanh approximation
// Input: [batch, seq_len, 2 * hidden_dim]
// Output: [batch, seq_len, hidden_dim]
template <typename scalar_t>
__global__ void geglu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int batch_size,
    const int seq_len,
    const int hidden_dim  // Output hidden dim (half of input)
) {
    const int total_elements = batch_size * seq_len * hidden_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    // Decode indices
    const int h_idx = idx % hidden_dim;
    const int seq_idx = (idx / hidden_dim) % seq_len;
    const int batch_idx = idx / (hidden_dim * seq_len);

    // Input has 2x hidden_dim
    const int input_hidden = 2 * hidden_dim;
    const int input_base = batch_idx * (seq_len * input_hidden) + seq_idx * input_hidden;

    // First half is gate, second half is value
    float gate = static_cast<float>(input[input_base + h_idx]);
    float value = static_cast<float>(input[input_base + hidden_dim + h_idx]);

    // Apply GELU to gate and multiply with value
    float result = gelu_tanh(gate) * value;

    output[idx] = static_cast<scalar_t>(result);
}

// GEGLU kernel with exact GELU (erf-based)
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

    const int input_hidden = 2 * hidden_dim;
    const int input_base = batch_idx * (seq_len * input_hidden) + seq_idx * input_hidden;

    float gate = static_cast<float>(input[input_base + h_idx]);
    float value = static_cast<float>(input[input_base + hidden_dim + h_idx]);

    float result = gelu_exact(gate) * value;

    output[idx] = static_cast<scalar_t>(result);
}

// SwiGLU kernel (SiLU-gated variant)
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

    const int input_hidden = 2 * hidden_dim;
    const int input_base = batch_idx * (seq_len * input_hidden) + seq_idx * input_hidden;

    float gate = static_cast<float>(input[input_base + h_idx]);
    float value = static_cast<float>(input[input_base + hidden_dim + h_idx]);

    float result = silu(gate) * value;

    output[idx] = static_cast<scalar_t>(result);
}

// Fused linear + GEGLU kernel
// Performs: output = GEGLU(input @ weight + bias)
// Weight shape: [input_dim, 2 * hidden_dim]
// This is memory-bandwidth bound, so fusing is beneficial
template <typename scalar_t>
__global__ void fused_geglu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,  // Can be nullptr
    const int batch_size,
    const int seq_len,
    const int input_dim,
    const int hidden_dim  // Output hidden dim
) {
    // This kernel handles small reductions inline
    // For large input_dim, use cuBLAS + separate GEGLU
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_outputs = batch_size * seq_len * hidden_dim;

    if (idx >= total_outputs) return;

    const int h_idx = idx % hidden_dim;
    const int seq_idx = (idx / hidden_dim) % seq_len;
    const int batch_idx = idx / (hidden_dim * seq_len);

    // Input position
    const scalar_t* input_row = input + batch_idx * (seq_len * input_dim) + seq_idx * input_dim;

    // Weight columns for gate and value
    const int weight_stride = 2 * hidden_dim;

    // Compute gate and value via dot product
    float gate = 0.0f;
    float value = 0.0f;

    for (int i = 0; i < input_dim; i++) {
        float in_val = static_cast<float>(input_row[i]);
        gate += in_val * static_cast<float>(weight[i * weight_stride + h_idx]);
        value += in_val * static_cast<float>(weight[i * weight_stride + hidden_dim + h_idx]);
    }

    // Add bias if present
    if (bias != nullptr) {
        gate += static_cast<float>(bias[h_idx]);
        value += static_cast<float>(bias[hidden_dim + h_idx]);
    }

    // Apply GEGLU
    float result = gelu_tanh(gate) * value;

    output[idx] = static_cast<scalar_t>(result);
}

// Launch functions
extern "C" {

// Standard GEGLU (tanh approximation)
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

// Exact GEGLU (erf-based)
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

// SwiGLU (SiLU-gated)
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
