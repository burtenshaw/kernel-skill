/*
 * PyTorch C++ Bindings for Qwen3-8B CUDA Kernels
 * Provides Python-callable wrappers for custom CUDA kernels.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAGuard.h>

// External declarations for CUDA kernel launch functions
extern "C" {
void rmsnorm_forward_fp16(__half*, const __half*, const __half*, int, int, int, float, cudaStream_t);
void rmsnorm_forward_bf16(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int, int, int, float, cudaStream_t);
void rmsnorm_forward_fp32(float*, const float*, const float*, int, int, int, float, cudaStream_t);
}

// ============================================================================
// RMSNorm Binding
// ============================================================================

void rmsnorm_forward(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    float eps
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    const at::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int ndim = input.dim();
    const int hidden_size = input.size(ndim - 1);
    const int64_t num_tokens = input.numel() / hidden_size;

    const int batch_size = 1;
    const int seq_len = num_tokens;

    if (input.scalar_type() == at::kHalf) {
        rmsnorm_forward_fp16(
            reinterpret_cast<__half*>(output.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<const __half*>(weight.data_ptr()),
            batch_size, seq_len, hidden_size, eps, stream
        );
    } else if (input.scalar_type() == at::kBFloat16) {
        rmsnorm_forward_bf16(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            batch_size, seq_len, hidden_size, eps, stream
        );
    } else if (input.scalar_type() == at::kFloat) {
        rmsnorm_forward_fp32(
            reinterpret_cast<float*>(output.data_ptr()),
            reinterpret_cast<const float*>(input.data_ptr()),
            reinterpret_cast<const float*>(weight.data_ptr()),
            batch_size, seq_len, hidden_size, eps, stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", input.scalar_type());
    }
}

// ============================================================================
// Module Registration
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward (CUDA)");
}
