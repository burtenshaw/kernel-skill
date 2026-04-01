/*
 * PyTorch C++ Bindings for Qwen3-8B CUDA Kernels
 * Provides Python-callable wrappers for custom CUDA kernels.
 */

#include <torch/extension.h>
#include <torch/library.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAGuard.h>

#include "torch_binding.h"

#if __has_include("registration.h")
#include "registration.h"
#define QWEN3_KERNEL_BUILDER 1
#else
#define QWEN3_KERNEL_BUILDER 0
#endif

// External declarations for CUDA kernel launch functions
extern "C" {
void rmsnorm_forward_fp16(__half*, const __half*, const __half*, int, int, int, float, cudaStream_t);
void rmsnorm_forward_bf16(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int, int, int, float, cudaStream_t);
void rmsnorm_forward_fp32(float*, const float*, const float*, int, int, int, float, cudaStream_t);
}

// ============================================================================
// RMSNorm Binding
// ============================================================================

void rmsnorm(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    float eps
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "input and weight must have the same dtype");
    TORCH_CHECK(output.scalar_type() == input.scalar_type(), "output must match the input dtype");
    TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
    TORCH_CHECK(weight.dim() == 1, "weight must be a 1D tensor");

    const at::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int ndim = input.dim();
    const int hidden_size = input.size(ndim - 1);
    const int64_t num_tokens = input.numel() / hidden_size;
    TORCH_CHECK(weight.numel() == hidden_size, "weight size must match the hidden dimension");
    TORCH_CHECK(output.sizes() == input.sizes(), "output must match the input shape");

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

#if QWEN3_KERNEL_BUILDER
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("rmsnorm(Tensor! out, Tensor input, Tensor weight, float eps) -> ()");
    ops.impl("rmsnorm", torch::kCUDA, &rmsnorm);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
#else
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm", &rmsnorm, "RMSNorm forward (CUDA)");
}
#endif
