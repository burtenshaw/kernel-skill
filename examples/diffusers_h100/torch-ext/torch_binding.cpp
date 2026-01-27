/*
 * PyTorch C++ Bindings for LTX-Video CUDA Kernels
 * Provides Python-callable wrappers for all custom CUDA kernels.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

// External declarations for CUDA kernel launch functions

// RMSNorm
extern "C" {
void rmsnorm_forward_fp16(__half*, const __half*, const __half*, int, int, int, float, cudaStream_t);
void rmsnorm_forward_bf16(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int, int, int, float, cudaStream_t);
void rmsnorm_forward_fp32(float*, const float*, const float*, int, int, int, float, cudaStream_t);

void rmsnorm_residual_forward_fp16(__half*, __half*, const __half*, const __half*, const __half*, int, int, int, float, cudaStream_t);
void rmsnorm_residual_forward_bf16(__nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int, int, int, float, cudaStream_t);
void rmsnorm_residual_forward_fp32(float*, float*, const float*, const float*, const float*, int, int, int, float, cudaStream_t);
}

// RoPE
extern "C" {
void rope_1d_forward_fp16(__half*, __half*, int, int, int, int, float, cudaStream_t);
void rope_1d_forward_bf16(__nv_bfloat16*, __nv_bfloat16*, int, int, int, int, float, cudaStream_t);
void rope_1d_forward_fp32(float*, float*, int, int, int, int, float, cudaStream_t);

void rope_3d_forward_fp16(__half*, __half*, int, int, int, int, int, int, float, cudaStream_t);
void rope_3d_forward_bf16(__nv_bfloat16*, __nv_bfloat16*, int, int, int, int, int, int, float, cudaStream_t);
void rope_3d_forward_fp32(float*, float*, int, int, int, int, int, int, float, cudaStream_t);

void rope_3d_extended_forward_fp16(__half*, __half*, int, int, int, int, int, int, int, int, int, float, float, float, cudaStream_t);
void rope_3d_extended_forward_bf16(__nv_bfloat16*, __nv_bfloat16*, int, int, int, int, int, int, int, int, int, float, float, float, cudaStream_t);
void rope_3d_extended_forward_fp32(float*, float*, int, int, int, int, int, int, int, int, int, float, float, float, cudaStream_t);
}

// GEGLU
extern "C" {
void geglu_forward_fp16(__half*, const __half*, int, int, int, cudaStream_t);
void geglu_forward_bf16(__nv_bfloat16*, const __nv_bfloat16*, int, int, int, cudaStream_t);
void geglu_forward_fp32(float*, const float*, int, int, int, cudaStream_t);

void geglu_exact_forward_fp16(__half*, const __half*, int, int, int, cudaStream_t);
void geglu_exact_forward_bf16(__nv_bfloat16*, const __nv_bfloat16*, int, int, int, cudaStream_t);
void geglu_exact_forward_fp32(float*, const float*, int, int, int, cudaStream_t);

void swiglu_forward_fp16(__half*, const __half*, int, int, int, cudaStream_t);
void swiglu_forward_bf16(__nv_bfloat16*, const __nv_bfloat16*, int, int, int, cudaStream_t);
void swiglu_forward_fp32(float*, const float*, int, int, int, cudaStream_t);
}

// AdaLN
extern "C" {
void adaln_layernorm_forward_fp16(__half*, const __half*, const __half*, const __half*, const __half*, const __half*, int, int, int, float, cudaStream_t);
void adaln_layernorm_forward_bf16(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int, int, int, float, cudaStream_t);
void adaln_layernorm_forward_fp32(float*, const float*, const float*, const float*, const float*, const float*, int, int, int, float, cudaStream_t);

void adaln_rmsnorm_forward_fp16(__half*, const __half*, const __half*, const __half*, const __half*, int, int, int, float, cudaStream_t);
void adaln_rmsnorm_forward_bf16(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int, int, int, float, cudaStream_t);
void adaln_rmsnorm_forward_fp32(float*, const float*, const float*, const float*, const float*, int, int, int, float, cudaStream_t);

void adaln_zero_forward_fp16(__half*, const __half*, const __half*, const __half*, const __half*, const __half*, int, int, int, float, cudaStream_t);
void adaln_zero_forward_bf16(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int, int, int, float, cudaStream_t);
void adaln_zero_forward_fp32(float*, const float*, const float*, const float*, const float*, const float*, int, int, int, float, cudaStream_t);

void adaln_residual_forward_fp16(__half*, const __half*, const __half*, const __half*, const __half*, const __half*, int, int, int, float, cudaStream_t);
void adaln_residual_forward_bf16(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int, int, int, float, cudaStream_t);
void adaln_residual_forward_fp32(float*, const float*, const float*, const float*, const float*, const float*, int, int, int, float, cudaStream_t);
}

// ============================================================================
// RMSNorm Bindings
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

    // Assume batch_size * seq_len = num_tokens
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

void rmsnorm_residual_forward(
    torch::Tensor& output,
    torch::Tensor& residual_out,
    const torch::Tensor& input,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    float eps
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");

    const at::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int ndim = input.dim();
    const int hidden_size = input.size(ndim - 1);
    const int64_t num_tokens = input.numel() / hidden_size;
    const int batch_size = 1;
    const int seq_len = num_tokens;

    if (input.scalar_type() == at::kHalf) {
        rmsnorm_residual_forward_fp16(
            reinterpret_cast<__half*>(output.data_ptr()),
            reinterpret_cast<__half*>(residual_out.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<const __half*>(residual.data_ptr()),
            reinterpret_cast<const __half*>(weight.data_ptr()),
            batch_size, seq_len, hidden_size, eps, stream
        );
    } else if (input.scalar_type() == at::kBFloat16) {
        rmsnorm_residual_forward_bf16(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            batch_size, seq_len, hidden_size, eps, stream
        );
    } else if (input.scalar_type() == at::kFloat) {
        rmsnorm_residual_forward_fp32(
            reinterpret_cast<float*>(output.data_ptr()),
            reinterpret_cast<float*>(residual_out.data_ptr()),
            reinterpret_cast<const float*>(input.data_ptr()),
            reinterpret_cast<const float*>(residual.data_ptr()),
            reinterpret_cast<const float*>(weight.data_ptr()),
            batch_size, seq_len, hidden_size, eps, stream
        );
    }
}

// ============================================================================
// RoPE Bindings
// ============================================================================

void rope_1d_forward(
    torch::Tensor& query,
    torch::Tensor& key,
    float theta_base
) {
    TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
    TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
    TORCH_CHECK(query.dim() == 4, "query must be 4D: [batch, seq, heads, head_dim]");
    TORCH_CHECK(key.dim() == 4, "key must be 4D: [batch, seq, heads, head_dim]");

    const at::cuda::CUDAGuard device_guard(query.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int batch_size = query.size(0);
    const int seq_len = query.size(1);
    const int num_heads = query.size(2);
    const int head_dim = query.size(3);

    if (query.scalar_type() == at::kHalf) {
        rope_1d_forward_fp16(
            reinterpret_cast<__half*>(query.data_ptr()),
            reinterpret_cast<__half*>(key.data_ptr()),
            batch_size, seq_len, num_heads, head_dim, theta_base, stream
        );
    } else if (query.scalar_type() == at::kBFloat16) {
        rope_1d_forward_bf16(
            reinterpret_cast<__nv_bfloat16*>(query.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(key.data_ptr()),
            batch_size, seq_len, num_heads, head_dim, theta_base, stream
        );
    } else if (query.scalar_type() == at::kFloat) {
        rope_1d_forward_fp32(
            reinterpret_cast<float*>(query.data_ptr()),
            reinterpret_cast<float*>(key.data_ptr()),
            batch_size, seq_len, num_heads, head_dim, theta_base, stream
        );
    }
}

void rope_3d_forward(
    torch::Tensor& query,
    torch::Tensor& key,
    int num_frames,
    int height,
    int width,
    float theta_base
) {
    TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
    TORCH_CHECK(query.dim() == 4, "query must be 4D: [batch, t*h*w, heads, head_dim]");

    const at::cuda::CUDAGuard device_guard(query.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int batch_size = query.size(0);
    const int num_heads = query.size(2);
    const int head_dim = query.size(3);

    TORCH_CHECK(query.size(1) == num_frames * height * width,
                "Sequence length must equal num_frames * height * width");

    if (query.scalar_type() == at::kHalf) {
        rope_3d_forward_fp16(
            reinterpret_cast<__half*>(query.data_ptr()),
            reinterpret_cast<__half*>(key.data_ptr()),
            batch_size, num_frames, height, width, num_heads, head_dim,
            theta_base, stream
        );
    } else if (query.scalar_type() == at::kBFloat16) {
        rope_3d_forward_bf16(
            reinterpret_cast<__nv_bfloat16*>(query.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(key.data_ptr()),
            batch_size, num_frames, height, width, num_heads, head_dim,
            theta_base, stream
        );
    } else if (query.scalar_type() == at::kFloat) {
        rope_3d_forward_fp32(
            reinterpret_cast<float*>(query.data_ptr()),
            reinterpret_cast<float*>(key.data_ptr()),
            batch_size, num_frames, height, width, num_heads, head_dim,
            theta_base, stream
        );
    }
}

void rope_3d_extended_forward(
    torch::Tensor& query,
    torch::Tensor& key,
    int num_frames,
    int height,
    int width,
    int rope_dim_t,
    int rope_dim_h,
    int rope_dim_w,
    float theta_base_t,
    float theta_base_h,
    float theta_base_w
) {
    TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");

    const at::cuda::CUDAGuard device_guard(query.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int batch_size = query.size(0);
    const int num_heads = query.size(2);
    const int head_dim = query.size(3);

    if (query.scalar_type() == at::kHalf) {
        rope_3d_extended_forward_fp16(
            reinterpret_cast<__half*>(query.data_ptr()),
            reinterpret_cast<__half*>(key.data_ptr()),
            batch_size, num_frames, height, width, num_heads, head_dim,
            rope_dim_t, rope_dim_h, rope_dim_w,
            theta_base_t, theta_base_h, theta_base_w, stream
        );
    } else if (query.scalar_type() == at::kBFloat16) {
        rope_3d_extended_forward_bf16(
            reinterpret_cast<__nv_bfloat16*>(query.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(key.data_ptr()),
            batch_size, num_frames, height, width, num_heads, head_dim,
            rope_dim_t, rope_dim_h, rope_dim_w,
            theta_base_t, theta_base_h, theta_base_w, stream
        );
    } else if (query.scalar_type() == at::kFloat) {
        rope_3d_extended_forward_fp32(
            reinterpret_cast<float*>(query.data_ptr()),
            reinterpret_cast<float*>(key.data_ptr()),
            batch_size, num_frames, height, width, num_heads, head_dim,
            rope_dim_t, rope_dim_h, rope_dim_w,
            theta_base_t, theta_base_h, theta_base_w, stream
        );
    }
}

// ============================================================================
// GEGLU Bindings
// ============================================================================

void geglu_forward(
    torch::Tensor& output,
    const torch::Tensor& input
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    const at::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int ndim = input.dim();
    const int input_hidden = input.size(ndim - 1);
    const int hidden_dim = input_hidden / 2;
    const int64_t num_tokens = input.numel() / input_hidden;

    const int batch_size = 1;
    const int seq_len = num_tokens;

    if (input.scalar_type() == at::kHalf) {
        geglu_forward_fp16(
            reinterpret_cast<__half*>(output.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            batch_size, seq_len, hidden_dim, stream
        );
    } else if (input.scalar_type() == at::kBFloat16) {
        geglu_forward_bf16(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            batch_size, seq_len, hidden_dim, stream
        );
    } else if (input.scalar_type() == at::kFloat) {
        geglu_forward_fp32(
            reinterpret_cast<float*>(output.data_ptr()),
            reinterpret_cast<const float*>(input.data_ptr()),
            batch_size, seq_len, hidden_dim, stream
        );
    }
}

void geglu_exact_forward(
    torch::Tensor& output,
    const torch::Tensor& input
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");

    const at::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int ndim = input.dim();
    const int input_hidden = input.size(ndim - 1);
    const int hidden_dim = input_hidden / 2;
    const int64_t num_tokens = input.numel() / input_hidden;

    const int batch_size = 1;
    const int seq_len = num_tokens;

    if (input.scalar_type() == at::kHalf) {
        geglu_exact_forward_fp16(
            reinterpret_cast<__half*>(output.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            batch_size, seq_len, hidden_dim, stream
        );
    } else if (input.scalar_type() == at::kBFloat16) {
        geglu_exact_forward_bf16(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            batch_size, seq_len, hidden_dim, stream
        );
    } else if (input.scalar_type() == at::kFloat) {
        geglu_exact_forward_fp32(
            reinterpret_cast<float*>(output.data_ptr()),
            reinterpret_cast<const float*>(input.data_ptr()),
            batch_size, seq_len, hidden_dim, stream
        );
    }
}

void swiglu_forward(
    torch::Tensor& output,
    const torch::Tensor& input
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");

    const at::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int ndim = input.dim();
    const int input_hidden = input.size(ndim - 1);
    const int hidden_dim = input_hidden / 2;
    const int64_t num_tokens = input.numel() / input_hidden;

    const int batch_size = 1;
    const int seq_len = num_tokens;

    if (input.scalar_type() == at::kHalf) {
        swiglu_forward_fp16(
            reinterpret_cast<__half*>(output.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            batch_size, seq_len, hidden_dim, stream
        );
    } else if (input.scalar_type() == at::kBFloat16) {
        swiglu_forward_bf16(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            batch_size, seq_len, hidden_dim, stream
        );
    } else if (input.scalar_type() == at::kFloat) {
        swiglu_forward_fp32(
            reinterpret_cast<float*>(output.data_ptr()),
            reinterpret_cast<const float*>(input.data_ptr()),
            batch_size, seq_len, hidden_dim, stream
        );
    }
}

// ============================================================================
// AdaLN Bindings
// ============================================================================

void adaln_layernorm_forward(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const torch::Tensor& scale,
    const torch::Tensor& shift,
    float eps
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");

    const at::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int hidden_size = input.size(2);

    const void* bias_ptr = bias.has_value() ? bias->data_ptr() : nullptr;

    if (input.scalar_type() == at::kHalf) {
        adaln_layernorm_forward_fp16(
            reinterpret_cast<__half*>(output.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<const __half*>(weight.data_ptr()),
            reinterpret_cast<const __half*>(bias_ptr),
            reinterpret_cast<const __half*>(scale.data_ptr()),
            reinterpret_cast<const __half*>(shift.data_ptr()),
            batch_size, seq_len, hidden_size, eps, stream
        );
    } else if (input.scalar_type() == at::kBFloat16) {
        adaln_layernorm_forward_bf16(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(bias_ptr),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(shift.data_ptr()),
            batch_size, seq_len, hidden_size, eps, stream
        );
    } else if (input.scalar_type() == at::kFloat) {
        adaln_layernorm_forward_fp32(
            reinterpret_cast<float*>(output.data_ptr()),
            reinterpret_cast<const float*>(input.data_ptr()),
            reinterpret_cast<const float*>(weight.data_ptr()),
            reinterpret_cast<const float*>(bias_ptr),
            reinterpret_cast<const float*>(scale.data_ptr()),
            reinterpret_cast<const float*>(shift.data_ptr()),
            batch_size, seq_len, hidden_size, eps, stream
        );
    }
}

void adaln_rmsnorm_forward(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& scale,
    const torch::Tensor& shift,
    float eps
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");

    const at::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int hidden_size = input.size(2);

    if (input.scalar_type() == at::kHalf) {
        adaln_rmsnorm_forward_fp16(
            reinterpret_cast<__half*>(output.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<const __half*>(weight.data_ptr()),
            reinterpret_cast<const __half*>(scale.data_ptr()),
            reinterpret_cast<const __half*>(shift.data_ptr()),
            batch_size, seq_len, hidden_size, eps, stream
        );
    } else if (input.scalar_type() == at::kBFloat16) {
        adaln_rmsnorm_forward_bf16(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(shift.data_ptr()),
            batch_size, seq_len, hidden_size, eps, stream
        );
    } else if (input.scalar_type() == at::kFloat) {
        adaln_rmsnorm_forward_fp32(
            reinterpret_cast<float*>(output.data_ptr()),
            reinterpret_cast<const float*>(input.data_ptr()),
            reinterpret_cast<const float*>(weight.data_ptr()),
            reinterpret_cast<const float*>(scale.data_ptr()),
            reinterpret_cast<const float*>(shift.data_ptr()),
            batch_size, seq_len, hidden_size, eps, stream
        );
    }
}

void adaln_zero_forward(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& scale,
    const torch::Tensor& shift,
    const torch::Tensor& gate,
    float eps
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");

    const at::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int hidden_size = input.size(2);

    if (input.scalar_type() == at::kHalf) {
        adaln_zero_forward_fp16(
            reinterpret_cast<__half*>(output.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<const __half*>(weight.data_ptr()),
            reinterpret_cast<const __half*>(scale.data_ptr()),
            reinterpret_cast<const __half*>(shift.data_ptr()),
            reinterpret_cast<const __half*>(gate.data_ptr()),
            batch_size, seq_len, hidden_size, eps, stream
        );
    } else if (input.scalar_type() == at::kBFloat16) {
        adaln_zero_forward_bf16(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(shift.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()),
            batch_size, seq_len, hidden_size, eps, stream
        );
    } else if (input.scalar_type() == at::kFloat) {
        adaln_zero_forward_fp32(
            reinterpret_cast<float*>(output.data_ptr()),
            reinterpret_cast<const float*>(input.data_ptr()),
            reinterpret_cast<const float*>(weight.data_ptr()),
            reinterpret_cast<const float*>(scale.data_ptr()),
            reinterpret_cast<const float*>(shift.data_ptr()),
            reinterpret_cast<const float*>(gate.data_ptr()),
            batch_size, seq_len, hidden_size, eps, stream
        );
    }
}

// ============================================================================
// Module Registration
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // RMSNorm
    m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward (CUDA)");
    m.def("rmsnorm_residual_forward", &rmsnorm_residual_forward, "RMSNorm with residual forward (CUDA)");

    // RoPE
    m.def("rope_1d_forward", &rope_1d_forward, "1D RoPE forward (CUDA)");
    m.def("rope_3d_forward", &rope_3d_forward, "3D RoPE forward for video (CUDA)");
    m.def("rope_3d_extended_forward", &rope_3d_extended_forward, "3D RoPE with custom dims (CUDA)");

    // GEGLU
    m.def("geglu_forward", &geglu_forward, "GEGLU forward with tanh approx (CUDA)");
    m.def("geglu_exact_forward", &geglu_exact_forward, "GEGLU forward with exact GELU (CUDA)");
    m.def("swiglu_forward", &swiglu_forward, "SwiGLU forward (CUDA)");

    // AdaLN
    m.def("adaln_layernorm_forward", &adaln_layernorm_forward, "AdaLN with LayerNorm (CUDA)");
    m.def("adaln_rmsnorm_forward", &adaln_rmsnorm_forward, "AdaLN with RMSNorm (CUDA)");
    m.def("adaln_zero_forward", &adaln_zero_forward, "AdaLN-Zero forward (CUDA)");
}
