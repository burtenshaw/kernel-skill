"""
Qwen3-8B Custom CUDA Kernels
============================

Optimized CUDA kernels for Qwen/Qwen3-8B model on H100 GPUs (sm_90).

Qwen3-8B Architecture:
- hidden_size: 4096
- num_hidden_layers: 32
- rms_norm_eps: 1e-6
- RMSNorm modules: 65 (32 layers * 2 + 1 final)

Kernels:
- rmsnorm: Vectorized RMS Layer Normalization with warp reductions

Usage with transformers:
    from transformers import AutoModelForCausalLM
    from qwen3_kernels import rmsnorm

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )

    # Patch RMSNorm modules
    for name, module in model.named_modules():
        if 'RMSNorm' in type(module).__name__:
            eps = getattr(module, 'variance_epsilon', 1e-6)
            def make_forward(mod, epsilon):
                def forward(x):
                    return rmsnorm(x, mod.weight, eps=epsilon)
                return forward
            module.forward = make_forward(module, eps)
"""

from typing import Optional
import torch

# Import the compiled extension
try:
    from . import _ops as ops
except ImportError:
    ops = None


# =============================================================================
# Register RMSNorm as PyTorch custom op for torch.compile compatibility
# =============================================================================

_CUSTOM_OP_REGISTERED = False

if ops is not None:
    try:
        @torch.library.custom_op("qwen3_kernels::rmsnorm_forward", mutates_args=())
        def _rmsnorm_custom_op(
            input: torch.Tensor,
            weight: torch.Tensor,
            eps: float,
        ) -> torch.Tensor:
            out = torch.empty_like(input)
            ops.rmsnorm_forward(out, input.contiguous(), weight.contiguous(), eps)
            return out

        @_rmsnorm_custom_op.register_fake
        def _rmsnorm_fake(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
            return input.new_empty(input.shape)

        _CUSTOM_OP_REGISTERED = True
    except (AttributeError, Exception):
        pass


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    RMS Layer Normalization (torch.compile compatible).

    Formula: output = x * weight / sqrt(mean(x^2) + eps)

    Optimized for Qwen3-8B's hidden_size=4096 with vectorized BF16/FP16 loads.

    Args:
        input: Input tensor of shape [..., hidden_size]
        weight: Weight tensor of shape [hidden_size]
        eps: Epsilon for numerical stability (default: 1e-6, Qwen3 default)
        out: Optional pre-allocated output tensor

    Returns:
        Normalized tensor of same shape as input
    """
    if _CUSTOM_OP_REGISTERED:
        return torch.ops.qwen3_kernels.rmsnorm_forward(input, weight, eps)
    elif ops is not None:
        if out is None:
            out = torch.empty_like(input)
        ops.rmsnorm_forward(out, input.contiguous(), weight.contiguous(), eps)
        return out
    else:
        # Pure PyTorch fallback
        variance = input.pow(2).mean(dim=-1, keepdim=True)
        return input * torch.rsqrt(variance + eps) * weight


# Version info
__version__ = "0.1.0"
__all__ = ["rmsnorm"]
