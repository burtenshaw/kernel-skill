"""Python API for the Qwen3 RMSNorm kernel demo."""

from typing import Optional
import torch

try:
    from ._ops import ops as _builder_ops
except ImportError:
    _builder_ops = None

try:
    from . import _ops as _legacy_ops
except ImportError:
    _legacy_ops = None


def _call_compiled_rmsnorm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> bool:
    if _builder_ops is not None:
        _builder_ops.rmsnorm(out, input.contiguous(), weight.contiguous(), float(eps))
        return True

    if _legacy_ops is not None and hasattr(_legacy_ops, "rmsnorm"):
        _legacy_ops.rmsnorm(out, input.contiguous(), weight.contiguous(), float(eps))
        return True

    return False


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply RMSNorm to the last dimension of ``input``.
    """
    if out is None:
        out = torch.empty_like(input)

    if _call_compiled_rmsnorm(out, input, weight, eps):
        return out

    variance = input.pow(2).mean(dim=-1, keepdim=True)
    result = input * torch.rsqrt(variance + eps) * weight
    out.copy_(result)
    return out


def patch_rmsnorm_modules(model: torch.nn.Module) -> int:
    """Patch RMSNorm modules in a transformers model to use this kernel."""
    patched = 0

    for module in model.modules():
        if "RMSNorm" not in type(module).__name__:
            continue
        if getattr(module, "_qwen3_kernel_patched", False):
            continue

        weight = getattr(module, "weight", None)
        if weight is None:
            continue

        eps = getattr(module, "variance_epsilon", None)
        if eps is None:
            eps = getattr(module, "eps", 1e-6)

        original_forward = module.forward

        def make_forward(mod: torch.nn.Module, epsilon: float):
            def forward(hidden_states: torch.Tensor) -> torch.Tensor:
                return rmsnorm(hidden_states, mod.weight, eps=epsilon)

            return forward

        module._qwen3_original_forward = original_forward
        module._qwen3_kernel_patched = True
        module.forward = make_forward(module, float(eps))
        patched += 1

    return patched


__version__ = "0.1.0"
__all__ = ["patch_rmsnorm_modules", "rmsnorm"]
