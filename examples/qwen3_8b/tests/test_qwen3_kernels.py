import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "torch-ext"))

from qwen3_kernels import patch_rmsnorm_modules, rmsnorm


def reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def test_rmsnorm_fallback_matches_reference() -> None:
    x = torch.randn(2, 3, 8, dtype=torch.float32)
    weight = torch.randn(8, dtype=torch.float32)

    out = rmsnorm(x, weight, eps=1e-6)
    expected = reference_rmsnorm(x, weight, eps=1e-6)

    torch.testing.assert_close(out, expected)


def test_patch_rmsnorm_modules_patches_transformers_style_modules() -> None:
    class FakeRMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states + 1

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layernorm = FakeRMSNorm(8, 1e-6)
            self.post_attention_layernorm = FakeRMSNorm(8, 1e-5)

    model = ToyModel()
    patched = patch_rmsnorm_modules(model)

    assert patched == 2

    x = torch.randn(2, 3, 8)
    torch.testing.assert_close(
        model.input_layernorm(x),
        reference_rmsnorm(x, model.input_layernorm.weight, 1e-6),
    )
    torch.testing.assert_close(
        model.post_attention_layernorm(x),
        reference_rmsnorm(x, model.post_attention_layernorm.weight, 1e-5),
    )
