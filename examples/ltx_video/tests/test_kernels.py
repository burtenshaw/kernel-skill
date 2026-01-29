"""
Tests for LTX-Video custom CUDA kernels.

Run with: pytest tests/test_kernels.py -v
"""

import pytest
import torch
import sys
import os

# Add kernel module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'torch-ext'))

from ltx_kernels import (
    rmsnorm,
    rmsnorm_residual,
    rope_1d,
    rope_3d,
    geglu,
    swiglu,
    adaln_rmsnorm,
)


# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestRMSNorm:
    """Tests for RMSNorm kernel."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shape", [(2, 1024, 2048), (1, 4096, 4096)])
    def test_rmsnorm_correctness(self, dtype, shape):
        device = "cuda"
        eps = 1e-6

        input = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.ones(shape[-1], dtype=dtype, device=device)

        # Reference implementation
        variance = input.pow(2).mean(dim=-1, keepdim=True)
        expected = input * torch.rsqrt(variance + eps) * weight

        # Kernel implementation
        output = rmsnorm(input, weight, eps=eps)

        # Compare
        rtol = 1e-2 if dtype == torch.float16 else 1e-4
        atol = 1e-3 if dtype == torch.float16 else 1e-5
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_rmsnorm_residual(self, dtype):
        device = "cuda"
        shape = (2, 512, 1024)
        eps = 1e-6

        input = torch.randn(shape, dtype=dtype, device=device)
        residual = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.ones(shape[-1], dtype=dtype, device=device)

        # Reference
        combined = input + residual
        variance = combined.pow(2).mean(dim=-1, keepdim=True)
        expected = combined * torch.rsqrt(variance + eps) * weight

        # Kernel
        output, residual_out = rmsnorm_residual(input, residual, weight, eps=eps)

        rtol = 1e-3 if dtype == torch.bfloat16 else 1e-4
        atol = 1e-3 if dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
        torch.testing.assert_close(residual_out, combined, rtol=rtol, atol=atol)


class TestRoPE:
    """Tests for Rotary Position Embeddings kernels."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_rope_1d_shape(self, dtype):
        device = "cuda"
        batch, seq, heads, head_dim = 2, 128, 8, 64

        query = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
        key = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)

        q_out, k_out = rope_1d(query.clone(), key.clone())

        assert q_out.shape == query.shape
        assert k_out.shape == key.shape

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_rope_3d_shape(self, dtype):
        device = "cuda"
        batch, num_frames, height, width = 1, 5, 8, 12
        heads, head_dim = 8, 64
        seq = num_frames * height * width

        query = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
        key = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)

        q_out, k_out = rope_3d(query.clone(), key.clone(), num_frames, height, width)

        assert q_out.shape == query.shape
        assert k_out.shape == key.shape

    def test_rope_3d_video_dimensions(self):
        """Test 3D RoPE with typical LTX-Video dimensions."""
        device = "cuda"
        dtype = torch.bfloat16

        # Typical LTX-Video dimensions
        batch = 1
        num_frames = 25
        height = 30  # 480 / 16
        width = 44   # 704 / 16
        heads = 16
        head_dim = 64
        seq = num_frames * height * width

        query = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
        key = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)

        q_out, k_out = rope_3d(query.clone(), key.clone(), num_frames, height, width)

        assert q_out.shape == (batch, seq, heads, head_dim)
        assert not torch.isnan(q_out).any()
        assert not torch.isnan(k_out).any()


class TestGEGLU:
    """Tests for GEGLU and SwiGLU kernels."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shape", [(2, 512, 4096), (1, 1024, 8192)])
    def test_geglu_correctness(self, dtype, shape):
        device = "cuda"
        input = torch.randn(shape, dtype=dtype, device=device)

        # Reference
        gate, value = input.chunk(2, dim=-1)
        expected = torch.nn.functional.gelu(gate, approximate='tanh') * value

        # Kernel
        output = geglu(input, approximate=True)

        rtol = 1e-2 if dtype == torch.float16 else 1e-3
        atol = 1e-2 if dtype == torch.float16 else 1e-4
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_swiglu_correctness(self, dtype):
        device = "cuda"
        shape = (2, 512, 4096)
        input = torch.randn(shape, dtype=dtype, device=device)

        # Reference
        gate, value = input.chunk(2, dim=-1)
        expected = torch.nn.functional.silu(gate) * value

        # Kernel
        output = swiglu(input)

        rtol = 1e-3
        atol = 1e-4
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


class TestAdaLN:
    """Tests for Adaptive Layer Normalization kernels."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_adaln_rmsnorm_correctness(self, dtype):
        device = "cuda"
        batch, seq, hidden = 2, 256, 1024
        eps = 1e-6

        input = torch.randn(batch, seq, hidden, dtype=dtype, device=device)
        weight = torch.ones(hidden, dtype=dtype, device=device)
        scale = torch.randn(batch, hidden, dtype=dtype, device=device) * 0.1
        shift = torch.randn(batch, hidden, dtype=dtype, device=device) * 0.1

        # Reference
        variance = input.pow(2).mean(dim=-1, keepdim=True)
        normalized = input * torch.rsqrt(variance + eps)
        expected = normalized * weight * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # Kernel
        output = adaln_rmsnorm(input, weight, scale, shift, eps=eps)

        rtol = 1e-3 if dtype == torch.bfloat16 else 1e-4
        atol = 1e-3 if dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


class TestPreallocated:
    """Test pre-allocated output tensors."""

    def test_rmsnorm_preallocated(self):
        device = "cuda"
        dtype = torch.bfloat16
        shape = (2, 1024, 2048)

        input = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.ones(shape[-1], dtype=dtype, device=device)
        output = torch.empty_like(input)

        result = rmsnorm(input, weight, out=output)

        assert result is output
        assert not torch.isnan(result).any()

    def test_geglu_preallocated(self):
        device = "cuda"
        dtype = torch.bfloat16
        shape = (2, 512, 4096)

        input = torch.randn(shape, dtype=dtype, device=device)
        out_shape = list(shape)
        out_shape[-1] = shape[-1] // 2
        output = torch.empty(out_shape, dtype=dtype, device=device)

        result = geglu(input, out=output)

        assert result is output
        assert not torch.isnan(result).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
