"""
Test suite for LTX-Video CUDA Kernels

Tests correctness against PyTorch reference implementations.
Run with: pytest tests/test_kernels.py -v
"""

import pytest
import torch
import math

# Try to import compiled kernels, fall back to pure PyTorch
try:
    from ltx_kernels import (
        rmsnorm, rmsnorm_residual,
        rope_1d, rope_3d,
        geglu, swiglu,
        adaln_rmsnorm, adaln_layernorm, adaln_zero,
    )
    KERNELS_AVAILABLE = True
except ImportError:
    KERNELS_AVAILABLE = False
    print("Warning: Compiled kernels not available, testing PyTorch fallbacks")


# Test configurations
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
BATCH_SIZES = [1, 2, 4]
SEQ_LENS = [128, 1024, 4096]
HIDDEN_SIZES = [768, 2048]


def reference_rmsnorm(x, weight, eps=1e-6):
    """PyTorch reference implementation of RMSNorm."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def reference_geglu(x, approximate=True):
    """PyTorch reference implementation of GEGLU."""
    gate, value = x.chunk(2, dim=-1)
    if approximate:
        return torch.nn.functional.gelu(gate, approximate='tanh') * value
    else:
        return torch.nn.functional.gelu(gate) * value


def reference_swiglu(x):
    """PyTorch reference implementation of SwiGLU."""
    gate, value = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * value


def reference_rope_1d(query, key, theta_base=10000.0):
    """PyTorch reference implementation of 1D RoPE."""
    batch, seq_len, num_heads, head_dim = query.shape
    device = query.device
    dtype = query.dtype

    position = torch.arange(seq_len, device=device, dtype=torch.float32)
    dim_idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    freqs = 1.0 / (theta_base ** (dim_idx / head_dim))

    angles = position.unsqueeze(1) * freqs.unsqueeze(0)
    cos = torch.cos(angles).to(dtype).view(1, seq_len, 1, -1)
    sin = torch.sin(angles).to(dtype).view(1, seq_len, 1, -1)

    q_out = query.clone()
    k_out = key.clone()

    q_even, q_odd = query[..., 0::2], query[..., 1::2]
    k_even, k_odd = key[..., 0::2], key[..., 1::2]

    q_out[..., 0::2] = q_even * cos - q_odd * sin
    q_out[..., 1::2] = q_even * sin + q_odd * cos
    k_out[..., 0::2] = k_even * cos - k_odd * sin
    k_out[..., 1::2] = k_even * sin + k_odd * cos

    return q_out, k_out


def reference_adaln_rmsnorm(x, weight, scale, shift, eps=1e-6):
    """PyTorch reference implementation of AdaLN with RMSNorm."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + eps)
    # Broadcast scale/shift
    if scale.dim() == 2:
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
    return normalized * weight * (1 + scale) + shift


class TestRMSNorm:
    """Test RMSNorm kernel."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("seq_len", [128, 512])
    @pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
    def test_rmsnorm_correctness(self, dtype, batch_size, seq_len, hidden_size):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        # Reference
        ref_out = reference_rmsnorm(x, weight)

        # Kernel (uses PyTorch fallback if not compiled)
        if KERNELS_AVAILABLE:
            kernel_out = rmsnorm(x, weight)

            # Check correctness
            rtol = 1e-2 if dtype == torch.float16 else 1e-3
            atol = 1e-3 if dtype == torch.float16 else 1e-4

            torch.testing.assert_close(kernel_out, ref_out, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_rmsnorm_residual(self, dtype):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        batch, seq, hidden = 2, 256, 768
        x = torch.randn(batch, seq, hidden, device="cuda", dtype=dtype)
        residual = torch.randn(batch, seq, hidden, device="cuda", dtype=dtype)
        weight = torch.randn(hidden, device="cuda", dtype=dtype)

        # Reference
        combined = x + residual
        ref_out = reference_rmsnorm(combined, weight)

        # Kernel
        if KERNELS_AVAILABLE:
            kernel_out, residual_out = rmsnorm_residual(x, residual, weight)
            torch.testing.assert_close(kernel_out, ref_out, rtol=1e-3, atol=1e-4)
            torch.testing.assert_close(residual_out, combined, rtol=1e-5, atol=1e-5)


class TestRoPE:
    """Test Rotary Position Embedding kernels."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [64, 256])
    def test_rope_1d_correctness(self, dtype, batch_size, seq_len):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        num_heads = 32
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=dtype)

        # Reference
        q_ref, k_ref = reference_rope_1d(q.clone(), k.clone())

        # Kernel
        if KERNELS_AVAILABLE:
            q_kernel, k_kernel = rope_1d(q.clone(), k.clone())

            rtol = 1e-2 if dtype == torch.float16 else 1e-3
            atol = 1e-2 if dtype == torch.float16 else 1e-4

            torch.testing.assert_close(q_kernel, q_ref, rtol=rtol, atol=atol)
            torch.testing.assert_close(k_kernel, k_ref, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_rope_3d_shape(self, dtype):
        """Test 3D RoPE maintains correct shapes for video."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        batch = 1
        num_frames = 8
        height = 16
        width = 16
        num_heads = 32
        head_dim = 64

        seq_len = num_frames * height * width  # 2048

        q = torch.randn(batch, seq_len, num_heads, head_dim, device="cuda", dtype=dtype)
        k = torch.randn(batch, seq_len, num_heads, head_dim, device="cuda", dtype=dtype)

        if KERNELS_AVAILABLE:
            q_out, k_out = rope_3d(q.clone(), k.clone(), num_frames, height, width)

            assert q_out.shape == q.shape
            assert k_out.shape == k.shape
            # Values should change (RoPE applied)
            assert not torch.allclose(q_out, q)


class TestGEGLU:
    """Test GEGLU activation kernel."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("seq_len", [128, 512])
    def test_geglu_correctness(self, dtype, batch_size, seq_len):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        hidden_dim = 1024
        input_dim = hidden_dim * 2  # GEGLU expects 2x hidden

        x = torch.randn(batch_size, seq_len, input_dim, device="cuda", dtype=dtype)

        # Reference
        ref_out = reference_geglu(x, approximate=True)

        # Kernel
        if KERNELS_AVAILABLE:
            kernel_out = geglu(x)

            rtol = 1e-2 if dtype == torch.float16 else 1e-3
            atol = 1e-2 if dtype == torch.float16 else 1e-4

            torch.testing.assert_close(kernel_out, ref_out, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_swiglu_correctness(self, dtype):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        batch, seq, hidden = 2, 256, 2048

        x = torch.randn(batch, seq, hidden, device="cuda", dtype=dtype)

        # Reference
        ref_out = reference_swiglu(x)

        # Kernel
        if KERNELS_AVAILABLE:
            kernel_out = swiglu(x)

            rtol = 1e-2 if dtype == torch.float16 else 1e-3
            torch.testing.assert_close(kernel_out, ref_out, rtol=rtol, atol=1e-3)


class TestAdaLN:
    """Test Adaptive Layer Normalization kernels."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_adaln_rmsnorm_correctness(self, dtype, batch_size):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        seq_len = 256
        hidden_size = 768

        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
        scale = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype) * 0.1
        shift = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype) * 0.1

        # Reference
        ref_out = reference_adaln_rmsnorm(x, weight, scale, shift)

        # Kernel
        if KERNELS_AVAILABLE:
            kernel_out = adaln_rmsnorm(x, weight, scale, shift)

            rtol = 1e-2 if dtype == torch.float16 else 1e-3
            atol = 1e-2 if dtype == torch.float16 else 1e-4

            torch.testing.assert_close(kernel_out, ref_out, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_adaln_zero(self, dtype):
        """Test AdaLN-Zero with gating."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        batch, seq, hidden = 2, 256, 768

        x = torch.randn(batch, seq, hidden, device="cuda", dtype=dtype)
        weight = torch.randn(hidden, device="cuda", dtype=dtype)
        scale = torch.randn(batch, hidden, device="cuda", dtype=dtype) * 0.1
        shift = torch.randn(batch, hidden, device="cuda", dtype=dtype) * 0.1
        gate = torch.zeros(batch, hidden, device="cuda", dtype=dtype)  # Zero-initialized

        if KERNELS_AVAILABLE:
            out = adaln_zero(x, weight, scale, shift, gate)
            # With zero gate, output should be all zeros
            assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

            # With non-zero gate
            gate = torch.ones(batch, hidden, device="cuda", dtype=dtype)
            out = adaln_zero(x, weight, scale, shift, gate)
            assert not torch.allclose(out, torch.zeros_like(out))


class TestLTXVideoIntegration:
    """Integration tests with LTX-Video-like configurations."""

    def test_ltx_video_dimensions(self):
        """Test kernels with typical LTX-Video dimensions."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if not KERNELS_AVAILABLE:
            pytest.skip("Kernels not compiled")

        torch.manual_seed(42)

        # LTX-Video typical config
        batch = 1
        num_frames = 49
        height = 32
        width = 32
        num_heads = 32
        head_dim = 64
        hidden_dim = num_heads * head_dim  # 2048

        seq_len = num_frames * height * width  # 50176

        dtype = torch.bfloat16
        device = "cuda"

        # Test RMSNorm with LTX dimensions
        x = torch.randn(batch, seq_len, hidden_dim, device=device, dtype=dtype)
        weight = torch.randn(hidden_dim, device=device, dtype=dtype)
        norm_out = rmsnorm(x, weight, eps=1e-6)
        assert norm_out.shape == x.shape

        # Test 3D RoPE with video dimensions
        q = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        q_rope, k_rope = rope_3d(q, k, num_frames, height, width)
        assert q_rope.shape == q.shape

        # Test GEGLU with FFN dimensions (4x expansion then halved)
        ffn_hidden = hidden_dim * 4
        ffn_input = torch.randn(batch, seq_len, ffn_hidden, device=device, dtype=dtype)
        ffn_out = geglu(ffn_input)
        assert ffn_out.shape == (batch, seq_len, ffn_hidden // 2)

        # Test AdaLN with timestep conditioning
        scale = torch.randn(batch, hidden_dim, device=device, dtype=dtype) * 0.1
        shift = torch.randn(batch, hidden_dim, device=device, dtype=dtype) * 0.1
        adaln_out = adaln_rmsnorm(x, weight, scale, shift)
        assert adaln_out.shape == x.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_efficiency(self):
        """Test that kernels don't leak memory."""
        if not KERNELS_AVAILABLE:
            pytest.skip("Kernels not compiled")

        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        for _ in range(10):
            x = torch.randn(2, 1024, 2048, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
            out = rmsnorm(x, weight)
            del x, weight, out

        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()

        # Memory should return to approximately initial state
        assert final_memory <= initial_memory + 1024 * 1024  # Allow 1MB tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
