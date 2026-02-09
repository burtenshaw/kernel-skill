"""
LTX-Video Custom CUDA Kernels
=============================

Optimized CUDA kernels for Lightricks/LTX-Video model on H100 GPUs (sm_90).

Kernels:
- rmsnorm: RMS Layer Normalization with warp reductions
- rope_1d: 1D Rotary Position Embeddings (for text)
- rope_3d: 3D Rotary Position Embeddings (for video: time x height x width)
- geglu: GELU-Gated Linear Unit activation
- swiglu: SiLU-Gated Linear Unit activation
- adaln_rmsnorm: Adaptive Layer Normalization for DiT conditioning

Usage with diffusers:
    from diffusers import LTXPipeline
    from ltx_kernels import rmsnorm, rope_3d, geglu, adaln_rmsnorm

    # Create custom attention processor using optimized kernels
    class OptimizedLTXAttnProcessor:
        def __call__(self, attn, hidden_states, **kwargs):
            # Use custom kernels for normalization, RoPE, etc.
            ...
"""

from typing import Optional, Tuple
import torch

# Import the compiled extension
try:
    from . import _ops as ops
except ImportError:
    # Fallback for development/testing
    ops = None


# =============================================================================
# Register RMSNorm as PyTorch custom op for torch.compile compatibility
# =============================================================================

_CUSTOM_OP_REGISTERED = False

if ops is not None:
    try:
        # Register as custom op (PyTorch 2.1+)
        @torch.library.custom_op("ltx_kernels::rmsnorm_forward", mutates_args=())
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
        # torch.library.custom_op not available
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

    Args:
        input: Input tensor of shape [..., hidden_size]
        weight: Weight tensor of shape [hidden_size]
        eps: Epsilon for numerical stability (default: 1e-6)
        out: Optional output tensor (ignored when using custom op)

    Returns:
        Normalized tensor of same shape as input
    """
    if _CUSTOM_OP_REGISTERED:
        # Use custom op (works with torch.compile)
        return torch.ops.ltx_kernels.rmsnorm_forward(input, weight, eps)
    elif ops is not None:
        # Direct call (won't work with torch.compile)
        if out is None:
            out = torch.empty_like(input)
        ops.rmsnorm_forward(out, input.contiguous(), weight.contiguous(), eps)
        return out
    else:
        # Pure PyTorch fallback
        variance = input.pow(2).mean(dim=-1, keepdim=True)
        return input * torch.rsqrt(variance + eps) * weight


def rmsnorm_residual(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused RMSNorm with residual addition.

    Formula: combined = input + residual
             output = combined * weight / sqrt(mean(combined^2) + eps)

    Args:
        input: Input tensor
        residual: Residual tensor to add
        weight: Normalization weight
        eps: Epsilon for numerical stability
        out: Optional output tensor
        residual_out: Optional tensor to store combined input+residual

    Returns:
        Tuple of (normalized output, combined residual)
    """
    if out is None:
        out = torch.empty_like(input)
    if residual_out is None:
        residual_out = torch.empty_like(input)

    if ops is not None:
        ops.rmsnorm_residual_forward(
            out, residual_out, input.contiguous(), residual.contiguous(),
            weight.contiguous(), eps
        )
    else:
        combined = input + residual
        residual_out.copy_(combined)
        variance = combined.pow(2).mean(dim=-1, keepdim=True)
        out = combined * torch.rsqrt(variance + eps) * weight

    return out, residual_out


def rope_1d(
    query: torch.Tensor,
    key: torch.Tensor,
    theta_base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply 1D Rotary Position Embeddings (for text sequences).

    Args:
        query: Query tensor [batch, seq_len, num_heads, head_dim]
        key: Key tensor [batch, seq_len, num_heads, head_dim]
        theta_base: Base for frequency computation (default: 10000)

    Returns:
        Tuple of (rotated query, rotated key)

    Note:
        Modifies tensors in-place and returns them.
    """
    query = query.contiguous()
    key = key.contiguous()

    if ops is not None:
        ops.rope_1d_forward(query, key, theta_base)
    else:
        # Pure PyTorch fallback
        batch, seq_len, num_heads, head_dim = query.shape
        device = query.device
        dtype = query.dtype

        position = torch.arange(seq_len, device=device, dtype=torch.float32)
        dim_idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
        freqs = 1.0 / (theta_base ** (dim_idx / head_dim))

        angles = position.unsqueeze(1) * freqs.unsqueeze(0)
        cos = torch.cos(angles).to(dtype)
        sin = torch.sin(angles).to(dtype)

        # Apply rotation
        q_even, q_odd = query[..., 0::2], query[..., 1::2]
        k_even, k_odd = key[..., 0::2], key[..., 1::2]

        cos = cos.view(1, seq_len, 1, -1)
        sin = sin.view(1, seq_len, 1, -1)

        query[..., 0::2] = q_even * cos - q_odd * sin
        query[..., 1::2] = q_even * sin + q_odd * cos
        key[..., 0::2] = k_even * cos - k_odd * sin
        key[..., 1::2] = k_even * sin + k_odd * cos

    return query, key


def rope_3d(
    query: torch.Tensor,
    key: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    theta_base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply 3D Rotary Position Embeddings (for video: time x height x width).

    The head dimension is split into three parts for temporal and spatial encoding:
    - First 1/3: temporal position (frame index)
    - Second 1/3: height position (row index)
    - Last 1/3: width position (column index)

    Args:
        query: Query tensor [batch, num_frames*height*width, num_heads, head_dim]
        key: Key tensor [batch, num_frames*height*width, num_heads, head_dim]
        num_frames: Number of video frames
        height: Height of each frame (in patches)
        width: Width of each frame (in patches)
        theta_base: Base for frequency computation

    Returns:
        Tuple of (rotated query, rotated key)

    Note:
        This is the key positional encoding for LTX-Video's temporal coherence.
    """
    query = query.contiguous()
    key = key.contiguous()

    if ops is not None:
        ops.rope_3d_forward(query, key, num_frames, height, width, theta_base)
    else:
        # Pure PyTorch fallback (simplified)
        batch, seq_len, num_heads, head_dim = query.shape
        assert seq_len == num_frames * height * width

        device = query.device
        dtype = query.dtype

        # Split head_dim into 3 equal parts
        rope_dim_t = (head_dim // 3) // 2 * 2
        rope_dim_h = (head_dim // 3) // 2 * 2
        rope_dim_w = head_dim - rope_dim_t - rope_dim_h
        rope_dim_w = rope_dim_w // 2 * 2

        # Create position indices
        t_pos = torch.arange(num_frames, device=device).view(-1, 1, 1).expand(-1, height, width).reshape(-1)
        h_pos = torch.arange(height, device=device).view(1, -1, 1).expand(num_frames, -1, width).reshape(-1)
        w_pos = torch.arange(width, device=device).view(1, 1, -1).expand(num_frames, height, -1).reshape(-1)

        def apply_rope_section(tensor, positions, dim_start, section_dim):
            if section_dim == 0:
                return
            half_dim = section_dim // 2
            dim_idx = torch.arange(0, half_dim, device=device, dtype=torch.float32)
            freqs = 1.0 / (theta_base ** (2.0 * dim_idx / section_dim))
            angles = positions.float().unsqueeze(1) * freqs.unsqueeze(0)
            cos = torch.cos(angles).to(dtype).view(1, seq_len, 1, half_dim)
            sin = torch.sin(angles).to(dtype).view(1, seq_len, 1, half_dim)

            section = tensor[..., dim_start:dim_start + section_dim].clone()
            x1 = section[..., :half_dim]
            x2 = section[..., half_dim:]

            tensor[..., dim_start:dim_start + half_dim] = x1 * cos - x2 * sin
            tensor[..., dim_start + half_dim:dim_start + section_dim] = x1 * sin + x2 * cos

        apply_rope_section(query, t_pos, 0, rope_dim_t)
        apply_rope_section(query, h_pos, rope_dim_t, rope_dim_h)
        apply_rope_section(query, w_pos, rope_dim_t + rope_dim_h, rope_dim_w)

        apply_rope_section(key, t_pos, 0, rope_dim_t)
        apply_rope_section(key, h_pos, rope_dim_t, rope_dim_h)
        apply_rope_section(key, w_pos, rope_dim_t + rope_dim_h, rope_dim_w)

    return query, key


def rope_3d_extended(
    query: torch.Tensor,
    key: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    rope_dim_t: int,
    rope_dim_h: int,
    rope_dim_w: int,
    theta_base_t: float = 10000.0,
    theta_base_h: float = 10000.0,
    theta_base_w: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply 3D RoPE with custom dimension splits and theta bases.

    Args:
        query: Query tensor [batch, num_frames*height*width, num_heads, head_dim]
        key: Key tensor
        num_frames, height, width: Video dimensions
        rope_dim_t, rope_dim_h, rope_dim_w: Dimension allocations for each axis
        theta_base_t, theta_base_h, theta_base_w: Separate theta bases

    Returns:
        Tuple of (rotated query, rotated key)
    """
    query = query.contiguous()
    key = key.contiguous()

    if ops is not None:
        ops.rope_3d_extended_forward(
            query, key, num_frames, height, width,
            rope_dim_t, rope_dim_h, rope_dim_w,
            theta_base_t, theta_base_h, theta_base_w
        )

    return query, key


def geglu(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    approximate: bool = True,
) -> torch.Tensor:
    """
    GELU-Gated Linear Unit activation.

    Formula: output = GELU(gate) * value
    where input is split: [gate, value]

    Args:
        input: Input tensor [..., 2*hidden_dim]
        out: Optional output tensor
        approximate: Use tanh approximation (default: True, faster)

    Returns:
        Output tensor [..., hidden_dim]
    """
    hidden_dim = input.shape[-1] // 2
    out_shape = list(input.shape)
    out_shape[-1] = hidden_dim

    if out is None:
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)

    if ops is not None:
        if approximate:
            ops.geglu_forward(out, input.contiguous())
        else:
            ops.geglu_exact_forward(out, input.contiguous())
    else:
        gate, value = input.chunk(2, dim=-1)
        if approximate:
            out = torch.nn.functional.gelu(gate, approximate='tanh') * value
        else:
            out = torch.nn.functional.gelu(gate) * value

    return out


def swiglu(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    SiLU-Gated Linear Unit activation (SwiGLU).

    Formula: output = SiLU(gate) * value
    where input is split: [gate, value]

    Args:
        input: Input tensor [..., 2*hidden_dim]
        out: Optional output tensor

    Returns:
        Output tensor [..., hidden_dim]
    """
    hidden_dim = input.shape[-1] // 2
    out_shape = list(input.shape)
    out_shape[-1] = hidden_dim

    if out is None:
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)

    if ops is not None:
        ops.swiglu_forward(out, input.contiguous())
    else:
        gate, value = input.chunk(2, dim=-1)
        out = torch.nn.functional.silu(gate) * value

    return out


def adaln_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Adaptive Layer Normalization with RMSNorm base.

    Formula: output = RMSNorm(x) * weight * (1 + scale) + shift

    This is the primary conditioning mechanism for DiT (Diffusion Transformer).
    Scale and shift come from timestep/conditioning MLP.

    Args:
        input: Input tensor [batch, seq_len, hidden_size]
        weight: Normalization weight [hidden_size]
        scale: Adaptive scale from timestep MLP [batch, hidden_size]
        shift: Adaptive shift from timestep MLP [batch, hidden_size]
        eps: Epsilon for numerical stability
        out: Optional output tensor

    Returns:
        Conditioned normalized tensor
    """
    if out is None:
        out = torch.empty_like(input)

    if ops is not None:
        ops.adaln_rmsnorm_forward(
            out, input.contiguous(), weight.contiguous(),
            scale.contiguous(), shift.contiguous(), eps
        )
    else:
        # PyTorch fallback
        variance = input.pow(2).mean(dim=-1, keepdim=True)
        normalized = input * torch.rsqrt(variance + eps)
        # Broadcast scale/shift: [batch, hidden] -> [batch, 1, hidden]
        scale = scale.unsqueeze(1) if scale.dim() == 2 else scale
        shift = shift.unsqueeze(1) if shift.dim() == 2 else shift
        out = normalized * weight * (1 + scale) + shift

    return out


def adaln_layernorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Adaptive Layer Normalization with standard LayerNorm base.

    Args:
        input: Input tensor [batch, seq_len, hidden_size]
        weight: Normalization weight [hidden_size]
        scale: Adaptive scale [batch, hidden_size]
        shift: Adaptive shift [batch, hidden_size]
        bias: Optional LayerNorm bias [hidden_size]
        eps: Epsilon for numerical stability
        out: Optional output tensor

    Returns:
        Conditioned normalized tensor
    """
    if out is None:
        out = torch.empty_like(input)

    if ops is not None:
        ops.adaln_layernorm_forward(
            out, input.contiguous(), weight.contiguous(),
            bias, scale.contiguous(), shift.contiguous(), eps
        )
    else:
        mean = input.mean(dim=-1, keepdim=True)
        variance = input.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (input - mean) * torch.rsqrt(variance + eps)
        result = normalized * weight
        if bias is not None:
            result = result + bias
        scale = scale.unsqueeze(1) if scale.dim() == 2 else scale
        shift = shift.unsqueeze(1) if shift.dim() == 2 else shift
        out = result * (1 + scale) + shift

    return out


def adaln_zero(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    gate: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    AdaLN-Zero: Adaptive LayerNorm with zero-initialized gating.

    Formula: output = (RMSNorm(x) * weight * (1 + scale) + shift) * gate

    The gate is typically zero-initialized, allowing gradual contribution
    during training.

    Args:
        input: Input tensor [batch, seq_len, hidden_size]
        weight: Normalization weight
        scale: Adaptive scale
        shift: Adaptive shift
        gate: Zero-initialized gate [batch, hidden_size]
        eps: Epsilon
        out: Optional output tensor

    Returns:
        Gated conditioned normalized tensor
    """
    if out is None:
        out = torch.empty_like(input)

    if ops is not None:
        ops.adaln_zero_forward(
            out, input.contiguous(), weight.contiguous(),
            scale.contiguous(), shift.contiguous(), gate.contiguous(), eps
        )
    else:
        variance = input.pow(2).mean(dim=-1, keepdim=True)
        normalized = input * torch.rsqrt(variance + eps)
        scale = scale.unsqueeze(1) if scale.dim() == 2 else scale
        shift = shift.unsqueeze(1) if shift.dim() == 2 else shift
        gate = gate.unsqueeze(1) if gate.dim() == 2 else gate
        out = (normalized * weight * (1 + scale) + shift) * gate

    return out


# Version info
__version__ = "0.1.0"
__all__ = [
    "rmsnorm",
    "rmsnorm_residual",
    "rope_1d",
    "rope_3d",
    "rope_3d_extended",
    "geglu",
    "swiglu",
    "adaln_rmsnorm",
    "adaln_layernorm",
    "adaln_zero",
]
