"""
Example: Integrating Custom CUDA Kernels with LTX-Video Pipeline

This example shows how to use the optimized CUDA kernels with the
Lightricks/LTX-Video model in diffusers.
"""

import torch
from typing import Optional

# Import custom kernels
from ltx_kernels import (
    rmsnorm,
    rope_3d,
    geglu,
    adaln_rmsnorm,
)


class OptimizedLTXVideoAttnProcessor:
    """
    Custom attention processor for LTX-Video using optimized CUDA kernels.

    This processor replaces the default attention computation with:
    - Custom RMSNorm for query/key/value normalization
    - Custom 3D RoPE for spatio-temporal position encoding
    - Optimized attention (uses Flash Attention when available)
    """

    def __init__(
        self,
        num_frames: int,
        height: int,
        width: int,
        theta_base: float = 10000.0,
    ):
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.theta_base = theta_base

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,  # Timestep embedding
        **kwargs,
    ) -> torch.Tensor:
        """
        Process attention with optimized kernels.

        Args:
            attn: Attention module from diffusers
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            encoder_hidden_states: Cross-attention context (text embeddings)
            attention_mask: Optional attention mask
            temb: Timestep embedding for AdaLN conditioning
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Apply RMSNorm if the attention module has norm layers
        if hasattr(attn, 'norm_q') and attn.norm_q is not None:
            # Use custom RMSNorm
            hidden_states = rmsnorm(
                hidden_states,
                attn.norm_q.weight,
                eps=attn.norm_q.eps if hasattr(attn.norm_q, 'eps') else 1e-6
            )

        # Project to Q, K, V
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        value = attn.to_v(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)

        # Reshape for multi-head attention
        # [batch, seq, hidden] -> [batch, seq, heads, head_dim]
        head_dim = hidden_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        # Apply 3D RoPE for video positional encoding
        # Only apply to self-attention (not cross-attention with text)
        if encoder_hidden_states is None:
            query, key = rope_3d(
                query, key,
                num_frames=self.num_frames,
                height=self.height,
                width=self.width,
                theta_base=self.theta_base,
            )

        # Transpose for attention: [batch, heads, seq, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Compute attention (uses Flash Attention if available via PyTorch 2.0+)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, hidden_dim)

        # Output projection
        attn_output = attn.to_out[0](attn_output)  # Linear
        attn_output = attn.to_out[1](attn_output)  # Dropout

        return attn_output


class OptimizedLTXVideoFFN(torch.nn.Module):
    """
    Optimized Feed-Forward Network for LTX-Video using GEGLU activation.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim

        # GEGLU: project to 2x ffn_dim, then GEGLU halves it
        self.proj_in = torch.nn.Linear(hidden_dim, ffn_dim * 2, dtype=dtype, device=device)
        self.proj_out = torch.nn.Linear(ffn_dim, hidden_dim, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        # Project and apply GEGLU
        x = self.proj_in(x)
        x = geglu(x)  # Uses optimized CUDA kernel
        x = self.proj_out(x)
        return x


class OptimizedLTXVideoTransformerBlock(torch.nn.Module):
    """
    Optimized transformer block for LTX-Video using all custom kernels.

    Structure:
    1. AdaLN + Self-Attention with 3D RoPE
    2. Residual connection
    3. AdaLN + FFN with GEGLU
    4. Residual connection
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Normalization weights
        self.norm1_weight = torch.nn.Parameter(torch.ones(hidden_dim, dtype=dtype, device=device))
        self.norm2_weight = torch.nn.Parameter(torch.ones(hidden_dim, dtype=dtype, device=device))

        # Attention
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.to_q = torch.nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
        self.to_k = torch.nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
        self.to_v = torch.nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
        self.to_out = torch.nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)

        # FFN
        self.ffn = OptimizedLTXVideoFFN(hidden_dim, ffn_dim, dtype, device)

        # Video dimensions for 3D RoPE
        self.num_frames = num_frames
        self.height = height
        self.width = width

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_scale: torch.Tensor,
        timestep_shift: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            timestep_scale: [batch, hidden_dim] from timestep MLP
            timestep_shift: [batch, hidden_dim] from timestep MLP
        """
        batch_size, seq_len, _ = hidden_states.shape

        # ========== Self-Attention Block ==========
        # AdaLN normalization
        normed = adaln_rmsnorm(
            hidden_states,
            self.norm1_weight,
            timestep_scale,
            timestep_shift,
        )

        # Q, K, V projections
        query = self.to_q(normed)
        key = self.to_k(normed)
        value = self.to_v(normed)

        # Reshape for attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply 3D RoPE
        query, key = rope_3d(query, key, self.num_frames, self.height, self.width)

        # Attention computation
        query = query.transpose(1, 2)  # [batch, heads, seq, head_dim]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            dropout_p=0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_output = self.to_out(attn_output)

        # Residual
        hidden_states = hidden_states + attn_output

        # ========== FFN Block ==========
        # AdaLN normalization
        normed = adaln_rmsnorm(
            hidden_states,
            self.norm2_weight,
            timestep_scale,
            timestep_shift,
        )

        # FFN with GEGLU
        ffn_output = self.ffn(normed)

        # Residual
        hidden_states = hidden_states + ffn_output

        return hidden_states


def patch_ltx_pipeline(pipe, num_frames: int, height: int, width: int):
    """
    Patch an LTX-Video pipeline to use optimized kernels.

    Args:
        pipe: LTXPipeline from diffusers
        num_frames: Number of video frames
        height: Height in patches
        width: Width in patches
    """
    # Create optimized attention processor
    attn_processor = OptimizedLTXVideoAttnProcessor(
        num_frames=num_frames,
        height=height,
        width=width,
    )

    # Set for all attention layers
    attn_processors = {}
    for name in pipe.transformer.attn_processors.keys():
        attn_processors[name] = attn_processor

    pipe.transformer.set_attn_processor(attn_processors)

    print(f"Patched LTX-Video pipeline with optimized kernels")
    print(f"  - 3D RoPE for {num_frames}×{height}×{width} video")
    print(f"  - Custom RMSNorm with warp reductions")
    print(f"  - GEGLU with tanh approximation")

    return pipe


# Example usage
if __name__ == "__main__":
    print("LTX-Video Custom Kernel Integration Example")
    print("=" * 50)

    # Test the optimized transformer block
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # LTX-Video typical dimensions
    batch_size = 1
    num_frames = 49
    height = 32
    width = 32
    seq_len = num_frames * height * width  # 50176
    hidden_dim = 2048
    num_heads = 32
    ffn_dim = hidden_dim * 4

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Video: {num_frames} frames × {height}×{width}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num heads: {num_heads}")

    if device == "cuda":
        # Create optimized block
        block = OptimizedLTXVideoTransformerBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_frames=num_frames,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
        )

        # Test inputs
        x = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype, device=device)
        scale = torch.randn(batch_size, hidden_dim, dtype=dtype, device=device) * 0.1
        shift = torch.randn(batch_size, hidden_dim, dtype=dtype, device=device) * 0.1

        print(f"\nRunning forward pass...")

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = block(x, scale, shift)

        # Benchmark
        torch.cuda.synchronize()
        import time
        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(10):
                out = block(x, scale, shift)
                torch.cuda.synchronize()

        elapsed = (time.perf_counter() - start) / 10
        print(f"Average forward pass time: {elapsed*1000:.2f} ms")
        print(f"Output shape: {out.shape}")

        # Memory usage
        torch.cuda.empty_cache()
        print(f"\nPeak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    else:
        print("\nCUDA not available, skipping benchmark")

    print("\n" + "=" * 50)
    print("To use with diffusers LTXPipeline:")
    print("""
    from diffusers import LTXPipeline
    from ltx_kernels.integration import patch_ltx_pipeline

    pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video")
    pipe = patch_ltx_pipeline(pipe, num_frames=49, height=32, width=32)

    video = pipe(prompt="A cat walking", num_frames=49).frames
    """)
