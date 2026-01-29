"""
Example: Integrating Custom CUDA Kernels with LTX-Video Pipeline

This example shows how to use the optimized CUDA kernels with the
Lightricks/LTX-Video model in diffusers.
"""

import torch
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'torch-ext'))

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
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        theta_base: float = 10000.0,
    ):
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.theta_base = theta_base
        self._dims_inferred = False

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
            seq_len = query.shape[1]

            # Auto-infer dimensions if not provided or if they don't match current seq_len
            needs_inference = (
                self.num_frames is None or
                self.height is None or
                self.width is None or
                (self.num_frames * self.height * self.width != seq_len)
            )

            if needs_inference:
                # Try to infer from sequence length
                # Common patterns: 49x32x32, 121x32x32, 161x64x96, etc.
                possible_configs = [
                    # Full video sequences
                    (49, 32, 32),   # Short video, 256x256
                    (121, 32, 32),  # Medium video, 256x256
                    (161, 64, 96),  # Long video, 512x768
                    (49, 64, 96),   # Short video, 512x768
                    (121, 64, 96),  # Medium video, 512x768
                    # Single frame or downsampled (for hierarchical attention)
                    (1, 84, 96),    # Single frame, 672x768
                    (1, 126, 64),   # Single frame, 1008x512
                    (1, 64, 96),    # Single frame, 512x768
                    (7, 24, 48),    # 7 frames, downsampled
                ]

                dims_found = False
                for nf, h, w in possible_configs:
                    if nf * h * w == seq_len:
                        self.num_frames = nf
                        self.height = h
                        self.width = w
                        if not self._dims_inferred:
                            # print(f"Auto-inferred RoPE dimensions: {nf}f x {h}h x {w}w (seq_len={seq_len})")
                            self._dims_inferred = True
                        dims_found = True
                        break

                # If still not found, try to factor seq_len automatically
                if not dims_found:
                    # Try common spatial dimensions with varying frame counts
                    common_spatial = [(32, 32), (64, 96), (96, 64), (84, 96), (126, 64), (24, 48)]
                    for h, w in common_spatial:
                        if seq_len % (h * w) == 0:
                            nf = seq_len // (h * w)
                            self.num_frames = nf
                            self.height = h
                            self.width = w
                            print(f"Auto-factored RoPE dimensions: {nf}f x {h}h x {w}w (seq_len={seq_len})")
                            dims_found = True
                            break

                if not dims_found:
                    raise ValueError(
                        f"Cannot infer video dimensions from seq_len={seq_len}. "
                        f"Please provide num_frames, height, width explicitly to patch_ltx_pipeline(). "
                        f"These should be LATENT dimensions (typically 8x compressed from pixel dimensions)."
                    )

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

        # Prepare attention mask for scaled_dot_product_attention
        # The mask from diffusers may not be in the right format, so we need to handle it carefully
        prepared_mask = None
        if attention_mask is not None:
            # scaled_dot_product_attention expects mask shape: [batch, heads, query_seq, key_seq]
            # or broadcastable to that shape
            expected_mask_shape = (query.shape[0], query.shape[1], query.shape[2], key.shape[2])

            # Check if the mask can broadcast to the expected shape
            try:
                # Try to reshape/broadcast the mask
                if attention_mask.ndim == 2:
                    # [batch, seq] -> [batch, 1, 1, seq]
                    prepared_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                elif attention_mask.ndim == 3:
                    # [batch, query_seq, key_seq] -> [batch, 1, query_seq, key_seq]
                    prepared_mask = attention_mask.unsqueeze(1)
                elif attention_mask.ndim == 4:
                    # Already in the right format
                    prepared_mask = attention_mask
                else:
                    # Unexpected shape, skip mask
                    prepared_mask = None
            except:
                # If any error occurs, just skip the mask
                prepared_mask = None

        # Compute attention (uses Flash Attention if available via PyTorch 2.0+)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=prepared_mask,
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


def patch_ltx_pipeline(
    pipe,
    num_frames: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
):
    """
    Patch an LTX-Video pipeline to use optimized kernels.

    Args:
        pipe: LTXPipeline from diffusers
        num_frames: Number of frames in LATENT space (not pixel space).
                   If None, will be auto-inferred on first forward pass.
        height: Height in LATENT space (not pixels).
               If None, will be auto-inferred on first forward pass.
               For pixel height, divide by VAE spatial compression (typically 8).
        width: Width in LATENT space (not pixels).
              If None, will be auto-inferred on first forward pass.
              For pixel width, divide by VAE spatial compression (typically 8).

    Example:
        # Option 1: Let it auto-infer (recommended)
        pipe = patch_ltx_pipeline(pipe)

        # Option 2: Specify latent dimensions explicitly
        pipe = patch_ltx_pipeline(
            pipe,
            num_frames=161,     # No temporal compression in LTX-Video
            height=512 // 8,    # 64 (8x spatial compression)
            width=768 // 8,     # 96 (8x spatial compression)
        )

    Note:
        LTX-Video's VAE uses 8x spatial compression and 1x temporal compression.
        So for a 161-frame, 512x768 pixel video:
        - Latent frames: 161
        - Latent height: 512 / 8 = 64
        - Latent width: 768 / 8 = 96
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
    if num_frames and height and width:
        print(f"  - 3D RoPE for {num_frames}×{height}×{width} latent video")
    else:
        print(f"  - 3D RoPE with auto-inferred dimensions")
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
