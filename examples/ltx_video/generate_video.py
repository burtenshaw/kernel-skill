#!/usr/bin/env python3
"""
Generate a video of a dog using LTX-Video with custom H100 CUDA kernels.

This script REQUIRES the custom CUDA kernels to be built first.
The kernels are injected into the diffusers pipeline for optimized inference.

Requirements:
    pip install diffusers transformers accelerate torch

Build kernels first:
    # Using Nix (recommended)
    nix flake update && nix run .#build-and-copy -L

    # Or using pip/uv
    uv pip install -e .

Usage:
    python generate_video.py
    python generate_video.py --prompt "A golden retriever running on the beach"
    python generate_video.py --num-frames 49 --height 480 --width 704
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Add kernel module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'torch-ext'))

# Import custom kernels - REQUIRED (no fallback)
try:
    from ltx_kernels import rmsnorm
    # Note: LTX-Video uses GELU (not GEGLU), so we only need rmsnorm
    KERNELS_AVAILABLE = True
    print("Custom CUDA kernels loaded successfully!")
except ImportError as e:
    print("=" * 60)
    print("ERROR: Custom CUDA kernels not found!")
    print("=" * 60)
    print(f"\nImport error: {e}")
    print("\nPlease build the kernels first:")
    print("  # Using Nix (recommended)")
    print("  nix flake update && nix run .#build-and-copy -L")
    print("\n  # Or using pip/uv")
    print("  uv pip install -e .")
    print("=" * 60)
    sys.exit(1)

from diffusers import LTXPipeline
from diffusers.utils import export_to_video


# =============================================================================
# Custom Attention Processor with Optimized Kernels
# =============================================================================

class OptimizedLTXVideoAttnProcessor:
    """
    Optimized attention processor for LTX-Video using custom H100 CUDA kernels.

    Replaces the default RMSNorm operations with our fused CUDA implementations
    for better performance on H100 GPUs.
    """

    def __init__(self):
        if not torch.cuda.is_available():
            raise ValueError("CUDA is required for optimized attention processor")

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        from diffusers.models.transformers.transformer_ltx import apply_rotary_emb
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Q, K, V projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Use custom RMSNorm kernel for Q/K normalization
        # The attn.norm_q and attn.norm_k are torch.nn.RMSNorm modules
        query = rmsnorm(query, attn.norm_q.weight, eps=attn.norm_q.eps)
        key = rmsnorm(key, attn.norm_k.weight, eps=attn.norm_k.eps)

        # Apply rotary embeddings
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Reshape for multi-head attention
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Dispatch attention (uses PyTorch SDPA or other backends)
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=getattr(self, '_attention_backend', None),
            parallel_config=getattr(self, '_parallel_config', None),
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def patch_rmsnorm_modules(model: nn.Module) -> int:
    """
    Monkey-patch all RMSNorm modules in the model to use our custom CUDA kernel.

    Handles both torch.nn.RMSNorm and diffusers.models.normalization.RMSNorm.
    Also handles modules with elementwise_affine=False (no weight parameter).

    Returns the number of modules patched.
    """
    patched_count = 0

    for name, module in model.named_modules():
        # Check for any RMSNorm variant (torch.nn.RMSNorm or diffusers RMSNorm)
        module_class_name = type(module).__name__
        if module_class_name == 'RMSNorm':
            eps = getattr(module, 'eps', 1e-6)
            has_weight = hasattr(module, 'weight') and module.weight is not None

            if has_weight:
                # Module has learnable weight
                def make_patched_forward_with_weight(mod, epsilon):
                    def patched_forward(x):
                        return rmsnorm(x, mod.weight, eps=epsilon)
                    return patched_forward
                module.forward = make_patched_forward_with_weight(module, eps)
            else:
                # No weight (elementwise_affine=False) - use ones
                def make_patched_forward_no_weight(epsilon):
                    def patched_forward(x):
                        # Create weight of ones on the same device/dtype as input
                        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
                        return rmsnorm(x, weight, eps=epsilon)
                    return patched_forward
                module.forward = make_patched_forward_no_weight(eps)

            patched_count += 1
            # Debug: uncomment to see which modules are patched
            # print(f"  Patched RMSNorm: {name} (has_weight={has_weight})")

    return patched_count


def inject_optimized_kernels(pipe) -> dict:
    """
    Inject optimized CUDA kernels into the LTX-Video pipeline.

    This patches:
    1. Attention processors to use custom RMSNorm for Q/K normalization
    2. All RMSNorm modules in the transformer (norm1, norm2, norm_q, norm_k)

    Note: LTX-Video uses GELU (not GEGLU) for activations, so GEGLU kernel is not used.

    Returns a dict with counts of patched modules.
    """
    stats = {
        'attention_processors': 0,
        'rmsnorm_modules': 0,
    }

    if not hasattr(pipe, 'transformer'):
        print("  WARNING: Pipeline has no 'transformer' attribute!")
        return stats

    transformer = pipe.transformer

    # 1. Replace attention processors with optimized version
    # This handles norm_q and norm_k in attention modules
    for name, module in transformer.named_modules():
        if hasattr(module, 'set_processor') and hasattr(module, 'processor'):
            module.set_processor(OptimizedLTXVideoAttnProcessor())
            stats['attention_processors'] += 1
            # Debug: uncomment to see which processors are replaced
            # print(f"  Replaced processor: {name}")

    # 2. Patch RMSNorm modules (norm1, norm2 in transformer blocks)
    stats['rmsnorm_modules'] = patch_rmsnorm_modules(transformer)

    return stats


def benchmark_custom_kernels(device="cuda", dtype=torch.bfloat16):
    """
    Benchmark custom CUDA kernels to demonstrate H100 optimizations.
    """
    print("\n--- Custom Kernel Benchmarks ---")

    # RMSNorm benchmark (this is the primary kernel used in LTX-Video)
    batch, seq, hidden = 2, 1024, 2048
    x = torch.randn(batch, seq, hidden, device=device, dtype=dtype)
    weight = torch.ones(hidden, device=device, dtype=dtype)

    # Warmup
    for _ in range(3):
        _ = rmsnorm(x, weight)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        _ = rmsnorm(x, weight)
    torch.cuda.synchronize()
    rmsnorm_time = (time.perf_counter() - start) / 100 * 1000
    print(f"  RMSNorm [{batch}x{seq}x{hidden}]: {rmsnorm_time:.3f} ms")

    print("--- End Benchmarks ---\n")


def generate_dog_video(
    prompt: str = "A happy golden retriever dog running through a sunny park, wagging its tail, cinematic lighting, 4K quality",
    negative_prompt: str = "blurry, low quality, distorted, watermark, text",
    num_frames: int = 25,
    height: int = 480,
    width: int = 704,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42,
    output_path: str = "dog_video.mp4",
):
    """
    Generate a video of a dog using LTX-Video with custom H100 CUDA kernels.

    Args:
        prompt: Text description of the video to generate
        negative_prompt: Things to avoid in generation
        num_frames: Number of video frames (more = longer video)
        height: Video height in pixels
        width: Video width in pixels
        num_inference_steps: Denoising steps (more = higher quality, slower)
        guidance_scale: Classifier-free guidance strength
        seed: Random seed for reproducibility
        output_path: Where to save the video
    """
    print("=" * 60)
    print("LTX-Video Generation with Custom H100 CUDA Kernels")
    print("=" * 60)

    device = "cuda"
    dtype = torch.bfloat16  # BF16 is preferred for training/inference stability

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"Dtype: {dtype}")
    print("Custom kernels: Enabled")

    # Load the pipeline
    print("\nLoading LTX-Video pipeline...")
    start_time = time.time()

    pipe = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=dtype,
    )
    pipe.to(device)

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.1f}s")

    # Inject optimized CUDA kernels into the pipeline
    print("\nInjecting optimized CUDA kernels...")
    injection_stats = inject_optimized_kernels(pipe)
    print(f"  Attention processors replaced: {injection_stats['attention_processors']}")
    print(f"  RMSNorm modules patched: {injection_stats['rmsnorm_modules']}")

    # Benchmark custom kernels
    benchmark_custom_kernels(device=device, dtype=dtype)

    # Enable memory optimizations
    pipe.enable_model_cpu_offload()

    # Video generation parameters
    print("\nGeneration settings:")
    print(f"  Prompt: {prompt}")
    print(f"  Frames: {num_frames}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Steps: {num_inference_steps}")
    print(f"  Guidance scale: {guidance_scale}")
    print(f"  Seed: {seed}")

    # Generate video
    print("\nGenerating video...")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    gen_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1e9

    print("\nGeneration complete!")
    print(f"  Time: {gen_time:.1f}s ({gen_time/num_frames:.2f}s per frame)")
    print(f"  Peak memory: {peak_memory:.2f} GB")

    # Save video
    export_to_video(output.frames[0], output_path, fps=24)
    print(f"\nVideo saved to: {output_path}")

    # Also save as GIF for easy viewing
    gif_path = output_path.replace('.mp4', '.gif')
    export_to_video(output.frames[0], gif_path, fps=12)
    print(f"GIF saved to: {gif_path}")

    print("\n" + "=" * 60)
    print("Done!")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos using LTX-Video with custom H100 kernels"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A happy golden retriever dog running through a sunny park, wagging its tail, cinematic lighting, 4K quality",
        help="Text prompt describing the video to generate"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, low quality, distorted, watermark, text",
        help="Things to avoid in generation"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=25,
        help="Number of frames to generate (default: 25, ~1 second at 24fps)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height in pixels (default: 480)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=704,
        help="Video width in pixels (default: 704)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of denoising steps (default: 30)"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale (default: 7.5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dog_video.mp4",
        help="Output video path (default: dog_video.mp4)"
    )

    args = parser.parse_args()

    generate_dog_video(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
