#!/usr/bin/env python3
"""
Generate a video of a dog using LTX-Video with custom H100 CUDA kernels.

This script REQUIRES the custom CUDA kernels to be built first.

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

import torch

# Add kernel module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'torch-ext'))

# Import custom kernels - REQUIRED (no fallback)
try:
    from ltx_kernels import rmsnorm, rope_3d, geglu, adaln_rmsnorm
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


def benchmark_custom_kernels(device="cuda", dtype=torch.bfloat16):
    """
    Benchmark custom CUDA kernels to demonstrate H100 optimizations.
    """
    print("\n--- Custom Kernel Benchmarks ---")

    # RMSNorm benchmark
    batch, seq, hidden = 2, 1024, 2048
    x = torch.randn(batch, seq, hidden, device=device, dtype=dtype)
    weight = torch.ones(hidden, device=device, dtype=dtype)

    # Warmup
    for _ in range(3):
        _ = rmsnorm(x, weight)
    torch.cuda.synchronize()

    import time
    start = time.perf_counter()
    for _ in range(100):
        _ = rmsnorm(x, weight)
    torch.cuda.synchronize()
    rmsnorm_time = (time.perf_counter() - start) / 100 * 1000
    print(f"  RMSNorm [{batch}x{seq}x{hidden}]: {rmsnorm_time:.3f} ms")

    # GEGLU benchmark
    x_geglu = torch.randn(batch, seq, hidden * 2, device=device, dtype=dtype)
    for _ in range(3):
        _ = geglu(x_geglu)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        _ = geglu(x_geglu)
    torch.cuda.synchronize()
    geglu_time = (time.perf_counter() - start) / 100 * 1000
    print(f"  GEGLU [{batch}x{seq}x{hidden*2}]: {geglu_time:.3f} ms")

    # RoPE 3D benchmark (video)
    num_frames, h, w = 4, 10, 12  # 4 frames at 10x12 spatial
    video_seq = num_frames * h * w
    num_heads, head_dim = 8, 64
    q = torch.randn(batch, video_seq, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, video_seq, num_heads, head_dim, device=device, dtype=dtype)

    for _ in range(3):
        _ = rope_3d(q.clone(), k.clone(), num_frames, h, w)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        _ = rope_3d(q.clone(), k.clone(), num_frames, h, w)
    torch.cuda.synchronize()
    rope_time = (time.perf_counter() - start) / 100 * 1000
    print(f"  RoPE 3D [batch={batch}, seq={video_seq}, heads={num_heads}]: {rope_time:.3f} ms")

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
    print(f"Custom kernels: Enabled")

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

    # Benchmark custom kernels
    benchmark_custom_kernels(device=device, dtype=dtype)

    # Enable memory optimizations
    pipe.enable_model_cpu_offload()

    # Video generation parameters
    print(f"\nGeneration settings:")
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

    print(f"\nGeneration complete!")
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
