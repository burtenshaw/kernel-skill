"""
Generate a video using LTX-Video with custom CUDA kernels.
"""

import torch
import sys
import os
import fire

# Add kernel module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", 'torch-ext'))

from diffusers import LTXPipeline
from diffusers.utils import export_to_video
import time

# Import our custom kernels
from examples.ltx_video_integration import patch_ltx_pipeline


def main(optimize: bool = False, compile: bool = False):
    print("=" * 60)
    print("LTX-Video Generation with Custom CUDA Kernels")
    print("=" * 60)

    device = "cuda"
    dtype = torch.bfloat16

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"Dtype: {dtype}")

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

    # Video generation parameters
    prompt = """
    A woman with long brown hair and light skin smiles at another woman with long blonde hair.
    The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek.
    The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and 
    natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage
    """
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    # Use smaller dimensions for faster generation
    num_frames = 161
    height = 512
    width = 768
    num_inference_steps = 50

    if optimize:
        # LTX-Video VAE uses 8x spatial compression, 1x temporal compression
        # Convert pixel dimensions to latent dimensions
        vae_spatial_compression = 8
        latent_height = height // vae_spatial_compression
        latent_width = width // vae_spatial_compression
        latent_frames = num_frames  # No temporal compression

        print(f"\nPatching pipeline with optimized kernels:")
        print(f"  Pixel dimensions: {num_frames}f × {height}h × {width}w")
        print(f"  Latent dimensions: {latent_frames}f × {latent_height}h × {latent_width}w")

        # Note: We use attention processors rather than replacing entire transformer blocks
        # because it's cleaner and integrates better with diffusers' architecture.
        # The OptimizedLTXVideoTransformerBlock is a standalone implementation for
        # reference/benchmarking, but attention processors are the recommended way
        # to integrate custom kernels with diffusers pipelines.
        patch_ltx_pipeline(
            pipe,
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
        )
    if compile:
        pipe.transformer.compile_repeated_blocks(fullgraph=True)

    print(f"\nGeneration settings:")
    print(f"  Prompt: {prompt}")
    print(f"  Frames: {num_frames}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Steps: {num_inference_steps}")

    # Generate video
    print("\nGenerating video...")
    torch.cuda.reset_peak_memory_stats()

    for _ in range(3):
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=5,
            guidance_scale=7.5,
            generator=torch.Generator(device=device).manual_seed(42),
        )
    
    start_time = time.time()
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=7.5,
        generator=torch.Generator(device=device).manual_seed(42),
    )
    torch.cuda.synchronize()
    gen_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1e9

    print(f"\nGeneration complete!")
    print(f"  Time: {gen_time:.1f}s ({gen_time/num_frames:.2f}s per frame)")
    print(f"  Peak memory: {peak_memory:.2f} GB")

    # Save video
    output_path = f"ltx_video_output_opt@{optimize}_comp@{compile}.mp4"
    export_to_video(output.frames[0], output_path, fps=24)
    print(f"\nVideo saved to: {output_path}")

    # Also save as GIF for easy viewing
    gif_path = f"ltx_video_output_opt@{optimize}_comp@{compile}.gif"
    export_to_video(output.frames[0], gif_path, fps=12)
    print(f"GIF saved to: {gif_path}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    fire.Fire(main)
