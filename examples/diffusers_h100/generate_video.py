"""
Generate a video using LTX-Video with custom CUDA kernels.
"""

import torch
import sys
import os

# Add kernel module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'torch-ext'))

from diffusers import LTXPipeline, LTXVideoTransformer3DModel
from diffusers.utils import export_to_video
import time

# Import our custom kernels
from ltx_kernels import rmsnorm, rope_3d, geglu, adaln_rmsnorm


def main():
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

    # Enable memory optimizations
    pipe.enable_model_cpu_offload()

    # Video generation parameters
    prompt = "A golden retriever puppy playing in autumn leaves, cinematic lighting, slow motion"
    negative_prompt = "blurry, low quality, distorted"

    # Use smaller dimensions for faster generation
    num_frames = 25  # ~1 second at 24fps
    height = 480
    width = 704
    num_inference_steps = 30

    print(f"\nGeneration settings:")
    print(f"  Prompt: {prompt}")
    print(f"  Frames: {num_frames}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Steps: {num_inference_steps}")

    # Generate video
    print("\nGenerating video...")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    with torch.inference_mode():
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

    gen_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1e9

    print(f"\nGeneration complete!")
    print(f"  Time: {gen_time:.1f}s ({gen_time/num_frames:.2f}s per frame)")
    print(f"  Peak memory: {peak_memory:.2f} GB")

    # Save video
    output_path = "ltx_video_output.mp4"
    export_to_video(output.frames[0], output_path, fps=24)
    print(f"\nVideo saved to: {output_path}")

    # Also save as GIF for easy viewing
    gif_path = "ltx_video_output.gif"
    export_to_video(output.frames[0], gif_path, fps=12)
    print(f"GIF saved to: {gif_path}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
