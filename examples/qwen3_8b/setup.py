"""
Build script for Qwen3-8B custom CUDA kernels.

Usage:
    pip install -e .

Or build only:
    python setup.py build_ext --inplace
"""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# CUDA source files
cuda_sources = [
    "kernel_src/rmsnorm.cu",
]

# C++ binding source
cpp_sources = [
    "torch-ext/torch_binding.cpp",
]

# Compiler flags optimized for H100 (sm_90)
extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "-arch=sm_90",  # H100 compute capability
        "-gencode=arch=compute_90,code=sm_90",
        "-lineinfo",
    ],
}

setup(
    name="qwen3-kernels",
    version="0.1.0",
    description="Optimized CUDA kernels for Qwen3-8B on H100 GPUs",
    author="HuggingFace",
    packages=find_packages(where="torch-ext"),
    package_dir={"": "torch-ext"},
    ext_modules=[
        CUDAExtension(
            name="qwen3_kernels._ops",
            sources=cpp_sources + cuda_sources,
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.0"],
    extras_require={
        "transformers": [
            "transformers>=4.45.0",
            "accelerate>=0.20.0",
        ],
    },
)
