#!/usr/bin/env python3
"""Run the official kernels benchmark workflow for this demo."""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys


def main(argv: list[str]) -> int:
    root = pathlib.Path(__file__).resolve().parent

    kernels = shutil.which("kernels")
    if kernels is not None:
        cmd = [kernels, "benchmark", str(root), *argv]
    else:
        cmd = ["uvx", "--from", "kernels[benchmark]", "kernels", "benchmark", str(root), *argv]

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
