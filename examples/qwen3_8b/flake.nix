# Qwen3-8B RMSNorm kernel demo - Nix flake
#
# Usage:
#   nix flake update
#   nix run .#build-and-copy -L
#
# Optional cache:
#   nix run nixpkgs#cachix -- use huggingface

{
  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder";
  };

  outputs = { self, kernel-builder }:
    kernel-builder.lib.genFlakeOutputs {
      path = ./.;
    };
}
