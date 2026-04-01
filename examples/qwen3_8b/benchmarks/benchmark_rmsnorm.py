import torch

from kernels.benchmark import Benchmark

QWEN3_HIDDEN_SIZE = 4096
QWEN3_EPS = 1e-6
QWEN3_DTYPE = torch.bfloat16


def reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = QWEN3_EPS) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


class RMSNormWorkloads(Benchmark):
    seed = 0

    def _setup(self, shape: tuple[int, int, int]) -> None:
        self.x = torch.randn(shape, dtype=QWEN3_DTYPE, device=self.device)
        self.weight = torch.ones(shape[-1], dtype=QWEN3_DTYPE, device=self.device)
        self.out = torch.empty_like(self.x)

    def benchmark_short_prompt(self) -> None:
        self.kernel.rmsnorm(self.out, self.x, self.weight, QWEN3_EPS)

    def setup_short_prompt(self) -> None:
        self._setup((1, 128, QWEN3_HIDDEN_SIZE))

    def verify_short_prompt(self) -> torch.Tensor:
        return reference_rmsnorm(self.x, self.weight, QWEN3_EPS)

    def benchmark_medium_prompt(self) -> None:
        self.kernel.rmsnorm(self.out, self.x, self.weight, QWEN3_EPS)

    def setup_medium_prompt(self) -> None:
        self._setup((1, 512, QWEN3_HIDDEN_SIZE))

    def verify_medium_prompt(self) -> torch.Tensor:
        return reference_rmsnorm(self.x, self.weight, QWEN3_EPS)

    def benchmark_long_prompt(self) -> None:
        self.kernel.rmsnorm(self.out, self.x, self.weight, QWEN3_EPS)

    def setup_long_prompt(self) -> None:
        self._setup((1, 2048, QWEN3_HIDDEN_SIZE))

    def verify_long_prompt(self) -> torch.Tensor:
        return reference_rmsnorm(self.x, self.weight, QWEN3_EPS)

    def benchmark_batch4_prompt(self) -> None:
        self.kernel.rmsnorm(self.out, self.x, self.weight, QWEN3_EPS)

    def setup_batch4_prompt(self) -> None:
        self._setup((4, 512, QWEN3_HIDDEN_SIZE))

    def verify_batch4_prompt(self) -> torch.Tensor:
        return reference_rmsnorm(self.x, self.weight, QWEN3_EPS)

    def benchmark_extended_context(self) -> None:
        self.kernel.rmsnorm(self.out, self.x, self.weight, QWEN3_EPS)

    def setup_extended_context(self) -> None:
        self._setup((1, 8192, QWEN3_HIDDEN_SIZE))

    def verify_extended_context(self) -> torch.Tensor:
        return reference_rmsnorm(self.x, self.weight, QWEN3_EPS)
