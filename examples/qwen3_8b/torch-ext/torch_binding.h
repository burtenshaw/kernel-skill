#pragma once

#include <torch/torch.h>

void rmsnorm(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    float eps
);
