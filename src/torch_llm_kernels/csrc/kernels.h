#pragma once

#include <torch/extension.h>
#include <tuple> 

torch::Tensor swiglu_forward_cuda(const torch::Tensor& gate, const torch::Tensor& up);

std::tuple<torch::Tensor, torch::Tensor> swiglu_backward_cuda(
    const torch::Tensor& grad_output, const torch::Tensor& gate, const torch::Tensor& up);