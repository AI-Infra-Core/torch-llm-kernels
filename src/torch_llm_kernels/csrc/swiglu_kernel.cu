#include "kernels.h"
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/Dispatch.h>
#include <ATen/ScalarType.h>
#include <torch/library.h>

namespace torch_llm_kernels {

template <typename scalar_t>
__device__ __forceinline__ float silu_act(scalar_t x_in) {
    float x = static_cast<float>(x_in);
    return x / (1.0f + expf(-x));
}

template <typename scalar_t>
__global__ void swiglu_forward_kernel(
    const scalar_t* __restrict__ gate_ptr,
    const scalar_t* __restrict__ up_ptr,
    scalar_t* __restrict__ output_ptr,
    int num_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float gate_val = static_cast<float>(gate_ptr[idx]);
        float up_val = up_ptr[idx];
        float result = static_cast<scalar_t>(silu_act(gate_val)) * up_val;
        output_ptr[idx] = result;
    }
}

template <typename scalar_t>
__global__ void swiglu_backward_kernel(
    const scalar_t* __restrict__ grad_out_ptr,
    const scalar_t* __restrict__ gate_ptr,
    const scalar_t* __restrict__ up_ptr,
    scalar_t* __restrict__ grad_gate_ptr,
    scalar_t* __restrict__ grad_up_ptr,
    int num_elements) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        const float gate_val_f = static_cast<float>(gate_ptr[idx]);
        const float up_val_f = static_cast<float>(up_ptr[idx]);
        const float grad_out_val_f = static_cast<float>(grad_out_ptr[idx]);

        const float sig = 1.0f / (1.0f + expf(-gate_val_f));
        
        const float d_silu_val = sig * (1.0f + gate_val_f * (1.0f - sig));
        grad_gate_ptr[idx] = static_cast<scalar_t>(grad_out_val_f * up_val_f * d_silu_val);
        grad_up_ptr[idx] = static_cast<scalar_t>(grad_out_val_f * gate_val_f * sig);
    }
}

torch::Tensor swiglu_forward_cuda(const torch::Tensor& gate, const torch::Tensor& up) {
    auto output = torch::empty_like(gate);
    int num_elements = gate.numel();
    if (num_elements == 0) return output;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, gate.scalar_type(), "swiglu_forward_cuda", ([&] {
        const int block_size = 256;
        const int grid_size = (num_elements + block_size - 1) / block_size;
        swiglu_forward_kernel<scalar_t><<<grid_size, block_size>>>(
            gate.data_ptr<scalar_t>(), up.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), num_elements);
    }));
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> swiglu_backward_cuda(
    const torch::Tensor& grad_output, const torch::Tensor& gate, const torch::Tensor& up) {
    
    auto grad_gate = torch::empty_like(gate);
    auto grad_up = torch::empty_like(up);
    int num_elements = grad_output.numel();
    if (num_elements == 0) return std::make_tuple(grad_gate, grad_up);
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, gate.scalar_type(), "swiglu_backward_cuda", ([&] {
        const int block_size = 256;
        const int grid_size = (num_elements + block_size - 1) / block_size;
        swiglu_backward_kernel<scalar_t><<<grid_size, block_size>>>(
            grad_output.data_ptr<scalar_t>(), gate.data_ptr<scalar_t>(), up.data_ptr<scalar_t>(),
            grad_gate.data_ptr<scalar_t>(), grad_up.data_ptr<scalar_t>(), num_elements);
    }));
    return std::make_tuple(grad_gate, grad_up);
}

TORCH_LIBRARY(torch_llm_kernels, m) {
    m.def("swiglu_forward(Tensor gate, Tensor up) -> Tensor");
    m.def("swiglu_backward(Tensor grad_output, Tensor gate, Tensor up) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(torch_llm_kernels, CUDA, m) {
    m.impl("swiglu_forward", &swiglu_forward_cuda);
    m.impl("swiglu_backward", &swiglu_backward_cuda);
}

}