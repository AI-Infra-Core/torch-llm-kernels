import torch

@torch.library.register_fake("torch_llm_kernels::swiglu_forward")
def _(gate, up):
    torch._check(gate.shape == up.shape)
    torch._check(gate.dtype == up.dtype)
    torch._check(gate.device == up.device)
    return torch.empty_like(gate)

class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        gate = gate.contiguous()
        up = up.contiguous()
        output = torch.ops.torch_llm_kernels.swiglu_forward(gate, up)
        ctx.save_for_backward(gate, up)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_c = grad_output.contiguous()
        gate, up = ctx.saved_tensors
        
        grad_gate, grad_up = torch.ops.torch_llm_kernels.swiglu_backward(
            grad_output_c, gate, up
        )
        return grad_gate, grad_up

def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Applies the SwiGLU activation function using a custom CUDA kernel.
    Supports float32, float16, and bfloat16 inputs.
    """
    return SwiGLUFunction.apply(gate, up)