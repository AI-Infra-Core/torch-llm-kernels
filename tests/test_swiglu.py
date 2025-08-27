import pytest
import torch
from torch_llm_kernels import swiglu

IS_BF16_SUPPORTED = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
DTYPES = [torch.float32, torch.float16]
if IS_BF16_SUPPORTED:
    DTYPES.append(torch.bfloat16)

def swiglu_pytorch(gate, up):
    return torch.nn.functional.silu(gate.to(torch.float32)).to(gate.dtype) * up

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1, 1024), (16, 2048)])
def test_swiglu_forward(shape, dtype):
    if not torch.cuda.is_available(): pytest.skip("CUDA not available")

    device = "cuda"
    gate_custom = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    up_custom = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    output_custom = swiglu(gate_custom, up_custom)
    output_custom.sum().backward()

    gate_ref = gate_custom.detach().clone().requires_grad_(True)
    up_ref = up_custom.detach().clone().requires_grad_(True)
    output_ref = swiglu_pytorch(gate_ref, up_ref)
    output_ref.sum().backward()

    atol = 1e-5 if dtype == torch.float32 else 1e-2
    rtol = 1e-3 if dtype == torch.float32 else 1e-2
    assert torch.allclose(gate_custom.grad, gate_ref.grad, atol=atol, rtol=rtol)
    assert torch.allclose(up_custom.grad, up_ref.grad, atol=atol, rtol=rtol)

@pytest.mark.parametrize("dtype", DTYPES)
def test_swiglu_compile(dtype):
    if not torch.cuda.is_available(): pytest.skip("CUDA not available")

    device = "cuda"
    shape = (16, 2048)
    gate = torch.randn(shape, device=device, dtype=dtype)
    up = torch.randn(shape, device=device, dtype=dtype)
    
    compiled_swiglu = torch.compile(swiglu, mode="max-autotune")
    output_compiled = compiled_swiglu(gate, up)
    output_ref = swiglu_pytorch(gate, up)
    
    atol = 1e-5 if dtype == torch.float32 else 1e-2
    rtol = 1e-3 if dtype == torch.float32 else 1e-2
    assert torch.allclose(output_compiled, output_ref, atol=atol, rtol=rtol)