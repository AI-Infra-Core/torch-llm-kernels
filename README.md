# torch-llm-kernels

A high-performance CUDA kernel library for Large Language Models (LLMs), designed to be fully compatible with `torch.compile`.

## Features

- **High-Performance Kernels**: Custom-written CUDA kernels designed to maximize GPU throughput.
- **`torch.compile` Compatible**: All operators include the necessary meta implementations for seamless integration with PyTorch 2.x's compile mode.
- **Multi-Precision Support**: Kernels are templated to support FP32, FP16, and BFloat16 data types, essential for modern LLM training and inference.
- **Optimized for Speed**: Uses performance-tuning techniques like fast math intrinsics via the `-use_fast_math` flag.
- **Fully Tested**: Includes a comprehensive test suite using pytest to ensure correctness against native PyTorch implementations and to validate gradients.

## Roadmap

This library aims to implement a suite of the most critical kernels for accelerating LLMs. Contributions are highly welcome!

- ✅ SwiGLU kernel
- ⬜️ RMSNorm Kernel
- ⬜️ Rotary Position Embedding (RoPE) Kernel

## Quick Start

1.  **Installation**
    ```bash
    pip install .
    ```

2.  **Run Tests**
    ```bash
    pip install pytest
    pytest
    ```

## Usage

```python
import torch
from torch_llm_kernels import swiglu

device = "cuda"
gate = torch.randn(16, 1024, device=device, dtype=torch.float16)
up = torch.randn(16, 1024, device=device, dtype=torch.float16)

# Use the custom kernel
output = swiglu(gate, up)

# Compile with torch.compile
compiled_swiglu = torch.compile(swiglu)
compiled_output = compiled_swiglu(gate, up)
```

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you'd like to help add a new kernel or improve an existing one, please feel free to open an issue to discuss it or submit a pull request.

## License

This project is licensed under the MIT License.