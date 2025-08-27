import importlib
from pathlib import Path
import torch

from importlib.metadata import version

__version__ = version("torch-llm-kernels")

from importlib.resources import files
package_files = files("torch_llm_kernels")
for f in package_files.glob('_C*.so'):
    _lib_path = f
    break

if _lib_path.exists():
    torch.ops.load_library(_lib_path)
else:
    print("Warning: CUDA extension for torch-llm-kernels not compiled or found.")

from .ops.swiglu import swiglu

__all__ = ["__version__", "swiglu"]