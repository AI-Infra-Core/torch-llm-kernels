import os, glob, sys
from setuptools import setup

library_name = "torch_llm_kernels"

needs_ext_modules = 'build_ext' in sys.argv

def get_extensions():
    import torch
    from torch.utils.cpp_extension import CUDAExtension
    debug_mode = os.getenv("DEBUG", "0") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    extensions_dir = os.path.join("src", library_name, "csrc")

    include_dirs = [
        extensions_dir
    ]

    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))
    cuda_sources = list(glob.glob(os.path.join(extensions_dir, "*.cu")))

    sources += cuda_sources

    ext_modules = [
        CUDAExtension(
            f"{library_name}._C",
            sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=True,
        )
    ]

    return ext_modules

if needs_ext_modules:
    from torch.utils.cpp_extension import BuildExtension
    ext_modules = get_extensions()
    cmdclass = {"build_ext": BuildExtension}
else:
    ext_modules = []
    cmdclass = {}

setup(
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
