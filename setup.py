
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="grayscale_cuda",
    ext_modules=[
        CUDAExtension(
            name="grayscale_cuda",
            sources=["grayscale_cuda.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]}
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
        