# setup.py

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='copy_stochastic_cuda',
    ext_modules=[
        CUDAExtension(
            name='copy_stochastic_cuda',
            sources=['copy_stochastic_cuda.cpp', 'copy_stochastic_kernel.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
