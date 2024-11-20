# setup.py

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='stochastic_ops_cuda',
    ext_modules=[
        CUDAExtension(
            name='stochastic_ops_cuda',
            sources=['stochastic_ops.cu'],
            extra_compile_args={
                'cxx': [],
                'nvcc': [
                    '-gencode=arch=compute_80,code=sm_80',  # Adjust according to your GPU
                    '-lineinfo',
                ],
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
