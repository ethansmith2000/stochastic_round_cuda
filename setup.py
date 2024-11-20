# setup.py

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import setuptools

with open('README.md') as f:
    README = f.read()

setup(
    author="Placeholder",
    author_email="placeholder@gmail.com",
    name='stochastic_ops',
    license='MIT',
    description='Efficient optimizers',
    version='0.0.1',
    long_description=README,
    url='https://github.com/ethansmith2000/stochastic_round_cuda',
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    long_description_content_type="text/markdown",
    install_requires=['torch'],
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
    ],
    ext_modules=[
        CUDAExtension(
            name='stochastic_ops_cuda',
            sources=['stochastic_round/stochastic_ops.cu'],
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
