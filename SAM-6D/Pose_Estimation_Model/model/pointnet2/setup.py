# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob


CUDA_HOME = os.environ['HOME']
cuda_include_dir = f'{CUDA_HOME}/include'

_ext_src_root = "_ext_src"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

# Get the absolute path of the include directory, knowing its location relative to this setup.py file.
# For some unknown reason, we must provide absolute paths for include directories.
setup_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(setup_dir, _ext_src_root, 'include')

setup(
    name='pointnet2',
    packages = find_packages(),
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            include_dirs = [include_dir, cuda_include_dir],
            extra_compile_args={
                # "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                # "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                "cxx": [],
                "nvcc": ["-O3", 
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]},)
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True)}
)
