import os
from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    # 检测是否使用 NPU (Ascend)
    # use_npu = os.getenv('USE_NPU', '0') == '1'
    use_npu =1
    has_cuda = torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1'

    if use_npu:
        # Ascend NPU 模式：只编译 CPU 版本，过滤掉 CUDA 文件
        print('Compiling {} for Ascend NPU (CPU only)'.format(name))
        extension = CppExtension
        define_macros += [('USE_NPU', None)]
        # 过滤掉 .cu 文件
        sources = [s for s in sources if not s.endswith('.cu')]
    elif has_cuda:
        # CUDA 模式
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
            '-gencode=arch=compute_70,code=sm_70',
            '-gencode=arch=compute_75,code=sm_75',
            '-gencode=arch=compute_80,code=sm_80',
            '-gencode=arch=compute_86,code=sm_86',
            '-gencode=arch=compute_120,code=sm_120',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA/NPU'.format(name))
        extension = CppExtension
        # 过滤掉 .cu 文件
        sources = [s for s in sources if not s.endswith('.cu')]

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join('src', *module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if __name__ == '__main__':
    setup(
        name='bevfusion',
        version='1.0.0',
        description='BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird\'s Eye View Representation',
        author='BEVFusion Authors',
        packages=find_packages(where='src'),
        package_dir={'': 'src'},
        ext_modules=[
            # make_cuda_ext(
            #     name='bev_pool_ext',
            #     module='bevfusion.ops.bev_pool',
            #     sources=[
            #         'src/bev_pool.cpp',
            #         'src/bev_pool_cuda.cu',
            #     ],
            # ),
            make_cuda_ext(
                name='voxel_layer',
                module='bevfusion.ops.voxel',
                sources=[
                    'src/voxelization.cpp',
                    'src/scatter_points_cpu.cpp',
                    'src/scatter_points_cuda.cu',
                    'src/voxelization_cpu.cpp',
                    'src/voxelization_cuda.cu',
                ],
            ),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False,
        # install_requires=[
        #     'torch>=1.9.0',
        #     'numpy',
        #     'mmengine',
        #     'mmcv>=2.0.0',
        #     'mmdet>=3.0.0',
        #     'mmdet3d>=1.1.0',
        # ],
    )
