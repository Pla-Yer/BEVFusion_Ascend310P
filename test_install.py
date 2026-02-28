#!/usr/bin/env python
"""测试 BEVFusion 独立项目是否正确安装"""

import sys


def test_imports():
    """测试基本导入"""
    print("测试基本依赖导入...")
    try:
        import torch
        print(f"  ✓ PyTorch 版本: {torch.__version__}")
        print(f"  ✓ CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA 版本: {torch.version.cuda}")
    except ImportError as e:
        print(f"  ✗ PyTorch 导入失败: {e}")
        return False

    try:
        import mmcv
        print(f"  ✓ mmcv 版本: {mmcv.__version__}")
    except ImportError as e:
        print(f"  ✗ mmcv 导入失败: {e}")
        return False

    try:
        import mmdet
        print(f"  ✓ mmdet 版本: {mmdet.__version__}")
    except ImportError as e:
        print(f"  ✗ mmdet 导入失败: {e}")
        return False

    try:
        import mmdet3d
        print(f"  ✓ mmdet3d 版本: {mmdet3d.__version__}")
    except ImportError as e:
        print(f"  ✗ mmdet3d 导入失败: {e}")
        return False

    return True


def test_bevfusion_import():
    """测试 BEVFusion 模块导入"""
    print("\n测试 BEVFusion 模块导入...")
    try:
        from bevfusion import BEVFusion
        print("  ✓ BEVFusion 类导入成功")
    except ImportError as e:
        print(f"  ✗ BEVFusion 导入失败: {e}")
        return False

    try:
        from bevfusion import TransFusionHead, ConvFuser
        print("  ✓ TransFusionHead, ConvFuser 导入成功")
    except ImportError as e:
        print(f"  ✗ 检测头导入失败: {e}")
        return False

    try:
        from bevfusion.ops import bev_pool, Voxelization
        print("  ✓ bev_pool, Voxelization 导入成功")
    except ImportError as e:
        print(f"  ✗ ops 模块导入失败: {e}")
        return False

    return True


def test_cuda_extensions():
    """测试 CUDA 扩展"""
    print("\n测试 CUDA 扩展...")
    try:
        from bevfusion.ops.bev_pool import bev_pool_ext
        print("  ✓ bev_pool_ext CUDA 扩展加载成功")
    except ImportError as e:
        print(f"  ✗ bev_pool_ext 加载失败: {e}")
        print("    请运行: python setup.py build develop")
        return False

    try:
        from bevfusion.ops.voxel.voxel_layer import hard_voxelize, dynamic_voxelize
        print("  ✓ voxel_layer CUDA 扩展加载成功")
    except ImportError as e:
        print(f"  ✗ voxel_layer 加载失败: {e}")
        print("    请运行: python setup.py build develop")
        return False

    return True


def main():
    print("=" * 50)
    print("BEVFusion 独立项目安装测试")
    print("=" * 50)

    success = True

    if not test_imports():
        success = False

    if not test_bevfusion_import():
        success = False

    if not test_cuda_extensions():
        success = False

    print("\n" + "=" * 50)
    if success:
        print("✓ 所有测试通过！BEVFusion 已正确安装。")
    else:
        print("✗ 部分测试失败，请检查上述错误信息。")
    print("=" * 50)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
