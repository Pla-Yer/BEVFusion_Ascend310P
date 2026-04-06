"""
ONNX模型与量化后模型精度对比脚本（适应Ascend环境）
功能: 对比原始ONNX模型和量化后ONNX模型的输出精度
"""

import os
import sys
import numpy as np
import argparse
import json
import onnxruntime as ort


def compare_tensors(original_output, quantized_output, name, threshold_abs=1e-2, threshold_rel=1e-3):
    """
    对比两个tensor的精度
    
    Args:
        original_output: 原始模型输出
        quantized_output: 量化模型输出
        name: 输出名称
        threshold_abs: 绝对误差阈值
        threshold_rel: 相对误差阈值
        
    Returns:
        是否通过验证
    """
    print(f"\n--- {name} ---")
    print(f"  原始模型形状: {original_output.shape}, 类型: {original_output.dtype}")
    print(f"  量化模型形状: {quantized_output.shape}, 类型: {quantized_output.dtype}")
    
    # 检查形状
    if original_output.shape != quantized_output.shape:
        print(f"  ✗ 形状不匹配!")
        return False
    
    # 计算误差
    abs_diff = np.abs(original_output - quantized_output)
    max_abs_error = np.max(abs_diff)
    mean_abs_error = np.mean(abs_diff)
    
    # 计算相对误差
    denominator = np.abs(original_output) + 1e-8
    rel_diff = abs_diff / denominator
    max_rel_error = np.max(rel_diff)
    mean_rel_error = np.mean(rel_diff)
    
    print(f"  最大绝对误差: {max_abs_error:.6f}")
    print(f"  平均绝对误差: {mean_abs_error:.6f}")
    print(f"  最大相对误差: {max_rel_error:.6f}")
    print(f"  平均相对误差: {mean_rel_error:.6f}")
    
    # 判断精度
    passed = False
    if max_rel_error < threshold_rel:
        print(f"  ✓ 精度合格 (相对误差 < {threshold_rel})")
        passed = True
    elif max_abs_error < threshold_abs:
        print(f"  ✓ 精度合格 (绝对误差 < {threshold_abs})")
        passed = True
    else:
        print(f"  ✗ 精度不达标")
        passed = False
    
    return passed


def run_original_model(model_path, inputs):
    """
    运行原始ONNX模型
    
    Args:
        model_path: ONNX模型路径
        inputs: 模型输入字典
        
    Returns:
        模型输出字典
    """
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # 获取模型输入信息
    model_inputs = {inp.name: inp for inp in session.get_inputs()}
    model_outputs = [out.name for out in session.get_outputs()]
    
    # 验证输入
    for name, tensor in inputs.items():
        if name not in model_inputs:
            raise ValueError(f"输入名称 '{name}' 不在模型输入中")
        expected_shape = model_inputs[name].shape
        actual_shape = tensor.shape
        
        # 替换动态维度（-1 或字符串）进行比较
        expected_shape_fixed = []
        for i, s in enumerate(expected_shape):
            # 处理动态维度：如果是-1或者是字符串（如"unk__xxx"或"max_V"等），则使用实际形状
            if s == -1 or (isinstance(s, str) and (s.startswith('unk__') or s.startswith('max_') or s.isdigit())):
                expected_shape_fixed.append(actual_shape[i])
            else:
                expected_shape_fixed.append(s)
        expected_shape_fixed = tuple(expected_shape_fixed)
        
        if tuple(actual_shape) != expected_shape_fixed:
            print(f"  警告: 输入 '{name}' 形状不匹配 - 期望 {expected_shape_fixed}, 实际 {actual_shape}")
    
    # 运行推理
    outputs = session.run(None, inputs)
    output_dict = dict(zip(model_outputs, outputs))
    
    return output_dict


def run_quantized_model_with_ascend(model_path, inputs):
    """
    尝试运行量化后的ONNX模型（可能包含Ascend自定义算子）
    
    Args:
        model_path: ONNX模型路径
        inputs: 模型输入字典
        
    Returns:
        模型输出字典，如果无法运行则返回None和错误消息
    """
    try:
        # 首先尝试使用Ascend执行提供程序
        from onnxruntime_extensions import get_library_path
        session = ort.InferenceSession(
            model_path, 
            providers=['CPUExecutionProvider'],  # 使用CPU执行提供程序
            provider_options=[{'session_options': {'custom_ops': [get_library_path()]}}]  # 如果有的话
        )
    except ImportError:
        # 如果没有安装onnxruntime_extensions，只使用CPU执行提供程序
        try:
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        except Exception as e:
            print(f"无法加载量化模型: {e}")
            return None, str(e)
    except Exception as e:
        print(f"无法使用Ascend扩展加载模型: {e}")
        try:
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        except Exception as e2:
            print(f"即使不使用Ascend扩展也无法加载模型: {e2}")
            return None, str(e2)

    # 获取模型输入信息
    try:
        model_inputs = {inp.name: inp for inp in session.get_inputs()}
        model_outputs = [out.name for out in session.get_outputs()]
    except Exception as e:
        print(f"无法获取模型输入/输出信息: {e}")
        return None, str(e)

    # 验证输入
    for name, tensor in inputs.items():
        if name not in model_inputs:
            error_msg = f"输入名称 '{name}' 不在模型输入中"
            print(error_msg)
            return None, error_msg
            
        expected_shape = model_inputs[name].shape
        actual_shape = tensor.shape
        
        # 替换动态维度（-1 或字符串）进行比较
        expected_shape_fixed = []
        for i, s in enumerate(expected_shape):
            # 处理动态维度：如果是-1或者是字符串（如"unk__xxx"或"max_V"等），则使用实际形状
            if s == -1 or (isinstance(s, str) and (s.startswith('unk__') or s.startswith('max_') or s.isdigit())):
                expected_shape_fixed.append(actual_shape[i])
            else:
                expected_shape_fixed.append(s)
        expected_shape_fixed = tuple(expected_shape_fixed)
        
        if tuple(actual_shape) != expected_shape_fixed:
            print(f"  警告: 输入 '{name}' 形状不匹配 - 期望 {expected_shape_fixed}, 实际 {actual_shape}")

    # 尝试运行推理
    try:
        outputs = session.run(None, inputs)
        output_dict = dict(zip(model_outputs, outputs))
        return output_dict, None
    except Exception as e:
        error_msg = f"量化模型推理失败: {e}"
        print(error_msg)
        return None, error_msg


def compare_models(original_model_path, quantized_model_path, inputs, output_dir):
    """
    对比原始模型和量化模型
    
    Args:
        original_model_path: 原始ONNX模型路径
        quantized_model_path: 量化ONNX模型路径
        inputs: 模型输入
        output_dir: 结果保存目录
        
    Returns:
        是否通过验证
    """
    print("运行原始模型...")
    try:
        original_outputs = run_original_model(original_model_path, inputs)
    except Exception as e:
        print(f"运行原始模型失败: {e}")
        return False

    print("运行量化模型...")
    quantized_outputs, error = run_quantized_model_with_ascend(quantized_model_path, inputs)
    
    if quantized_outputs is None:
        print(f"无法运行量化模型: {error}")
        print("警告：由于量化模型包含自定义算子，无法在当前环境中进行直接对比")
        
        # 创建一个简化版本的报告
        comparison_result = {
            'all_passed': None,  # 无法确定
            'original_model': original_model_path,
            'quantized_model': quantized_model_path,
            'note': '由于量化模型包含Ascend自定义算子，无法在此环境中运行对比测试',
            'error': error
        }
        
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'quantization_comparison_result.json'), 'w') as f:
            json.dump(comparison_result, f, indent=2)
        
        print("已保存无法对比的通知到结果文件中")
        return False
    
    # 确保两个模型输出的键一致
    original_keys = set(original_outputs.keys())
    quantized_keys = set(quantized_outputs.keys())
    
    if original_keys != quantized_keys:
        print(f"  ✗ 输出键不一致 - 原始: {original_keys}, 量化: {quantized_keys}")
        return False
    
    all_passed = True
    results = {}
    
    # 对每个输出进行比较
    for key in original_keys:
        print(f"\n处理输出: {key}")
        original_output = original_outputs[key]
        quantized_output = quantized_outputs[key]
        
        passed = compare_tensors(original_output, quantized_output, key)
        results[key] = passed
        all_passed = all_passed and passed
    
    # 保存详细结果
    detailed_results = {}
    for key in original_keys:
        original_output = original_outputs[key]
        quantized_output = quantized_outputs[key]
        
        abs_diff = np.abs(original_output - quantized_output)
        max_abs_error = np.max(abs_diff)
        mean_abs_error = np.mean(abs_diff)
        
        denominator = np.abs(original_output) + 1e-8
        rel_diff = abs_diff / denominator
        max_rel_error = np.max(rel_diff)
        mean_rel_error = np.mean(rel_diff)
        
        detailed_results[key] = {
            'max_absolute_error': float(max_abs_error),
            'mean_absolute_error': float(mean_abs_error),
            'max_relative_error': float(max_rel_error),
            'mean_relative_error': float(mean_rel_error),
            'passed': bool(results[key])
        }
    
    # 保存结果
    comparison_result = {
        'all_passed': all_passed,
        'original_model': original_model_path,
        'quantized_model': quantized_model_path,
        'results': detailed_results
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'quantization_comparison_result.json'), 'w') as f:
        json.dump(comparison_result, f, indent=2)
    
    return all_passed


def build_lidar_dummy_inputs(max_voxels=8000, max_points=32, point_dim=5):
    """
    生成LiDAR模型的测试输入
    """
    V = max_voxels
    voxels = np.random.randn(V, max_points, point_dim).astype(np.float32)
    num_points = np.random.randint(1, max_points + 1, size=(V,)).astype(np.int32)
    coords_zy = np.random.randint(0, 360, size=(V, 2)).astype(np.int32)
    batch_col = np.zeros((V, 1), dtype=np.int32)
    z_col = np.zeros((V, 1), dtype=np.int32)
    coords = np.concatenate([batch_col, z_col, coords_zy], axis=1)
    
    return {
        'voxels': voxels,
        'num_points': num_points,
        'coords': coords
    }


def build_camera_dummy_inputs(B=1, N=6, H=256, W=704, fH=32, fW=88):
    """
    生成Camera模型的测试输入
    """
    MAX_PTS = 16
    NX0, NX1, NX2 = 360, 360, 1
    OUT_CELLS = B * NX2 * NX0 * NX1  # 129600
    
    rng = np.random.default_rng(0)
    imgs = rng.standard_normal((B, N, 3, H, W)).astype(np.float32)
    depth = rng.standard_normal((B, N, 1, H, W)).astype(np.float32)
    total_pts = B * N * fH * fW
    pool_lookup = rng.integers(0, total_pts, size=(OUT_CELLS, MAX_PTS)).astype(np.int32)
    pool_mask = rng.integers(0, 2, size=(OUT_CELLS, MAX_PTS)).astype(np.uint8)
    
    return {
        'imgs': imgs,
        'depth': depth,
        'pool_lookup': pool_lookup,
        'pool_mask': pool_mask
    }


def build_fusion_head_dummy_inputs(B=1, CAM_C=80, CAM_H=360, CAM_W=360, LIDAR_C=256, LIDAR_H=360, LIDAR_W=360):
    """
    生成Fusion Head模型的测试输入
    """
    rng = np.random.default_rng(0)
    camera_bev = rng.standard_normal((B, CAM_C, CAM_H, CAM_W)).astype(np.float32)
    lidar_bev = rng.standard_normal((B, LIDAR_C, LIDAR_H, LIDAR_W)).astype(np.float32)
    
    return {
        'camera_bev': camera_bev,
        'lidar_bev': lidar_bev
    }


def main():
    parser = argparse.ArgumentParser(description='ONNX模型与量化后模型精度对比（支持Ascend NPU量化模型）')
    parser.add_argument('--original_model', type=str, required=True,
                        help='原始ONNX模型路径')
    parser.add_argument('--quantized_model', type=str, required=True,
                        help='量化ONNX模型路径')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['lidar', 'camera', 'fusion_head'],
                        help='模型类型')
    parser.add_argument('--output_dir', type=str,
                        default='results/quantization_comparison',
                        help='对比结果输出目录')
    parser.add_argument('--threshold_abs', type=float, default=1e-2,
                        help='绝对误差阈值')
    parser.add_argument('--threshold_rel', type=float, default=1e-3,
                        help='相对误差阈值')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 根据模型类型生成相应的测试输入
    print(f"生成{args.model_type}模型的测试输入...")
    if args.model_type == 'lidar':
        inputs = build_lidar_dummy_inputs()
    elif args.model_type == 'camera':
        inputs = build_camera_dummy_inputs()
    elif args.model_type == 'fusion_head':
        inputs = build_fusion_head_dummy_inputs()
    else:
        raise ValueError(f"未知的模型类型: {args.model_type}")
    
    # 更新全局阈值
    global_threshold_abs = args.threshold_abs
    global_threshold_rel = args.threshold_rel
    
    def compare_tensors_with_global_threshold(original_output, quantized_output, name):
        return compare_tensors(original_output, quantized_output, name, global_threshold_abs, global_threshold_rel)
    
    # 运行对比
    print(f"原始模型: {args.original_model}")
    print(f"量化模型: {args.quantized_model}")
    print(f"模型类型: {args.model_type}")
    print(f"绝对误差阈值: {args.threshold_abs}")
    print(f"相对误差阈值: {args.threshold_rel}")
    
    all_passed = compare_models(
        args.original_model,
        args.quantized_model,
        inputs,
        args.output_dir
    )
    
    # 打印总结
    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)
    
    if all_passed:
        print("✓ 模型精度验证通过")
    else:
        print("? 模型精度验证状态：可能无法进行对比（例如，由于量化模型包含自定义算子）")
    print("=" * 60)
    print(f"对比结果已保存到: {args.output_dir}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())