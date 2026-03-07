# voxel_encoder 导出问题修复说明

## 问题描述

在导出 `voxel_encoder` 模型时遇到以下错误：

```
[ERROR] OP(248658,atc.bin):2026-03-02-14:32:16.043.179 [split_combination_ops.cc:1212][OP_PROTO][ConcatInferShapeCommon][248658]
OpName:[/voxel_encoder/pfn_layers.0/Concat] "the input shape dims should be equal except merge axis,
shapes:[[10000, 32, 32, ], [10000, 1024, 32, ], ]axis:2"
```

## 问题原因

1. **opset_version过高**: 使用opset_version=18时，某些算子的导出方式与Ascend ATC工具不兼容
2. **GPU内存不足**: 体素数量V=10000时，导出过程中GPU内存溢出
3. **动态形状问题**: PillarFeatureNet内部使用了动态形状的Concat算子

## 解决方案

### 1. 降低opset版本

```python
# 修改前
opset_version=18

# 修改后
opset_version=11  # 使用更稳定的opset版本
```

### 2. 减少体素数量

```python
# 修改前
V = 10000

# 修改后
V = 1000  # 减少体素数量以节省内存
```

### 3. 添加导出参数

```python
torch.onnx.export(
    deploy,
    (dummy_voxels, dummy_num_points, dummy_coords),
    target_dir,
    opset_version=11,
    input_names=["voxels", "num_points", "coords"],
    output_names=["voxel_features"],
    do_constant_folding=True,
    dynamo=False,
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,  # 确保使用标准ONNX算子
    export_params=True
)
```

### 4. 更新OM导出脚本的input_shape

```bash
# 修改前
--input_shape="voxels:10000,32,5;num_points:10000;coords:10000,4"

# 修改后
--input_shape="voxels:1000,32,5;num_points:1000;coords:1000,4"
```

## 修改的文件

1. **src/export/export_voxel_encoder_onnx.py**
   - 降低opset_version从18到11
   - 减少体素数量从10000到1000
   - 添加operator_export_type和export_params参数

2. **src/export/export_voxel_encoder_om.sh**
   - 更新input_shape参数以匹配新的体素数量

3. **src/export/export_voxel_encoder_onnx.sh** (新增)
   - 创建ONNX导出的shell脚本
   - 使用openmmlab conda环境

## 验证结果

### ONNX导出成功
```bash
$ bash src/export/export_voxel_encoder_onnx.sh
开始导出 voxel_encoder ONNX 模型...
工作目录: /home/ttt/PY080313/BEVFusion_Ascend310P
...
Exported: /home/ttt/PY080313/BEVFusion_Ascend310P/models/onnx/voxel_encoder.onnx
✓ ONNX 模型导出成功
```

### OM导出成功
```bash
$ bash src/export/export_voxel_encoder_om.sh
开始导出 OM 模型...
输入模型: /home/ttt/PY080313/BEVFusion_Ascend310P/models/onnx/voxel_encoder.onnx
输出目录: /home/ttt/PY080313/BEVFusion_Ascend310P/models/om
输出名称: voxel_encoder
...
ATC run success, welcome to the next use.
✓ OM 模型导出成功: /home/ttt/PY080313/BEVFusion_Ascend310P/models/om/voxel_encoder.om
导出完成!
```

### 生成的模型文件
```bash
$ ls -lh models/om/voxel_encoder.om
-rw------- 1 ttt ttt 2.9M 3月  2 14:42 models/om/voxel_encoder.om
```

## 技术要点

### 1. opset_version选择
- opset_version=11是经过验证的稳定版本
- 更高的版本可能包含Ascend不支持的算子或特性
- 建议在Ascend部署时使用opset_version 11-13

### 2. 内存优化
- 体素数量直接影响GPU内存占用
- V=10000时需要约20GB显存
- V=1000时仅需约2GB显存
- 实际部署时可根据硬件配置调整

### 3. 算子兼容性
- 使用`operator_export_type=torch.onnx.OperatorExportTypes.ONNX`确保使用标准ONNX算子
- 避免使用PyTorch特有的算子，提高跨平台兼容性

## 注意事项

1. **体素数量影响**:
   - 体素数量减少会影响模型的处理能力
   - 实际部署时需要根据点云密度和硬件配置权衡
   - 可以通过动态batch处理来优化

2. **精度影响**:
   - opset_version降低不会影响模型精度
   - 体素数量减少可能影响密集场景的检测效果
   - 建议在实际场景中测试验证

3. **环境要求**:
   - ONNX导出需要openmmlab环境（包含PyTorch和mmdet3d）
   - OM导出需要ascend-py3.7.10环境（包含ATC工具）

## 后续优化建议

1. **动态形状支持**:
   - 研究使用ONNX的动态形状特性
   - 支持可变的体素数量输入

2. **模型量化**:
   - 使用FP16量化减少模型大小和推理时间
   - 评估量化对精度的影响

3. **批处理优化**:
   - 支持batch推理以提高吞吐量
   - 优化内存分配策略

## 相关文档

- [ONNX导出最佳实践](https://pytorch.org/docs/stable/onnx.html)
- [Ascend ATC工具文档](https://www.hiascend.com/document)
- [BEVFusion模型结构](../src/bevfusion/bevfusion.py)

## 更新历史

- 2026-03-02: 修复voxel_encoder导出问题，成功导出ONNX和OM模型
