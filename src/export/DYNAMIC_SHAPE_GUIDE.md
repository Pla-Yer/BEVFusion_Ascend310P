# VoxelEncoder 动态Shape导出说明

## 概述

本文档说明如何为VoxelEncoder模型导出支持动态体素数量的ONNX和OM模型，配置3个动态档位：6000、8000、10000。

## 背景

在实际应用中，点云的体素数量是动态变化的：
- 通常情况：约6000个体素
- 中等密度：约8000个体素  
- 高密度场景：约10000个体素

为了支持这种动态变化，需要导出支持动态shape的模型。

## 解决方案

### 1. ONNX导出 - 支持动态轴

修改 `src/export/export_voxel_encoder_onnx.py`，添加动态轴配置：

```python
# 定义动态轴，支持可变的体素数量
dynamic_axes = {
    'voxels': {0: 'V'},        # 第0维（体素数量）是动态的
    'num_points': {0: 'V'},    # 第0维（体素数量）是动态的
    'coords': {0: 'V'},        # 第0维（体素数量）是动态的
    'voxel_features': {0: 'V'} # 输出的第0维也是动态的
}

# 使用更稳定的导出参数，支持动态shape
torch.onnx.export(
    deploy,
    (dummy_voxels, dummy_num_points, dummy_coords),
    target_dir,
    opset_version=11,
    input_names=["voxels", "num_points", "coords"],
    output_names=["voxel_features"],
    dynamic_axes=dynamic_axes,  # 添加动态轴配置
    do_constant_folding=True,
    dynamo=False,
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    export_params=True
)
```

**关键点**：
- 使用`dynamic_axes`参数指定哪些维度是动态的
- 所有输入和输出的第0维（体素数量V）都标记为动态
- 使用最大体素数量（10000）进行导出，确保模型能处理最大情况

### 2. OM导出 - 配置动态档位

修改 `src/export/export_voxel_encoder_om.sh`，使用`--dynamic_dims`参数：

```bash
atc \
  --model="$ONNX_MODEL" \
  --framework=5 \
  --output="$OUTPUT_DIR/$OUTPUT_NAME" \
  --input_format=NCHW \
  --input_shape="voxels:-1,32,5;num_points:-1;coords:-1,4" \
  --dynamic_dims="6000,6000,6000;8000,8000,8000;10000,10000,10000" \
  --soc_version=Ascend310P1 \
  --log=error \
  --output_type=FP32
```

**关键参数说明**：

1. **`--input_shape`**: 使用`-1`表示动态维度
   - `voxels:-1,32,5`: 第0维动态，其他维度固定
   - `num_points:-1`: 第0维动态
   - `coords:-1,4`: 第0维动态，第1维固定为4

2. **`--dynamic_dims`**: 指定动态维度的具体档位
   - 格式：`"val1,val2,val3;val4,val5,val6;val7,val8,val9"`
   - 每个档位用分号`;`分隔
   - 每个档位中，每个动态维度的值用逗号`,`分隔
   - 我们有3个输入，每个输入的第0维都是动态的，所以每个档位需要3个值
   - 档位1: `6000,6000,6000` (voxels,num_points,coords的第0维都是6000)
   - 档位2: `8000,8000,8000`
   - 档位3: `10000,10000,10000`

## 导出结果

### ONNX模型
```bash
$ bash src/export/export_voxel_encoder_onnx.sh
开始导出 voxel_encoder ONNX 模型（支持动态shape）...
...
Exported: /home/ttt/PY080313/BEVFusion_Ascend310P/models/onnx/voxel_encoder.onnx
✓ ONNX 模型导出成功（支持动态体素数量）
```

### OM模型
```bash
$ bash src/export/export_voxel_encoder_om.sh
开始导出 OM 模型（支持动态shape）...
输入模型: /home/ttt/PY080313/BEVFusion_Ascend310P/models/onnx/voxel_encoder.onnx
输出目录: /home/ttt/PY080313/BEVFusion_Ascend310P/models/om
输出名称: voxel_encoder
动态档位: 6000, 8000, 10000
...
ATC run success, welcome to the next use.
✓ OM 模型导出成功: /home/ttt/PY080313/BEVFusion_Ascend310P/models/om/voxel_encoder.om
导出完成!
```

### 模型文件
```bash
$ ls -lh models/om/voxel_encoder.om
-rw------- 1 ttt ttt 59M 3月  2 15:22 models/om/voxel_encoder.om
```

**注意**：动态shape的OM模型文件大小（59M）比静态shape（2.9M）大很多，因为包含了3个档位的优化版本。

## 使用方法

### 运行时选择档位

在推理时，根据实际体素数量选择最接近的档位：

```python
# 假设当前有7500个体素
actual_voxels = 7500

# 选择最接近的档位
if actual_voxels <= 6000:
    gear = 6000
elif actual_voxels <= 8000:
    gear = 8000
else:
    gear = 10000

# 调整输入到选定的档位
# 方法1: padding到档位大小
padded_voxels = pad_to_size(voxels, gear)

# 方法2: 使用档位大小运行，然后截取结果
output = model(padded_voxels)
result = output[:actual_voxels]
```

### 性能优化

ATC工具会为每个档位生成优化的计算图：
- **档位6000**: 针对低密度场景优化
- **档位8000**: 针对中等密度场景优化
- **档位10000**: 针对高密度场景优化

运行时会自动选择最匹配的档位，无需手动指定。

## 技术细节

### ATC动态shape参数对比

| 参数 | 说明 | 使用场景 |
|------|------|----------|
| `--input_shape` | 固定shape | 静态shape模型 |
| `--input_shape_range` | shape范围（已废弃） | 旧版本动态shape |
| `--dynamic_batch_size` | 动态batch size | 仅batch维度动态 |
| `--dynamic_image_size` | 动态图像尺寸 | 图像H/W维度动态 |
| `--dynamic_dims` | 通用动态维度 | 任意维度动态 |

**注意**：这些参数互斥，只能使用其中一个。

### dynamic_dims格式详解

```
--dynamic_dims="gear1_dim1,gear1_dim2,...;gear2_dim1,gear2_dim2,...;..."
```

- **分号`;`**: 分隔不同的档位
- **逗号`,`**: 分隔同一档位中不同动态维度的值
- **值的数量**: 必须等于所有输入中`-1`的总数

**示例**：
- 1个输入，1个动态维度：`"64;128;256"`
- 2个输入，各1个动态维度：`"64,64;128,128;256,256"`
- 1个输入，2个动态维度：`"64,64;128,128"` (如H和W都动态)

## 常见问题

### Q1: 为什么OM模型文件这么大？

A: 动态shape的OM模型包含了多个档位的优化版本，每个档位都会生成独立的计算图和优化策略，因此文件大小是静态模型的数倍。

### Q2: 可以添加更多档位吗？

A: 可以，但要注意：
- 档位越多，模型文件越大
- 档位越多，编译时间越长
- 建议根据实际场景选择3-5个档位

### Q3: 档位之间的值怎么处理？

A: ATC会自动选择最接近的档位。例如：
- 实际6500个体素 → 使用档位8000（padding到8000）
- 实际7500个体素 → 使用档位8000
- 实际9000个体素 → 使用档位10000

### Q4: 动态shape会影响精度吗？

A: 不会。动态shape只是改变了输入尺寸的处理方式，不影响模型权重和计算逻辑，精度与静态shape模型完全一致。

## 参考资料

- [ATC工具使用说明](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/devaids/atctool/atlasatc_16_0020.html)
- [动态Shape配置](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/devaids/atctool/atlasatcparam_16_0020.html)
- [ONNX动态轴](https://pytorch.org/docs/stable/onnx.html#dynamic-axes)

## 更新历史

- 2026-03-02: 完成VoxelEncoder动态shape导出，支持6000/8000/10000三个档位
