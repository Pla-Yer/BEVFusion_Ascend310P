# OM模型推理结果分析工具

本目录包含用于分析BEVFusion OM模型推理结果不正确问题的工具脚本。


## 工具说明

### 1. `get_om_data.py` - 获取OM模型数据

在Ascend环境中运行，收集OM模型的输入和输出数据。

**功能：**
- 加载点云数据
- 执行Voxelization
- 运行OM模型推理
- 保存输入数据（voxels, coords, num_points）
- 保存输出数据（dense_heatmap, top_cls, query_heatmap_score等）

**使用方法：**
```bash
# 在Ascend环境中运行
python get_om_data.py \
    --model_path models/om/bevfusion_full.om \
    --pcl_path data/nuscenes-mini/samples/LIDAR_TOP/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530398292.pcd.bin \
    --sample_token sample_0 \
    --output_dir ./om_data
```

**输出：**
```
om_data/
└── sample_0/
    ├── voxels.npy              # 输入：体素数据 [6000, 32, 5]
    ├── coords.npy              # 输入：坐标数据 [6000, 4]
    ├── num_points.npy          # 输入：每个体素的点数 [6000]
    ├── dense_heatmap.npy       # 输出：密集热力图
    ├── top_cls.npy             # 输出：top-k类别
    ├── query_heatmap_score.npy # 输出：查询热力图分数
    ├── heatmap_q.npy           # 输出：查询热力图
    ├── center.npy              # 输出：中心点
    ├── height.npy              # 输出：高度
    ├── dim.npy                 # 输出：尺寸
    ├── rot.npy                 # 输出：旋转
    ├── vel.npy                 # 输出：速度
    └── metadata.json           # 元数据
```

### 2. `get_pytorch_data.py` - 获取PyTorch模型数据

在OpenMMLab环境中运行，收集PyTorch模型的输入和输出数据。

**功能：**
- 加载点云数据
- 执行Voxelization（可选择使用与OM相同的voxels数据）
- 运行PyTorch模型推理
- 保存输入数据
- 保存输出数据
- 保存中间特征（用于调试）

**使用方法：**
```bash
# 在OpenMMLab环境中运行
python get_pytorch_data.py \
    --config_path src/configs/bevfusion_lidar_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    --checkpoint_path work_dirs/bevfusion/epoch_20.pth \
    --pcl_path data/nuscenes-mini/samples/LIDAR_TOP/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530398292.pcd.bin \
    --sample_token sample_0 \
    --output_dir ./pytorch_data \
    --om_data_dir ./om_data  # 可选：使用与OM相同的voxels数据
```

**输出：**
```
pytorch_data/
└── sample_0/
    ├── voxels.npy              # 输入数据
    ├── coords.npy
    ├── num_points.npy
    ├── dense_heatmap.npy       # 输出数据
    ├── top_cls.npy
    ├── query_heatmap_score.npy
    ├── heatmap_q.npy
    ├── center.npy
    ├── height.npy
    ├── dim.npy
    ├── rot.npy
    ├── vel.npy
    ├── voxel_features.npy      # 中间特征
    ├── bev_feat.npy
    ├── neck_feat.npy
    ├── fusion_feat.npy
    └── metadata.json
```

### 3. `compare_data.py` - 数据对比分析

对比OM模型和PyTorch模型的输入输出数据，找出差异点。

**功能：**
- 对比输入数据（voxels, coords, num_points）
- 对比输出数据（所有输出张量）
- 分析中间特征统计信息
- 诊断问题原因
- 生成可视化图表

**使用方法：**
```bash
python compare_data.py \
    --om_data_dir ./om_data \
    --pytorch_data_dir ./pytorch_data \
    --sample_token sample_0 \
    --output_dir ./analysis_results
```

**输出：**
```
analysis_results/
├── sample_0_analysis.json          # 详细分析结果
├── visualizations/
│   └── sample_0/
│       ├── heatmap_comparison.png  # 热力图对比
│       └── output_scatter.png      # 输出散点图
```

### 4. `run_analysis.py` - 主分析脚本

一键运行完整的分析流程。

**功能：**
- 自动执行所有步骤
- 生成总结报告
- 提供问题诊断和建议

**使用方法：**
```bash
# 完整流程
python run_analysis.py \
    --config_path src/configs/bevfusion_lidar_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    --checkpoint_path work_dirs/bevfusion/epoch_20.pth \
    --om_model_path models/om/bevfusion_full.om \
    --pcl_path data/nuscenes-mini/samples/LIDAR_TOP/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530398292.pcd.bin \
    --sample_token sample_0 \
    --output_dir ./analysis_results

# 跳过已完成的步骤
python run_analysis.py --skip_om --skip_pytorch
```

**输出：**
```
analysis_results/
├── om_data/                        # OM模型数据
├── pytorch_data/                   # PyTorch模型数据
├── sample_0_analysis.json          # 详细分析结果
├── sample_0_report.txt             # 总结报告
└── visualizations/                 # 可视化图表
```

## 使用流程

### 方案一：分步执行（推荐用于调试）

1. **在Ascend环境中收集OM数据**
   ```bash
   cd src/accuracy_check/analyze_om_evaluator
   python get_om_data.py
   ```

2. **在OpenMMLab环境中收集PyTorch数据**
   ```bash
   # 使用相同的voxels数据
   python get_pytorch_data.py --om_data_dir ./om_data
   ```

3. **对比分析数据**
   ```bash
   python compare_data.py
   ```

### 方案二：一键执行（推荐用于快速分析）

```bash
python run_analysis.py
```

## 分析结果解读

### 1. 输入数据对比

如果输入数据不匹配，可能的原因：
- Voxelization实现不一致
- 点云预处理不一致
- coords格式转换错误

### 2. 输出数据对比

如果输出数据不匹配，可能的原因：
- 模型导出错误
- ONNX转换引入误差
- OM模型编译精度设置问题

### 3. 关键输出分析

- **top_cls不匹配**: 说明topk选择的索引不同，需要检查heatmap计算和NMS实现
- **heatmap不匹配**: 说明特征提取或head计算有问题
- **center/dim/rot不匹配**: 说明预测头或坐标解码有问题

## 常见问题

### Q1: 如何确认是输入数据问题还是模型问题？

A: 使用 `--om_data_dir` 参数让PyTorch模型使用与OM相同的voxels数据。如果输出仍然不匹配，则是模型问题；如果输出匹配，则是输入数据问题。

### Q2: 如何处理不同环境的依赖？

A:
- `get_om_data.py` 需要在Ascend环境中运行（需要acl库）
- `get_pytorch_data.py` 需要在OpenMMLab环境中运行（需要mmdet3d）
- `compare_data.py` 和 `run_analysis.py` 可以在任意环境运行（只需要numpy和matplotlib）

### Q3: 如何分析多个样本？

A: 修改 `--sample_token` 参数，对每个样本运行分析，然后对比结果。

## 解决方案建议

根据分析结果，可能的解决方案：

1. **输入数据问题**
   - 统一Voxelization实现
   - 检查点云预处理流程
   - 验证coords格式转换

2. **模型导出问题**
   - 检查ONNX导出脚本
   - 验证opset版本
   - 检查动态shape处理

3. **OM模型编译问题**
   - 调整精度设置（FP32/FP16）
   - 检查算子支持情况
   - 验证模型优化选项

4. **后处理问题**
   - 检查分数计算逻辑
   - 验证坐标解码实现
   - 确认NMS参数设置
