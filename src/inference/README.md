# BEVFusion OM模型推理、评价与可视化

本目录包含BEVFusion模型在Ascend NPU上的完整推理、评价和可视化实现。

## 目录结构

```
src/inference/
├── bevfusion_net.py          # OM模型推理基础类
├── bevfusion_evaluator.py    # 推理和评价主程序
├── visualize.py              # 可视化工具
├── run_inference.sh          # 推理脚本
├── run_visualization.sh      # 可视化脚本
└── README.md                 # 本文档
```

## 环境要求

### 推理环境 (Ascend)
- Python 3.7.10
- acl (Ascend Computing Language)
- numpy
- pyquaternion
- nuscenes-devkit

### 可视化环境 (PyTorch)
- Python 3.8+
- numpy
- matplotlib
- pyquaternion
- nuscenes-devkit

## 使用方法

### 1. 准备工作

#### 1.1 导出OM模型
如果还没有OM模型，请先运行导出脚本：

```bash
cd ../../export
bash export_bevfusion_full_om.sh
```

#### 1.2 准备数据集
确保NuScenes mini数据集已下载到 `data/nuscenes-mini/` 目录。

### 2. 运行推理和评价

```bash
# 激活Ascend环境
conda activate ascend

# 运行推理脚本
bash run_inference.sh
```

或者直接运行Python脚本：

```bash
conda activate ascend
python bevfusion_evaluator.py
```

### 3. 运行可视化

```bash
# 激活openmmlab环境
conda activate openmmlab

# 运行可视化脚本
bash run_visualization.sh
```

或者直接运行Python脚本：

```bash
conda activate openmmlab
python visualize.py
```

## 输出结果

### 推理结果
- `results_nusc.json`: NuScenes格式的检测结果文件

### 评价结果
- `eval_results/`: 包含以下内容
  - `metrics_summary.json`: 评价指标摘要
  - `metrics_details.json`: 详细评价指标
  - `plots/`: PR曲线和TP曲线图

### 可视化结果
- `vis_results/`: 包含以下内容
  - `bev_val_*.png`: BEV可视化图（绿色：预测，红色虚线：真值）
  - `summary_statistics.png`: 检测统计摘要图

## 主要功能

### 1. BEVFusionNet类 (`bevfusion_net.py`)
- 封装Ascend ACL推理接口
- 支持OM模型加载和推理
- 自动处理输入输出数据传输

### 2. BEVFusionEvaluator类 (`bevfusion_evaluator.py`)
- 完整的推理流程
- BEVFusion输出解码
- 基于距离的NMS
- 坐标系转换（LiDAR -> Global）
- NuScenes官方评价
- 性能统计

### 3. NuScenesVisualizer类 (`visualize.py`)
- BEV可视化
- 预测与真值对比
- 批量可视化
- 统计图表生成

## 配置参数

### BEVFusion模型参数
```python
pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.30, 0.30, 8.0]
grid_size = [360, 360, 1]
out_size_factor = 2
```

### NMS阈值
```python
NMS_THRESHOLDS = {
    "car": 1.5,
    "truck": 2.0,
    "construction_vehicle": 2.0,
    "bus": 2.5,
    "trailer": 3.0,
    "barrier": 0.5,
    "motorcycle": 1.0,
    "bicycle": 0.8,
    "pedestrian": 0.5,
    "traffic_cone": 0.3
}
```

## 性能统计

推理脚本会输出详细的性能统计信息，包括：
- 体素化时间
- 模型推理时间
- 结果解码时间
- NMS处理时间
- 坐标转换时间
- 总处理时间
- FPS统计

## 注意事项

1. **环境切换**：
   - 推理使用 `ascend` 环境（ACL）
   - 可视化使用 `openmmlab` 环境（matplotlib）

2. **模型路径**：
   - 默认模型路径：`models/om/bevfusion_full.om`
   - 可在脚本中修改

3. **数据集路径**：
   - 默认数据集路径：`data/nuscenes-mini/`
   - 可在脚本中修改

4. **输出解码**：
   - BEVFusion的输出解码需要根据具体模型结构调整
   - 当前实现基于TransFusionHead的输出格式

## 故障排除

### 问题1: OM模型加载失败
- 检查模型路径是否正确
- 检查模型是否与当前Ascend版本兼容

### 问题2: 推理结果为空
- 检查输入数据是否正确
- 检查体素化参数是否匹配
- 检查分数阈值设置

### 问题3: 评价失败
- 确保NuScenes数据集完整
- 检查结果文件格式是否正确

## 参考

- [NPUProject_centerpoint](../NPUProject_centerpoint/): CenterPoint参考实现
- [NuScenes数据集](https://www.nuscenes.org/)
- [Ascend ACL文档](https://www.hiascend.com/document)
