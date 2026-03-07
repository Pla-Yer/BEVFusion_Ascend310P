## D-001: Replace SparseConv3d with Pillar

### Date
2026-02-28

### Background
原始 BEVFusion 使用 3D 稀疏卷积 (SparseConv3d) 处理点云数据。Ascend 310P NPU 不支持稀疏卷积算子,需要寻找替代方案。

### Options Considered

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A. PointPillars | 将点云转换为伪图像 | 成熟方案,易于实现 | 损失高度信息 |
| B. 自定义稀疏算子 | 在 NPU 上实现稀疏卷积 | 保持原始精度 | 开发周期长,难度大 |
| C. Dense Voxel | 使用密集体素 | 实现简单 | 内存占用大,效率低 |
| D. Point-based | 直接处理点云 | 精度高 | NPU 支持有限 |

### Choice
**Option A: PointPillars**

### Reason
1. PointPillars 是成熟的工业方案,在 nuScenes 上有良好表现
2. 实现复杂度低,开发周期短
3. 没有特殊算子需求,无需 NPU 特殊支持
4. 内存效率高,适合边缘部署

### Risk
- 高度信息损失可能导致精度下降
- 需要重新训练或微调模型

### Mitigation
- 使用 Pillar 特征增强(添加高度统计特征)
- 评估精度损失,必要时进行微调

### Outcome

| 模型版本 | mAP | NDS | data_time | time |
|---------|-----|-----|-----------|------|
| second | 0.2625 | 0.3200 | 0.0033 | 0.1293 |
| pointpillar | 0.2115 | 0.2425 | 0.0048 | 0.1915 |
| bevfusion(second) | 0.2506 | 0.2810 | 0.0092 | 0.2007 |
| befusion(pointpillar) | 0.2195 | 0.2474 | 0.0153 | 0.3160 |
