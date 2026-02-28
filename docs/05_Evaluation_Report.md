# Evaluation Report

> 本文档记录模型精度评估结果。
>
> 最后更新: 2026-02-28
>
> 评估数据集: nuScenes-mini Validation Set

---

## Evaluation Metrics

### nuScenes Detection Metrics

| Metric | Description |
|--------|-------------|
| mAP | Mean Average Precision |
| NDS | nuScenes Detection Score |
| mATE | Mean Average Translation Error |
| mASE | Mean Average Scale Error |
| mAOE | Mean Average Orientation Error |
| mAVE | Mean Average Velocity Error |
| mAAE | Mean Average Attribute Error |

---

## Baseline Results

### GPU Baseline (Original BEVFusion)

**模型**: BEVFusion with SparseConv3d

**硬件**: NVIDIA GPU

**精度**: FP32

| Metric | Value  |
|--------|-------|
| mAP    | 0.2625 |
| NDS    | 0.3200 |
| mATE   | 0.4822 |
| mASE   | 0.5015 |
| mAOE   | 1.1876 |
| mAVE   | 0.7442 |
| mAAE   | 0.3849 |

**备注**: 使用mini数据集复现获取

---

### Pillar Version (GPU)

**模型**: BEVFusion with Pillar Encoder

**硬件**: NVIDIA GPU

**精度**: FP32

| Metric | Value  | Diff vs Baseline |
| ------ | ------ | ---------------- |
| mAP    | 0.2115 | -0.0510          |
| NDS    | 0.2425 | -0.0775          |
| mATE   | 0.6122 | +0.1300          |
| mASE   | 0.5807 | +0.0792          |
| mAOE   | 1.0655 | -0.1221          |
| mAVE   | 0.9680 | +0.2238          |
| mAAE   | 0.4713 | +0.0864          |

**备注**: 使用mini数据集，将特征编码从sparseconvd3d变为pointpillar风格。

### Pillar Version (CPU Implementation)

**模型**: BEVFusion with Pillar Encoder

**硬件**: CPU (Voxel, Pillar) + NPU (Backbone, Head)

**精度**: FP16

| Metric | Value | Diff vs Baseline |
|--------|-------|------------------|
| mAP | TBD | TBD |
| NDS | TBD | TBD |
| mATE | TBD | TBD |
| mASE | TBD | TBD |
| mAOE | TBD | TBD |
| mAVE | TBD | TBD |
| mAAE | TBD | TBD |

**备注**: 待评估

---

## Per-Class Results

### Baseline (GPU)

| Class                | AP     | ATE    | ASE    | AOE    |
|-------|----|----|-----|-----|
| car                  | 0.6894 | 0.2643 | 0.1848 | 1.1446 |
| truck                | 0.3742 | 0.2534 | 0.2118 | 1.5470 |
| bus                  | 0.6651 | 0.4908 | 0.2573 | 0.3768 |
| trailer              | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| construction_vehicle | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| pedestrian           | 0.8028 | 0.1576 | 0.2935 | 0.5360 |
| motorcycle           | 0.0881 | 0.2354 | 0.3601 | 1.4029 |
| bicycle              | 0.0000 | 0.3136 | 0.3532 | 2.6811 |
| traffic_cone         | 0.0057 | 0.1070 | 0.3538 | –      |
| barrier              | 0.0000 | 1.0000 | 1.0000 | 1.0000 |

### Pillar Version (GPU)

| Class                | AP     | ATE    | ASE    | AOE    | Diff AP |
| -------------------- | ------ | ------ | ------ | ------ | ------- |
| car                  | 0.6387 | 0.3075 | 0.1905 | 1.3475 | -0.0507 |
| truck                | 0.2529 | 0.2694 | 0.2650 | 1.3577 | -0.1213 |
| bus                  | 0.5247 | 0.7361 | 0.1923 | 0.8360 | -0.1404 |
| trailer              | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000  |
| construction_vehicle | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000  |
| pedestrian           | 0.6240 | 0.1876 | 0.3093 | 0.5619 | -0.1788 |
| motorcycle           | 0.0748 | 0.3124 | 0.3560 | 1.4869 | -0.0133 |
| bicycle              | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000  |
| traffic_cone         | 0.0000 | 0.3093 | 0.4940 | –      | -0.0057 |
| barrier              | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000  |


### Pillar Version (310P)

| Class | AP | ATE | ASE | AOE | Diff AP |
|-------|----|----|-----|-----|---------|
| car | TBD | TBD | TBD | TBD | TBD |
| truck | TBD | TBD | TBD | TBD | TBD |
| bus | TBD | TBD | TBD | TBD | TBD |
| trailer | TBD | TBD | TBD | TBD | TBD |
| construction_vehicle | TBD | TBD | TBD | TBD | TBD |
| pedestrian | TBD | TBD | TBD | TBD | TBD |
| motorcycle | TBD | TBD | TBD | TBD | TBD |
| bicycle | TBD | TBD | TBD | TBD | TBD |
| traffic_cone | TBD | TBD | TBD | TBD | TBD |
| barrier | TBD | TBD | TBD | TBD | TBD |

备注：AP 取 `dist_0.5 / 1.0 / 2.0 / 4.0` 平均值

---

## Precision Analysis

### Expected Precision Impact

| Component | Expected Impact | Reason |
|-----------|-----------------|--------|
| SparseConv3d → Pillar | -0.5% ~ -1.0% mAP | 高度信息损失 |
| FP32 → FP16 | -0.1% ~ -0.3% mAP | 数值精度降低 |
| ONNX Conversion | -0.05% ~ -0.1% mAP | 算子转换误差 |
| OM Conversion | -0.05% ~ -0.1% mAP | 模型优化误差 |
| **Total Expected** | **-0.7% ~ -1.5% mAP** | |

### Height Information Loss Analysis

**问题**: Pillar 方法将点云压缩为 2D 伪图像，损失高度维度信息。

**影响**:
- 对高度敏感的类别（如 pedestrian, bicycle）可能受影响更大
- 对大型物体（如 truck, bus）影响较小

**缓解措施**:
- 在 Pillar 特征中添加高度统计量（max_z, min_z, mean_z）
- 使用更强的特征编码网络

---

## Ablation Studies

### Study 1: Pillar vs SparseConv3d

| Configuration | mAP | NDS |
|---------------|-----|-----|
| SparseConv3d (3x, 3x, 3x) | TBD | TBD |
| Pillar (baseline) | TBD | TBD |
| Pillar + Height Features | TBD | TBD |

### Study 2: Precision Mode

| Precision | mAP | NDS | Latency |
|-----------|-----|-----|---------|
| FP32 | TBD | TBD | TBD |
| FP16 | TBD | TBD | TBD |
| INT8 | TBD | TBD | TBD |

### Study 3: Voxel Resolution

| Voxel Size | mAP | NDS | Memory |
|------------|-----|-----|--------|
| [0.05, 0.05, 0.2] | TBD | TBD | High |
| [0.075, 0.075, 0.2] | TBD | TBD | Medium |
| [0.1, 0.1, 0.2] | TBD | TBD | Low |

---

## Error Analysis

### Common Failure Cases

1. **远距离小物体检测**
   - 原因: 点云稀疏，特征不足
   - 影响: pedestrian, bicycle 远距离 AP 低

2. **遮挡场景**
   - 原因: 部分点云缺失
   - 影响: 所有类别 AP 下降

3. **高速运动物体**
   - 原因: 速度估计误差
   - 影响: mAVE 指标

### Per-Distance Performance

| Distance Range | mAP | Notes |
|----------------|-----|-------|
| 0-10m | TBD | 近距离性能 |
| 10-20m | TBD | 中距离性能 |
| 20-30m | TBD | 远距离性能 |
| 30m+ | TBD | 超远距离性能 |

---

## Comparison with Target

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| mAP | > 0.50 | TBD | ⏸️ |
| NDS | > 0.55 | TBD | ⏸️ |
| mAP Drop | < 1% | TBD | ⏸️ |
| NDS Drop | < 1% | TBD | ⏸️ |

---

## Evaluation Log

### Eval-001: Initial Pillar Evaluation
- **Date**: TBD
- **Config**: Pillar baseline
- **Result**: TBD
- **Notes**: TBD

### Eval-002: Height Feature Enhancement
- **Date**: TBD
- **Config**: Pillar + height features
- **Result**: TBD
- **Notes**: TBD

---

## Next Steps

1. [ ] 获取 GPU baseline 评估结果
2. [ ] 完成 Pillar 版本评估
3. [ ] 分析精度差异来源
4. [ ] 进行消融实验
5. [ ] 优化精度
