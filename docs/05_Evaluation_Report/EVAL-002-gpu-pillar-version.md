## EVAL-002: GPU Pillar Version

### Date
2026-02-28

### Configuration
- Model: BEVFusion with Pillar Encoder
- Hardware: NVIDIA GPU
- Precision: FP32
- Dataset: nuScenes-mini Validation Set

### Metrics

| Metric | Value | Baseline | Diff | Status |
|--------|-------|----------|------|--------|
| mAP | 0.2115 | 0.2625 | -0.0510 | ⚠️ |
| NDS | 0.2425 | 0.3200 | -0.0775 | ⚠️ |
| mATE | 0.6122 | 0.4822 | +0.1300 | ⚠️ |
| mASE | 0.5807 | 0.5015 | +0.0792 | ⚠️ |
| mAOE | 1.0655 | 1.1876 | -0.1221 | ✅ |
| mAVE | 0.9680 | 0.7442 | +0.2238 | ⚠️ |
| mAAE | 0.4713 | 0.3849 | +0.0864 | ⚠️ |

### Per-Class Results

| Class | AP | ATE | ASE | AOE | Diff AP |
|-------|----|----|-----|-----|---------|
| car | 0.6387 | 0.3075 | 0.1905 | 1.3475 | -0.0507 |
| truck | 0.2529 | 0.2694 | 0.2650 | 1.3577 | -0.1213 |
| bus | 0.5247 | 0.7361 | 0.1923 | 0.8360 | -0.1404 |
| trailer | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| construction_vehicle | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| pedestrian | 0.6240 | 0.1876 | 0.3093 | 0.5619 | -0.1788 |
| motorcycle | 0.0748 | 0.3124 | 0.3560 | 1.4869 | -0.0133 |
| bicycle | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| traffic_cone | 0.0000 | 0.3093 | 0.4940 | – | -0.0057 |
| barrier | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

### Analysis
使用mini数据集,将特征编码从SparseConv3d变为PointPillar风格。结果显示:
- mAP下降5.1%,NDS下降7.75%
- 位置误差(mATE)增加13%,说明定位精度下降
- 尺度误差(mASE)增加7.92%,说明尺寸估计变差
- 朝向误差(mAOE)改善12.21%,这是唯一的正面结果
- 速度误差(mAVE)增加22.38%,说明速度估计变差

### Comparison
与EVAL-001 GPU Baseline相比,精度有明显下降,主要原因是Pillar方法损失了高度信息。

### Notes
- 高度信息损失对pedestrian和bus影响较大
- 需要考虑添加高度特征增强来缓解精度下降
- mAOE改善可能是因为Pillar方法对朝向估计更稳定
