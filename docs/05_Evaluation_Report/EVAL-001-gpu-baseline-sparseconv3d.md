## EVAL-001: GPU Baseline (SparseConv3d)

### Date
2026-02-28

### Configuration
- Model: BEVFusion with SparseConv3d
- Hardware: NVIDIA GPU
- Precision: FP32
- Dataset: nuScenes-mini Validation Set

### Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| mAP | 0.2625 | > 0.50 | ⏸️ |
| NDS | 0.3200 | > 0.55 | ⏸️ |
| mATE | 0.4822 | - | - |
| mASE | 0.5015 | - | - |
| mAOE | 1.1876 | - | - |
| mAVE | 0.7442 | - | - |
| mAAE | 0.3849 | - | - |

### Per-Class Results

| Class | AP | ATE | ASE | AOE |
|-------|----|----|-----|-----|
| car | 0.6894 | 0.2643 | 0.1848 | 1.1446 |
| truck | 0.3742 | 0.2534 | 0.2118 | 1.5470 |
| bus | 0.6651 | 0.4908 | 0.2573 | 0.3768 |
| trailer | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| construction_vehicle | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| pedestrian | 0.8028 | 0.1576 | 0.2935 | 0.5360 |
| motorcycle | 0.0881 | 0.2354 | 0.3601 | 1.4029 |
| bicycle | 0.0000 | 0.3136 | 0.3532 | 2.6811 |
| traffic_cone | 0.0057 | 0.1070 | 0.3538 | – |
| barrier | 0.0000 | 1.0000 | 1.0000 | 1.0000 |

### Analysis
使用mini数据集复现获取的baseline结果。该结果作为后续所有优化的对比基准。

### Comparison
这是第一个baseline评估,无对比数据。

### Notes
- 使用mini数据集,结果可能与完整数据集有差异
- 部分类别(trailer, construction_vehicle, bicycle, barrier)AP为0,可能需要更多数据或优化
