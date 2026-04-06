## EVAL-003: 310P Pillar Version

### Date
2026-03-06

### Configuration
- Model: BEVFusion with Pillar Encoder
- Hardware: CPU (Voxel, Pillar) + NPU (Backbone, Head)
- Precision: FP16
- Dataset: nuScenes-mini Validation Set
- Data Processing: LoadPointsFromMultiSweeps=0 (单帧)

### Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| mAP | 0.1193 | > 0.50 | ⚠️ |
| NDS | 0.0972 | > 0.55 | ⚠️ |
| mATE | 0.7489 | - | - |
| mASE | 0.8755 | - | - |
| mAOE | 1.1282 | - | - |
| mAVE | 1.4359 | - | - |
| mAAE | 1.0000 | - | - |

### Per-Class Results

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

### Analysis

#### 问题发现
完成全流程推理后,mAP只有0.11,而torch原模型mAP为0.23,存在显著差距。由于之前的实验已验证OM模型与torch模型的输出可以保证一致,因此问题定位到后处理或数据配置。

#### 调试过程
1. **尝试后处理调整**:
   - 将检测阈值变为0
   - 将topk从200增加到500
   - 修改坐标映射关系
   - 结果: 这些调整都没有显著效果

2. **配置文件分析**:
   - 分析mm3d的配置文件
   - 发现训练与测试时都有参数`LoadPointsFromMultiSweeps`
   - 该参数实现: 将之前的n个点云也加载进入当前帧
   - 我的配置pipeline没有这个设置

3. **根因确认**:
   - 缺少多帧点云数据导致对某一帧的分析少了很多数据
   - 单帧点云信息不足,影响检测精度

#### 对比验证

**PyTorch单帧结果 (LoadPointsFromMultiSweeps=0)**:
```
mAP: 0.1171
mATE: 0.7108
mASE: 0.6044
mAOE: 1.2353
mAVE: 1.3260
mAAE: 0.6203
NDS: 0.1650
```

**OM推理结果 (单帧)**:
```
mAP: 0.1193
mATE: 0.7489
mASE: 0.8755
mAOE: 1.1282
mAVE: 1.4359
mAAE: 1.0000
NDS: 0.0972
```

**对比分析**:
- mAP相近(0.1171 vs 0.1193),说明OM推理基本正确
- 其他指标存在差异,需要后续精度提升改进

**OM多帧结果 (LoadPointsFromMultiSweeps>0)**:
```
mAP: 0.19
```

**PyTorch多帧结果**:
```
mAP: 0.23
```

**差距分析**:
- 多帧情况下,OM的mAP(0.19)仍低于PyTorch(0.23)
- 需要后续分析原因

### Comparison

| Configuration | mAP | NDS | Notes |
|---------------|-----|-----|-------|
| PyTorch (单帧) | 0.1171 | 0.1650 | LoadPointsFromMultiSweeps=0 |
| OM (单帧) | 0.1193 | 0.0972 | 单帧推理 |
| PyTorch (多帧) | 0.23 | - | LoadPointsFromMultiSweeps>0 |
| OM (多帧) | 0.19 | - | 多帧推理 |

### Notes
- **数据配置重要性**: LoadPointsFromMultiSweeps参数对精度影响显著
- **单帧vs多帧**: 多帧点云提供更丰富的时序信息,提升检测精度
- **后续工作**:
  - 分析多帧情况下OM与PyTorch的精度差距原因
  - 优化后处理流程
  - 提升其他指标(mATE, mASE, mAOE, mAVE, mAAE)
  - 完善per-class结果分析
