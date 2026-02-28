# Experiment Log Table

> 本文档记录所有实验的概览和结果。
>
> 最后更新: 2026-02-28
>
> 实验编号格式: E-XXX

---

## Experiment Index

| Exp ID | Date | Description | mAP | NDS |
|--------|------|-------------|-----|-----|
| E-001 | 2026-02-28 | GPU Baseline | 0.2625 | 0.3200 |
| E-002 | 2026-02-28 | Pillar Baseline | 0.2115 | 0.2425 |
| E-003 | 2026-02-28 | Pillar Baseline Change(voxel_size,pts_voxel_encoder) | TBD | TBD |
| E-004 | 2026-02-28 | E-003 Change(pts_backbone) | TBD | TBD |
| E-005 | TBD | Voxel Resolution 0.05 | TBD | TBD |
| E-006 | TBD | Voxel Resolution 0.1 | TBD | TBD |

---

## Detailed Experiment Log

### E-001: GPU Baseline

**Date**: TBD

**Description**: 原始 BEVFusion 在 GPU 上的基准性能

**Configuration**:
- Hardware: NVIDIA GPU
- Config: bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py

**Results**:
| Metric | Value  |
|--------|-------|
| mAP    | 0.2625 |
| NDS    | 0.3200 |
| mATE   | 0.4822 |
| mASE   | 0.5015 |
| mAOE   | 1.1876 |
| mAVE   | 0.7442 |
| mAAE   | 0.3849 |

**Notes**: mini

---

### E-002: Pillar Baseline

**Date**: TBD

**Description**: 使用 Pillar Encoder 替代 SparseConv3d

**Configuration**:
- Hardware: GPU
- Config：bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d2.py

**Results**:
| Metric | Value  | Diff vs Baseline |
|--------|-------|---------------|
| mAP    | 0.2115 | -0.0510          |
| NDS    | 0.2425 | -0.0775          |
| mATE   | 0.6122 | +0.1300          |
| mASE   | 0.5807 | +0.0792          |
| mAOE   | 1.0655 | -0.1221          |
| mAVE   | 0.9680 | +0.2238          |
| mAAE   | 0.4713 | +0.0864          |

**Notes**: mini

---

### E-003: Pillar Baseline Change(voxel_size,pts_voxel_encoder) 

**Date**: 2026.02.28

**Description**: 基于Pillar baseline，修改了voxel_size,pts_voxel_encoder配置

**Configuration**:
- Hardware: GPU
- Config：bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d3.py
- Changes：
  - Voxel Size: [0.30, 0.30, 8.0]->[0.20,0.20,8]
  - feat_channels=[64,128,256]->[64]


**Results**:
| Metric | Value  | Diff vs E-002 |
|--------|-------|---------------|
| mAP    | 0.1804 | -0.0311       |
| NDS    | 0.2373 | -0.0052       |
| mATE   | 0.7084 | +0.0962       |
| mASE   | 0.6260 | +0.0453       |
| mAOE   | 0.9343 | -0.1312       |
| mAVE   | 0.8022 | -0.1658       |
| mAAE   | 0.4582 | -0.0131       |

**Notes**: test

---

### E-004: E-003 Change(pts_backbone)

**Date**: 2026.02.28

**Description**: 在E-003的基础上，加深Pts网络

**Configuration**:
- Hardware: GPU
- Config：bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d4.py
- Changes：
  - pts_backbone - out_channels：[128, 256]->[64,128, 256]
  - pts_backbone - layer_nums:[5, 5]->[3, 5, 5]

**Results**:

| Metric | Value  | Diff vs E-003 |
| ------ | ------ | ------------- |
| mAP    | 0.1301 | -0.0503       |
| NDS    | 0.1781 | -0.0592       |
| mATE   | 0.7692 | +0.0608       |
| mASE   | 0.6635 | +0.0375       |
| mAOE   | 1.0734 | +0.1391       |
| mAVE   | 0.9232 | +0.1210       |
| mAAE   | 0.5139 | +0.0557       |

**Notes**: test

---

### E-005: Voxel Resolution 0.05

**Date**: TBD

**Description**: 使用更小的体素尺寸

**Configuration**:
- Hardware: CPU + Ascend 310P
- Model: BEVFusion with Pillar Encoder
- Precision: FP16
- Dataset: nuScenes Validation
- Voxel Size: [0.05, 0.05, 0.2]

**Results**:
| Metric | Value | Diff vs E-002 |
|--------|-------|---------------|
| mAP | TBD | TBD |
| NDS | TBD | TBD |
| Memory | TBD | TBD |
| Latency | TBD | TBD |

**Notes**: 待运行

---

### E-006: Voxel Resolution 0.1

**Date**: TBD

**Description**: 使用更大的体素尺寸

**Configuration**:
- Hardware: CPU + Ascend 310P
- Model: BEVFusion with Pillar Encoder
- Precision: FP16
- Dataset: nuScenes Validation
- Voxel Size: [0.1, 0.1, 0.2]

**Results**:
| Metric | Value | Diff vs E-002 |
|--------|-------|---------------|
| mAP | TBD | TBD |
| NDS | TBD | TBD |
| Memory | TBD | TBD |
| Latency | TBD | TBD |

**Notes**: 待运行

---

## Experiment Template

```markdown
### E-XXX: [Experiment Title]

**Date**: YYYY-MM-DD

**Description**: [Brief description]

**Configuration**:
- Hardware: [Hardware config]
- Model: [Model config]
- Precision: [FP32/FP16/INT8]
- Dataset: [Dataset name]
- [Other parameters]

**Results**:
| Metric | Value | Diff vs Baseline |
|--------|-------|------------------|
| mAP | TBD | TBD |
| NDS | TBD | TBD |

**Notes**: [Additional notes]
```

---

## Experiment Categories

### A. Architecture Experiments
- E-002: Pillar Baseline
- E-003: Pillar + Height Features

### B. Precision Experiments
- E-004: FP16 Precision

### C. Resolution Experiments
- E-002: Voxel 0.075 (baseline)
- E-005: Voxel 0.05
- E-006: Voxel 0.1

### D. Performance Experiments
- TBD

---

## Best Practices

1. **每次只改变一个变量**
2. **记录完整的配置信息**
3. **保存实验日志和检查点**
4. **对比分析结果差异**
5. **及时更新文档**
