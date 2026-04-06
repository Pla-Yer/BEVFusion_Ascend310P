# System Architecture

> 最后更新: 2026-03-20
> 
> 本文档描述 BEVFusion 从 GPU 到 Ascend 310P 的系统架构迁移设计。

---

## Original GPU Pipeline

原始 BEVFusion 在 GPU 上的数据流：

<div align=center>
<img src="https://user-images.githubusercontent.com/34888372/215313913-4b43f8a1-e2e2-49ba-b631-992155351922.png" width="800"/>
</div>

---

## Reconstructed 310P Pipeline

迁移后的 Ascend 310P 数据流：

```
┌─────────────────────────────────────────────────────────────────┐
│                   310P Pipeline (Reconstructed)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      CPU Layer                           │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │                                                          │    │
│  │  LiDAR Data ──► Voxelization ──► Pillar Encoder          │    │
│  │       │              │                │                  │    │
│  │       │              ▼                ▼                  │    │
│  │       │         Voxel Features   Pillar Features         │    │
│  │       │                                │                 │    │
│  └───────┼────────────────────────────────┼─────────────────┘    │
│          │                                │                      │
│          ▼                                ▼                      │
│  ┌───────┴────────────────────────────────┴─────────────────┐    │
│  │                      NPU Layer                           │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │                                                          │    │
│  │  Camera Data ──► Image Backbone (OM) ──► BEV Feature      │    │
│  │       │              (ResNet+FPN)          │              │    │
│  │       │                  │                 │              │    │
│  │       │                  ▼                 ▼              │    │
│  │       │            Image Features    BEV Fusion (OM)      │    │
│  │       │                                    │              │    │
│  │       │                           Detection Head (OM)     │    │
│  │       │                                    │              │    │
│  └───────┼────────────────────────────────────┼──────────────┘    │
│          │                                    │                  │
│          ▼                                    ▼                  │
│  ┌───────┴────────────────────────────────────┴──────────────┐   │
│  │                      CPU Layer                           │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │                                                          │    │
│  │                    NMS (CPU)                             │    │
│  │                       │                                  │    │
│  │                       ▼                                  │    │
│  │                  Detections                              │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## CPU/NPU Boundary Design

### CPU 负责

| 模块           | 原因           | 实现方式              |
| ------------ | ------------ | ----------------- |
| Voxelization | 稀疏操作，NPU 不支持 | NumPy/PyTorch CPU |
| NMS          | 后处理，计算量小     | PyTorch CPU       |
| Depth Generation | 深度估计计算 | CPU |
| Geometry Calculation | 几何变换计算 | CPU |

### NPU 负责

| 模块                | 原因              | 实现方式  |
| ----------------- | --------------- | ----- |
| Image Backbone    | 密集计算，NPU 高效     | OM 模型 |
| Pillar Encoder    | 替代 SparseConv3d | OM 模型 |
| BEV Fusion        | 密集计算            | OM 模型 |
| Pts Backbone+Neck | 密集计算            | OM 模型 |
| Detection Head    | 密集计算            | OM 模型 |
| BEV Pool          | 特征池化            | 算道优化+并行化(已完成) |

---

## Data Flow Diagram

### 详细数据流

```
Step 1: 数据加载
├── Camera Images: [N, 3, H, W] × 6 cameras
└── LiDAR Points: [M, 4] (x, y, z, intensity)

Step 2: CPU Voxelization
├── Input: LiDAR Points [M, 4]
├── Output: Voxel Features [V, max_points, 4]
└── Voxel Coords: [V, 3]

Step 3: NPU Pillar Encoder
├── Input: Voxel Features [V, max_points, 4]
├── Output: Pillar Features [P, C]
└── Pillar Coords: [P, 3]

Step 4: NPU Image Backbone
├── Input: Camera Images [6, 3, H, W]
├── Output: Image BEV Features [B, C, H_bev, W_bev]
└── Model: OM format

Step 5: NPU BEV Fusion
├── Input: Pillar Features + Image BEV Features
├── Output: Fused BEV Features [B, C, H_bev, W_bev]
└── Model: OM format

Step 6: NPU Pts Backbone+Neck
├── Input: Fused BEV Features
├── Output: Inhenced BEV Features [B, C, H_bev, W_bev]
└── Model: OM format

Step 7: NPU Detection Head
├── Input: Inhenced BEV Features 
├── Output: Raw Detections [N, 10] (cls, reg, dir)
└── Model: OM format

Step 8: CPU NMS
├── Input: Raw Detections
└── Output: Final Detections [K, 9] (box3d, score, label)
```

## References

- [BEVFusion Paper](https://arxiv.org/abs/2205.13542)
- [Ascend 310P Documentation](https://www.hiascend.com/)
- [Torch-NPU](https://github.com/Ascend/pytorch)
