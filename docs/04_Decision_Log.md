# Decision Log

> 本文档记录项目中所有关键技术决策。
>
> 最后更新: 2026-02-28
>
> 格式：D-XXX: 决策标题

---

## Decision Index

| ID | Title | Date | Status |
|----|-------|------|--------|
| D-001 | Replace SparseConv3d with Pillar | 2026-02-28 | ✅ Adopted |
| D-002 | CPU/NPU Boundary Design | 2026-02-28 | ✅ Adopted |
| D-003 | FP16 Precision Mode | 2026-02-28 | 🔄 Evaluating |
| D-004 | Voxelization Implementation | 2026-02-28 | ✅ Adopted |
| D-005 | NMS on CPU | 2026-02-28 | ✅ Adopted |

---

## D-001: Replace SparseConv3d with Pillar

### Date
2026-02-28

### Background
原始 BEVFusion 使用 3D 稀疏卷积 (SparseConv3d) 处理点云数据。Ascend 310P NPU 不支持稀疏卷积算子，需要寻找替代方案。

### Options Considered

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A. PointPillars | 将点云转换为伪图像 | 成熟方案，易于实现 | 损失高度信息 |
| B. 自定义稀疏算子 | 在 NPU 上实现稀疏卷积 | 保持原始精度 | 开发周期长，难度大 |
| C. Dense Voxel | 使用密集体素 | 实现简单 | 内存占用大，效率低 |
| D. Point-based | 直接处理点云 | 精度高 | NPU 支持有限 |

### Choice
**Option A: PointPillars**

### Reason
1. PointPillars 是成熟的工业方案，在 nuScenes 上有良好表现
2. 实现复杂度低，开发周期短
3. 没有特殊算子需求，无需 NPU 特殊支持
4. 内存效率高，适合边缘部署

### Risk
- 高度信息损失可能导致精度下降 
- 需要重新训练或微调模型

### Mitigation
- 使用 Pillar 特征增强（添加高度统计特征）
- 评估精度损失，必要时进行微调

### Outcome
| 模型版本              | mAP    | NDS    | data_time | time   |
| --------------------- | ------ | ------ | --------- | ------ |
| second                | 0.2625 | 0.3200 | 0.0033    | 0.1293 |
| pointpillar           | 0.2115 | 0.2425 | 0.0048    | 0.1915 |
| bevfusion(second)     | 0.2506 | 0.2810 | 0.0092    | 0.2007 |
| befusion(pointpillar) | 0.2195 | 0.2474 | 0.0153    | 0.3160 |

---

## D-002: CPU/NPU Boundary Design

### Date
2026-02-28

### Background
需要合理划分 CPU 和 NPU 的计算边界，平衡性能和开发复杂度。

### Options Considered

| Option | CPU | NPU | Pros | Cons |
|--------|-----|-----|------|------|
| A. 最小 NPU | Voxel, Pillar, Fusion, Head | 仅 Backbone | NPU 开发少 | 性能差 |
| B. 最大 NPU | 仅 Voxel, NMS | Pillar, Backbone, Fusion, Head | 性能最优 | 开发复杂 |
| C. 平衡方案 | Voxel, NMS | Pillar, Backbone, Fusion, Head | 平衡 | 需要优化数据传输 |

### Choice
**Option C: 平衡方案**

### Reason
1. Voxelization  涉及稀疏操作，NPU 不擅长，且需要定制动态输入算子
2. Backbone、Fusion、Head 是密集计算，NPU 效率高
3. NMS 计算量小，CPU 实现简单
4. 最小化 NPU 自定义算子开发

### Risk
- CPU/NPU 数据传输可能成为瓶颈
- CPU 实现性能可能不足

### Mitigation
- 使用共享内存减少数据拷贝
- 优化 CPU 实现（多线程、向量化）
- 监控数据传输时间

### Outcome
待验证

---

## D-003: FP16 Precision Mode

### Date
2026-02-28

### Background
Ascend 310P 支持 FP16 和 INT8 精度。需要选择合适的精度模式。

### Options Considered

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A. FP32 | 全精度推理 | 精度最高 | NPU 效率低，内存大 |
| B. FP16 | 半精度推理 | 平衡精度和性能 | 可能有精度损失 |
| C. INT8 | 8位整数量化 | 性能最优，内存最小 | 精度损失大，需要校准 |
| D. 混合精度 | 关键层 FP16，其他 INT8 | 平衡 | 实现复杂 |

### Choice
**Option B: FP16** (当前阶段)

### Reason
1. FP16 在精度和性能间取得平衡
2. 无需复杂的量化校准流程
3. Ascend 310P 对 FP16 有良好支持
4. 后续可升级到 INT8

### Risk
- 某些层可能对 FP16 敏感
- 精度损失需要验证

### Mitigation
- 逐层验证精度
- 对敏感层保持 FP32
- 建立精度回归测试

### Outcome
待验证

---

## D-004: Voxelization Implementation

### Date
2026-02-28

### Background
需要在 CPU 上实现高效的 Voxelization。

### Options Considered

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A. NumPy | 纯 NumPy 实现 | 简单，易调试 | 性能一般 |
| B. PyTorch CPU | PyTorch CPU 实现 | 可利用向量化 | 依赖 PyTorch |
| C. Cython | Cython 扩展 | 性能好 | 开发复杂 |
| D. C++ Extension | C++ 扩展 | 性能最优 | 开发复杂 |

### Choice
**Option B: PyTorch CPU** (主) + **Option A: NumPy** (备)

### Reason
1. PyTorch CPU 可利用向量化优化
2. 与其他模块保持一致
3. 开发效率高
4. NumPy 作为备选，便于调试

### Risk
- 性能可能不如 C++ 实现

### Mitigation
- 使用 torch.jit 优化
- 多线程并行处理
- 必要时开发 C++ 扩展

### Outcome
待验证

---

## D-005: NMS on CPU

### Date
2026-02-28

### Background
需要决定 NMS 的实现位置。

### Options Considered

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A. CPU NMS | 在 CPU 上实现 | 简单，灵活 | 可能成为瓶颈 |
| B. NPU NMS | 在 NPU 上实现 | 性能好 | 需要自定义算子 |
| C. 集成到 Head OM | NMS 作为模型一部分 | 端到端 | 灵活性差 |

### Choice
**Option A: CPU NMS**

### Reason
1. NMS 计算量相对较小
2. CPU 实现简单，易于调试
3. 保持灵活性（可调整阈值）
4. 避免额外的 NPU 算子开发

### Risk
- 高密度场景可能成为瓶颈

### Mitigation
- 使用优化的 NMS 实现
- 监控 NMS 耗时
- 必要时迁移到 NPU

### Outcome
待验证

---

## Decision Template

```markdown
## D-XXX: [Decision Title]

### Date
YYYY-MM-DD

### Background
[Describe the context and problem]

### Options Considered
| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A. | | | |
| B. | | | |

### Choice
**Option X: [Selected Option]**

### Reason
1. [Reason 1]
2. [Reason 2]

### Risk
- [Risk 1]
- [Risk 2]

### Mitigation
- [Mitigation 1]
- [Mitigation 2]

### Outcome
[Result after implementation]
```

---

## Notes

- 所有重大技术决策必须记录在此
- 决策需要经过充分讨论和评估
- 定期回顾决策结果，总结经验
