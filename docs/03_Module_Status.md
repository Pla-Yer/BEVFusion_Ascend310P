# Module Status

> 最后更新: 2026-04-02
> 
> 本文档追踪各模块的开发、转换和集成状态。

---

## Status Legend

| 符号  | 含义  |
| --- | --- |
| ✅   | 完成  |
| 🔄  | 进行中 |
| ⏸️  | 暂停  |
| ❌   | 阻塞  |
| ➖   | 不适用 |

---

## Overall Progress

```
[██████████████████░░] 90%
```

---

## Module Status Table

### 点云处理模块

| Module         | CPU | ONNX | OM  | Precision Verified | Integrated | Status |
| -------------- | --- | ---- | --- | ------------------ | ---------- | ------ |
| Voxelization   | ✅   | ➖    | ➖   | ✅                  | ✅          | 已完成    |
| Pillar Encoder | ➖   | ✅    | ✅   | ✅                  | ✅          | 已完成(已优化) |

### 图像处理模块

| Module                  | CPU | ONNX | OM  | Precision Verified | Integrated | Status |
| ----------------------- | --- | ---- | --- | ------------------ | ---------- | ------ |
| Image Backbone (ResNet) | ➖   | ✅    | ✅   | ⏸️                 | ⏸️         | 待精度验证  |
| FPN                     | ➖   | ✅    | ✅   | ⏸️                 | ⏸️         | 待精度验证  |
| Image Neck              | ➖   | ✅    | ✅   | ⏸️                 | ⏸️         | 待精度验证  |
| Lift-Splat-Shoot        | ➖   | ⏸️   | ⏸️  | ➖                  | ⏸️         | 待开发    |

### BEV 融合模块

| Module     | CPU | ONNX | OM  | Precision Verified | Integrated | Status |
| ---------- | --- | ---- | --- | ------------------ | ---------- | ------ |
| BEV Fusion | ➖   | ✅    | ✅   | ✅                  | ✅          | 已完成    |
| BEV Pool   | ➖   | ✅    | ✅   | ✅                  | ✅          | 已优化(见PERF-006~009) |

### 点云检测网络模块

| Module            | CPU | ONNX | OM  | Precision Verified | Integrated | Status |
| ----------------- | --- | ---- | --- | ------------------ | ---------- | ------ |
| Pts Backbone+Neck | ➖   | ✅    | ✅   | ✅                  | ✅          | 已完成    |

### 检测头模块

| Module         | CPU | ONNX | OM  | Precision Verified | Integrated | Status |
| -------------- | --- | ---- | --- | ------------------ | ---------- | ------ |
| Detection Head | ➖   | ✅    | ✅   | ✅                  | ✅          | 已完成    |

### 后处理模块

| Module       | CPU | ONNX | OM  | Precision Verified | Integrated | Status |
| ------------ | --- | ---- | --- | ------------------ | ---------- | ------ |
| NMS          | ✅   | ➖    | ➖   | ✅                  | ✅          | 已完成    |
| Box Decoder  | ✅   | ➖    | ➖   | ✅                  | ✅          | 已完成    |
| Score Filter | ✅   | ➖    | ➖   | ✅                  | ✅          | 已完成    |

---

## Detailed Module Status

### 1. Voxelization

**状态**: ✅ 已完成

**描述**: 将点云转换为体素表示

**实现方式**: CPU (NumPy/PyTorch)

**关键参数**:

- Voxel Size: [0.075, 0.075, 0.2]
- Point Cloud Range: [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
- Max Voxels: 16000
- Max Points per Voxel: 32

**进度**:

- [x] 基础实现
- [x] 性能优化
- [x] 单元测试
- [x] 集成测试

**阻塞项**: 无

---

### 2. Pillar Encoder

**状态**: ✅ 已完成(已优化)

**描述**: 替代 SparseConv3d 的 Pillar 特征编码器

**实现方式**: NPU (OM)

**关键参数**:

- Input: Voxel Features
- Output: Pseudo-Image [C, H, W]
- Pillar Size: [0.075, 0.075]

**进度**:

- [x] 架构设计
- [x] 基础实现
- [x] 精度验证
- [x] 性能优化 (见PERF-005)

**优化成果**:
- stageA_encoder从178.9ms降至41ms
- NPU总推理从246ms降至108ms

**阻塞项**: 无

---

### 3. Image Backbone

**状态**: 🔄 进行中

**描述**: ResNet + FPN 图像特征提取

**实现方式**: NPU (OM)

**关键参数**:

- Backbone: ResNet-50
- FPN: 4 levels
- Input: [6, 3, 256, 704]
- Output: Multi-scale features

**进度**:

- [x] 模型选择
- [x] ONNX 导出
- [x] OM 转换
- [ ] 精度验证
- [ ] 性能测试

**阻塞项**:

- 动态 shape 处理
- FP16 精度验证

---

### 4. BEV Pool

**状态**: ✅ 已优化

**描述**: BEV 特征池化操作

**实现方式**: NPU (OM) + 算道优化

**关键参数**:

- Depth: 118
- Grid Size: [128, 128]

**进度**:

- [x] 算子分析
- [x] NPU 实现
- [x] 精度验证
- [x] 性能优化 (见PERF-006~009)

**优化成果**:
- Scatter优化：1800ms → 182ms (见PERF-006)
- 预处理并行化：792ms → 355ms (见PERF-007)
- 模型推理并行化：355ms → 270ms (见PERF-008)
- 流水线优化：270ms → 166ms (见PERF-009)
- 最终性能：FPS达到6.01

**阻塞项**: 无

---

### 5. Pts Backbone+Neck

**状态**: ✅ 已完成

**描述**: 点云特征提取网络

**实现方式**: NPU (OM)

**关键参数**:

- Backbone: SECOND
- FPN: 2 levels
- Input: [1,80+256,180,180]
- Output: Multi-scale features,[1,512,180,180]

**进度**:

- [x] 模型分析
- [x] ONNX 导出
- [x] OM 转换
- [x] 精度验证
- [x] 集成测试

**阻塞项**: 无

---

### 6. Detection Head

**状态**: ✅ 已完成

**描述**: 3D 目标检测头

**实现方式**: NPU (OM)

**关键参数**:

- Classes: 10
- Anchors: Multi-scale
- Output: [cls, reg, dir]

**进度**:

- [x] 模型分析
- [x] ONNX 导出
- [x] OM 转换
- [x] 精度验证
- [x] 集成测试

**阻塞项**: 无

### 7. NMS

**状态**: ✅ 已完成

**描述**: 非极大值抑制后处理

**实现方式**: CPU (PyTorch)

**关键参数**:

- IoU Threshold: 0.1
- Score Threshold: 0.1
- Max Output: 500

**进度**:

- [x] 基础实现
- [x] 性能优化
- [x] 精度验证

**阻塞项**: 无

---

## Integration Status

| Integration Point   | Status | Notes |
| ------------------- | ------ | ----- |
| Data Loading        | ✅      | 完成    |
| Points→Voxel        | ✅      | 完成    |
| Voxel → Pillar      | ✅      | 完成(已优化) |
| Pillar → BEV Fusion | ✅      | 完成    |
| Image → BEV Fusion  | ✅      | 完成    |
| BEV Fusion → Head   | ✅      | 已完成   |
| Head → NMS          | ✅      | 已完成   |
| End-to-End (Lidar)  | ✅      | 已完成   |
| End-to-End (Fusion) | ✅      | 已完成(性能已优化) |

---

## Weekly Update Log

### Week 1 (2026-02-28)

- 项目前期工作总结
- 架构设计完成

### Week 2 (2026-03-07)

- 完成Voxelization CPU实现
- 完成Pillar Encoder OM模型转换
- 完成端到端推理流程搭建
- 完成初步性能分析(见PERF-001)
- 完成初步精度评估(见EVAL-003)
- 解决多个技术问题(见ERR-001~009)

### Week 3 (2026-03-13)

- 完成各模块性能测试和精度测试
- 完成ScatterNdUpdate算子优化(见PERF-005)
- 完成体素化数量调整决策(见D-004)
- 完成模型量化策略决策(见D-005)
- 完成融合模型评估(见EVAL-004)
- 识别bev_pool性能瓶颈

### Week 4 (2026-03-20)

- 学习AscendC算子开发，实现add与sigmoid算子
- 完成BEVPool Scatter优化(见PERF-006)：1800ms → 182ms
- 完成预处理并行化优化(见PERF-007)：792ms → 355ms
- 完成模型推理并行化优化(见PERF-008)：355ms → 270ms
- 完成流水线与双缓冲优化(见PERF-009)：270ms → 166ms
- 完成OM模型集成策略决策(见D-006)
- 完成并行化优化策略决策(见D-007)
- 最终性能：FPS达到6.01，总流程时间166ms

### Week 5 (2026-03-30)

- 学习量化原理和AMCT量化工具
- 完成Camera分支量化实验(见PERF-010)：100MB→27MB，55ms→45ms，精度无损
- 完成LiDAR分支量化实验：133MB→132MB，精度下降(0.21→0.17)
- 完成Fusion分支量化实验：25MB→9MB，精度下降(0.21→0.12)，延迟增加(45ms→50ms)
- 完成量化实验决策(见D-008)
- 识别量化失败根因：动态shape校准失效、混合精度边界惩罚、结构不匹配

---

## Next Steps

1. ~~模型量化：使用PTQ（Post_Train Quantization）进行量化优化~~ (已完成，见PERF-010)
2. 量化策略优化：针对LiDAR和Fusion分支重新设计量化方案
3. 进一步性能优化：探索更多并行化和优化机会
4. 精度优化：在保证性能的前提下提升模型精度
