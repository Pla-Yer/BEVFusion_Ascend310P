# Module Status

> 最后更新: 2026-02-28
>
> 本文档追踪各模块的开发、转换和集成状态。

---

## Status Legend

| 符号 | 含义 |
|------|------|
| ✅ | 完成 |
| 🔄 | 进行中 |
| ⏸️ | 暂停 |
| ❌ | 阻塞 |
| ➖ | 不适用 |

---

## Overall Progress

```
[████████░░░░░░░░░░░░] 40%
```

---

## Module Status Table

### 点云处理模块

| Module | CPU | ONNX | OM | Precision Verified | Integrated | Status |
|--------|-----|------|----|--------------------|------------|--------|
| Voxelization | 🔄 | ➖ | ➖ | ➖ | ⏸️ | CPU 实现中 |
| Pillar Encoder | ➖ | ➖ | 🔄 | ➖ | ⏸️ | 替代 SparseConv |

### 图像处理模块

| Module | CPU | ONNX | OM | Precision Verified | Integrated | Status |
|--------|-----|------|----|--------------------|------------|--------|
| Image Backbone (ResNet) | ➖ | ✅ | ✅ | ⏸️ | ⏸️ | 待精度验证 |
| FPN | ➖ | ✅ | ✅ | ⏸️ | ⏸️ | 待精度验证 |
| Image Neck | ➖ | ✅ | ✅ | ⏸️ | ⏸️ | 待精度验证 |
| Lift-Splat-Shoot | ➖ | ⏸️ | ⏸️ | ➖ | ⏸️ | 待开发 |

### BEV 融合模块

| Module | CPU | ONNX | OM | Precision Verified | Integrated | Status |
|--------|-----|------|----|--------------------|------------|--------|
| BEV Fusion | ➖ | ✅ | ✅ | ✅ | ⏸️ | 待合并 |

### 点云检测网络模块

| Module            | CPU  | ONNX | OM   | Precision Verified | Integrated | Status |
| ----------------- | ---- | ---- | ---- | ------------------ | ---------- | ------ |
| Pts Backbone+Neck | ➖    | ✅    | ✅    | ✅                  | ⏸️          | 待合并 |

### 检测头模块

| Module | CPU | ONNX | OM | Precision Verified | Integrated | Status |
|--------|-----|------|----|--------------------|------------|--------|
| Detection Head | ➖ | ✅    | ✅    | ✅                  | ⏸️ | 待合并 |

### 后处理模块

| Module | CPU | ONNX | OM | Precision Verified | Integrated | Status |
|--------|-----|------|----|--------------------|------------|--------|
| NMS | 🔄 | ➖ | ➖ | ➖ | ⏸️ | CPU 实现中 |
| Box Decoder | 🔄 | ➖ | ➖ | ➖ | ⏸️ | CPU 实现中 |
| Score Filter | 🔄 | ➖ | ➖ | ➖ | ⏸️ | CPU 实现中 |

---

## Detailed Module Status

### 1. Voxelization

**状态**: 🔄 进行中

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
- [ ] 单元测试
- [ ] 集成测试

**阻塞项**: 无

---

### 2. Pillar Encoder

**状态**: 🔄 进行中

**描述**: 替代 SparseConv3d 的 Pillar 特征编码器

**实现方式**: NPU (OM)

**关键参数**:
- Input: Voxel Features
- Output: Pseudo-Image [C, H, W]
- Pillar Size: [0.075, 0.075]

**进度**:
- [x] 架构设计
- [x] 基础实现
- [ ] 精度验证
- [ ] 性能优化

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

**状态**: 🔄 NPU 适配中

**描述**: BEV 特征池化操作

**实现方式**: NPU (自定义算子/OM)

**关键参数**:
- Depth: 118
- Grid Size: [128, 128]

**进度**:
- [x] 算子分析
- [x] NPU 实现
- [x] 精度验证
- [ ] 性能优化

**阻塞项**:
- NPU 算子开发

---

### 5. Pts Backbone+Neck

**状态**: ✅待合并

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

**阻塞项**: 无

---

### 6. Detection Head

**状态**: ✅待合并

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

**阻塞项**: 无

### 7. NMS

**状态**: 🔄 CPU 实现中

**描述**: 非极大值抑制后处理

**实现方式**: CPU (PyTorch)

**关键参数**:
- IoU Threshold: 0.1
- Score Threshold: 0.1
- Max Output: 500

**进度**:
- [x] 基础实现
- [ ] 性能优化
- [ ] 精度验证

**阻塞项**: 无

---

## Integration Status

| Integration Point | Status | Notes |
|-------------------|--------|-------|
| Data Loading | ✅ | 完成 |
| Points→Voxel | 🔄 | 进行中 |
| Voxel → Pillar | ⏸️ | 待 Pillar 完成 |
| Pillar → BEV Fusion | ⏸️ | 待开发 |
| Image → BEV Fusion | ⏸️ | 待开发 |
| BEV Fusion → Head | ✅ | 已完成 |
| Head → NMS | ✅ | 已完成 |
| End-to-End | ⏸️ | 待开发 |

---

## Weekly Update Log

### Week 1 (2026-02-28)
- 项目前期工作总结
- 架构设计完成

---

## Next Steps

1. 完成 Voxelization 验证
2. 完成 Pillar Encoder OM模型转化
3. 搭建好雷达分支检测全流程
