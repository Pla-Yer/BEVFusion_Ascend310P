# Project Overview
> 最后更新: 2026-03-20
>
> 本文档描述了 BEVFusion 迁移到 Ascend 310P 的项目计划
## Background

BEVFusion 是一种先进的多传感器融合感知算法，将摄像头和激光雷达数据融合到统一的鸟瞰图(BEV)空间中进行3D目标检测。原始实现深度依赖 CUDA-based SparseConv3d 和自定义 CUDA 算子。

**核心问题：**

- Ascend 310P NPU 不支持 CUDA
- Ascend 310P 缺乏稀疏卷积原生算子支持

## Core Challenges

### 1. SparseConv3d 迁移

- 原始 BEVFusion 使用 3D 稀疏卷积处理点云
- Ascend 310P 无原生稀疏卷积算子
- 需要设计替代方案（如 Pillar-based 方法）

### 2. 自定义 CUDA 算子迁移

- `voxelization` 算子需要 CPU 实现
- `bev_pool` 算子需要 NPU 适配
- 其他自定义算子的边界划分

### 3. CPU/NPU 边界设计

- 合理划分计算负载
- 最小化数据传输开销
- 保证端到端推理效率

## Project Goals

### 主要目标

1. **端到端推理**：在 Ascend 310P 上实现完整的 BEVFusion 推理流程
2. **精度保持**：mAP 下降 < 10%，NDS 下降 < 15%
3. **稳定评估**：支持 nuScenes 验证集完整评估

### 当前任务

- **雷达侧端到端推理**：✅ 已完成 - 在 Ascend 310P 上实现 BEVFusion 的雷达分支推理流程
- **融合模型推理**：✅ 已完成 - 完成 LiDAR + Camera 融合模型端到端推理
- **性能优化**：✅ 已完成 - 全流程优化完成，总流程时间从2334ms降至166ms，FPS达到6.01
- **精度优化**：✅ 已完成 - mAP从0.189提升到0.219，完成FP16量化
- **bev_pool优化**：✅ 已完成 - 通过算法优化和并行化实现性能提升（见PERF-006~009）
- **模型量化**：🔄 计划中 - 计划使用PTQ进行量化优化

## Technical Stack

| 组件            | 原始方案                | 迁移实现                 | 状态 |
| ------------- | ------------------- | -------------------- | ---- |
| 点云体素化         | CUDA算子              | CPU实现                | ✅ 完成 |
| 点云编码          | SparseConv3d        | Pillar Encoder OM 模型 | ✅ 完成 |
| backbone+neck | SECOND              | NPU OM 模型            | ✅ 完成 |
| 检测头           | Transformer Decoder | OM 模型                | ✅ 完成 |
| NMS           | CUDA NMS            | CPU NMS              | ✅ 完成 |
| 评估            | mm3d pipline        | Nuscenes pipline     | ✅ 完成 |
| 模型量化          | FP32                | FP16                 | ✅ 完成 |
| BEV Pool      | CUDA算子              | 算道优化+并行化             | ✅ 完成 |
| 流水线优化         | -                   | 双缓冲+预取               | ✅ 完成 |

## Project Scope

### In Scope

- 模型训练
- 模型迁移与适配
- 精度验证与调优
- 性能优化
- 部署文档

### Out of Scope

- 数据集收集
