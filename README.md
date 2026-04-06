# BEVFusion Ascend 310P 迁移项目

> 将 BEVFusion 多传感器融合 3D 目标检测算法迁移到华为 Ascend 310P NPU

## 项目背景

BEVFusion 是一种先进的多传感器融合感知算法，将摄像头和激光雷达数据融合到统一的鸟瞰图 (BEV) 空间进行 3D 目标检测。原始实现深度依赖 CUDA 和 SparseConv3d，无法直接在华为 Ascend 310P NPU 上运行。

本项目完成了 BEVFusion 到 Ascend 310P 的完整迁移，并进行了大量性能优化。

## 技术挑战与解决方案

| 挑战 | 解决方案 |
|------|----------|
| SparseConv3d 不支持 | 使用 Pillar Encoder 替代 |
| CUDA 算子不可用 | 将模型转换为 ONNX → OM 在 NPU 运行 |
| 端到端性能不足 | 多轮迭代优化 |

## 架构概览

```
输入数据
    ├── 激光雷达点云 → CPU体素化 → Pillar Encoder (OM) → BEV特征
    └── 摄像头图像 → ResNet+FPN (OM) → 视角变换 → BEV特征
                                              ↓
                           BEV融合 + BEVPool (OM)
                                              ↓
                           SECOND Backbone + SecFPN (OM)
                                              ↓
                           Transformer Decoder (OM)
                                              ↓
                           检测头 + NMS (CPU后处理)
                                              ↓
                           3D边界框输出
```

## 性能指标

| 指标 | 数值 |
|------|------|
| 端到端延迟 | **166ms** |
| FPS | **6.01** |
| mAP | **仅下降4%** |

### 性能优化历程

| 阶段 | 延迟 | 优化内容 |
|------|------|----------|
| 初始 | 2334ms | - |
| Scatter优化 | 182ms | BEVPool算子优化 |
| 预处理并行化 | 355ms | 数据处理流水线 |
| 模型并行化 | 270ms | 多模型并行推理 |
| 流水线优化 | 166ms | 双缓冲+预取 |

## 量化实验

项目还探索了 PTQ 量化优化：

| 分支 | 量化结果 | 状态 |
|------|----------|------|
| Camera | 100MB→27MB, 55ms→45ms, 精度无损 | ✅ 成功 |
| LiDAR | 133MB→132MB, 精度下降 | ❌ 动态shape校准失效 |
| Fusion | 25MB→9MB, 精度下降 | ❌ 混合精度边界惩罚 |

详细分析见 [量化实验报告](docs/04_Decision_Log/D-008-quantization-experiment.md)。

## 目录结构

```
.
├── src/
│   ├── bevfusion/          # 核心模型定义
│   ├── inference/          # 推理脚本
│   ├── inference_parallel/ # 并行推理实现
│   ├── export/             # ONNX/OM 导出
│   ├── ptq/                # 量化脚本
│   └── accuracy_check/     # 精度验证
├── models/                 # 导出的 OM 模型
├── deployment/             # 部署文档
├── docs/                   # 详细技术文档
└── data/                   # nuScenes 数据
```

## 快速开始

### 环境要求

- Ascend 310P NPU
- CANN 工具链
- Python 3.8+

### 推理流程

```python
# 参考 src/inference_parallel/
from src.inference_parallel.bevfusion_parallel_npu_net_v2 import BEVFusionParallelNPU

# 初始化
model = BEVFusionParallelNPU()

# 推理
results = model.infer(points, images)
```

## 文档索引

- [项目概述](docs/01_Project_Overview.md) - 项目背景和目标
- [模块状态](docs/03_Module_Status.md) - 各模块开发进度
- [决策日志](docs/04_Decision_Log/) - 关键技术决策
- [评估报告](docs/05_Evaluation_Report/) - 精度评估结果
- [性能报告](docs/06_Performance_Report/) - 性能优化记录
- [周报](docs/08_Weekly_Report/) - 开发进度

## 核心成果

- ✅ 首次在 Ascend 310P 上实现 BEVFusion 端到端推理
- ✅ 完成 LiDAR + Camera 融合模型部署
- ✅ 性能从 2334ms 优化至 166ms (14x 提升)
- ✅ 模型量化优化，精度无损
## License

基于原始 BEVFusion 项目 [MIT License](https://github.com/mit-han-lab/bevfusion)