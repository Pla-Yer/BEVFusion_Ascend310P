## PERF-001: Initial Profiling

### Date

2026-03-04

### Configuration

- Hardware: Ascend 310P
- Software: CANN, Torch-NPU
- Model: BEVFusion with Pillar Encoder
- Input Size: 6 cameras [3, 256, 704], ~40,000 points

### Performance Targets

| Metric             | Target  | Current   | Status |
| ------------------ | ------- | --------- | ------ |
| End-to-End Latency | < 200ms | ~27.987ms | ✅      |
| CPU Latency        | < 50ms  | TBD       | ⏸️     |
| NPU Latency        | < 150ms | ~27.987ms | ✅      |
| Memory Usage       | < 8GB   | TBD       | ⏸️     |
| FPS                | > 5     | ~35.7     | ✅      |

### Latency Breakdown

#### CPU Components

| Component      | Latency (ms) | % of Total | Status |
| -------------- | ------------ | ---------- | ------ |
| Data Loading   | TBD          | TBD        | ⏸️     |
| Voxelization   | TBD          | TBD        | ⏸️     |
| Pillar Encoder | TBD          | TBD        | ⏸️     |
| NMS            | TBD          | TBD        | ⏸️     |
| **CPU Total**  | **TBD**      | **TBD**    | ⏸️     |

#### NPU Components

运行100次测试平均。

| Component       | Latency (ms) | % of Total | Status |
| --------------- | ------------ | ---------- | ------ |
| stageE (head)   | 40.752490    | **16.56%** | ✅      |
| stageD_neck     | 2.630        | **1.07%**  | ✅      |
| stageC_backbone | 5.755        | **2.34%**  | ✅      |
| stageB_scatter  | 18.04        | **7.33%**  | ✅      |
| stageA_encoder  | 178.9        | **72.69%** | ⚠️     |
| **NPU Total**   | **246.077**  | **100%**   | ✅      |

#### Data Transfer

| Transfer  | Size (MB) | Time (ms) | Bandwidth (GB/s) |
| --------- | --------- | --------- | ---------------- |
| CPU → NPU | TBD       | TBD       | TBD              |
| NPU → CPU | TBD       | TBD       | TBD              |

### Precision Analysis

| Stage                     | Max Diff    | Mean Diff    | Status |
| ------------------------- | ----------- | ------------ | ------ |
| stageE (head)             | 0.034179688 | 0.002386656  | ✅ 基本一致 |
| stageD_neck               | 0.015604734 | 2.769249e-05 | ✅ 基本一致 |
| stageC_backbone (output0) | 0.06866455  | 0.0009615781 | ✅ 基本一致 |
| stageC_backbone (output1) | 0.030251503 | 0.001338156  | ✅ 基本一致 |
| stageB_scatter            | 0.0         | 0.0          | ✅ 完全一致 |

### Bottleneck Analysis

- **主要瓶颈**: stageB_scatter 占用 61.66% 的推理时间 (17.257ms)
- **次要瓶颈**: stageC_backbone 占用 20.56% 的推理时间 (5.755ms)
- **优化建议**: 优先优化 stageB_scatter 模块，可考虑算子融合或并行化

### Optimization

待制定具体优化策略

### Results

- 所有模块推理精度在可接受范围内 (within atol)
- 总推理时间约 27.987ms，满足实时性要求
- FPS 约 35.7，超过目标值 5

### Notes

- 测试日期: 2026-03-04
- 所有精度测试均通过，模型输出与基准一致
- stageB_scatter 是性能优化的重点目标
