## PERF-009: 流水线与双缓冲优化

### Date
2026-03-20

### Configuration
- Hardware: Ascend 310P1
- Software: Ascend Toolkit 8.3.RC1
- Model: BEVFusion Parallel NPU v2
- Input Size: 6 cameras, 256x704, dynamic voxels

### Metrics

**优化前性能：**
| Component | Latency (ms) | % of Total | Status |
|-----------|--------------|------------|--------|
| sweep_loading | 63.27 | 23.4% | ⚠️ |
| voxelization | 21.60 | 8.0% | ✅ |
| image_loading | 52.25 | 19.3% | ⚠️ |
| precompute_depth | 50.87 | 18.8% | ⚠️ |
| precompute_geometry | 0.63 | 0.2% | ✅ |
| npu_inference | 135.18 | 50.1% | ✅ |
| decode | 0.66 | 0.2% | ✅ |
| nms | 2.69 | 1.0% | ✅ |
| transform | 2.54 | 0.9% | ✅ |
| **total_per_sample** | **270.11** | **100%** | ✅ |

**优化后性能：**
| Component | Latency (ms) | % of Total | Status |
|-----------|--------------|------------|--------|
| sweep_loading | 53.40 | 32.1% | ⚠️ |
| voxelization | 18.14 | 10.9% | ✅ |
| image_loading | 24.21 | 14.6% | ✅ |
| precompute_depth | 84.86 | 51.1% | ⚠️ |
| precompute_geometry | 0.57 | 0.3% | ✅ |
| npu_inference | 133.71 | 80.5% | ✅ |
| decode | 1.40 | 0.8% | ✅ |
| nms | 6.32 | 3.8% | ⚠️ |
| transform | 6.50 | 3.9% | ⚠️ |
| **total_per_sample** | **166.09** | **100%** | ✅ |
| host_staging | 3.90 | - | ✅ |
| prefetch_wait | 14.78 | - | ✅ |

### Bottleneck Analysis

**识别的瓶颈：**
- 预处理与推理串行执行
- 每帧都需要等待预处理完成才能开始推理
- Host与Device之间数据传输存在等待时间

### Optimization

**双缓冲机制：**

**1. Host端双缓冲**
```python
# 在net里预分配两个host slot
# 分别存：voxels / num_points / coords / imgs / depth
# 当前帧推理时，下一帧可以先把预处理结果写进另一个slot
```

**2. 预处理与推理流水化**
```python
# evaluator现在会把下一帧的：
# - 点云加载
# - 图像加载
# - 体素化
# - depth预计算
# - host staging
# 放到后台预取线程里，和当前帧的NPU推理+后处理重叠起来
```

**优化效果：**
- 避免互相踩内存
- 减少反复分配/拼接
- 预处理与推理并行执行

### Results

**性能提升：**
- 总流程时间：270.11ms → 166.09ms（**提升38.5%**）
- 有效吞吐量：3.7 samples/s → **6.01 samples/s**
- Wall time显著降低

**Pipeline性能：**
```
Total wall time      : 13.473s
Mean wall/sample     : 166.33ms
Effective throughput : 6.01 samples/s
```

**NPU推理详细性能：**
```
Stage              Total(s)   Mean(ms)    Std(ms)    Min(ms)    Max(ms)
lut_build             0.235     234.67       0.00     234.67     234.67
host_stage            0.316       3.90       6.14       1.19      46.37
lidar_h2d             0.275       3.39       0.28       2.43       4.93
camera_h2d            0.765       9.44       0.83       8.41      15.74
branch_stage          7.056      87.11       3.98      70.22     101.49
fusion_h2d            0.000       0.00       0.00       0.00       0.01
fusion_infer          3.687      45.52       0.53      45.02      47.76
fusion_d2h            0.080       0.99       0.29       0.82       3.41
total                10.825     133.65       4.06     116.99     147.58
```

**关键成果：**
- ✅ 实现预处理与推理流水化
- ✅ 双缓冲机制有效减少等待时间
- ✅ FPS达到6.01
- ✅ 总流程时间降至166ms

### Notes

**技术要点：**
- 受AscendC算子开发中二级缓冲的启发
- 预取线程与主线程并行工作
- 有效隐藏了预处理延迟

**下一步工作计划：**
- 量化：初步计划使用PTQ（Post_Train Quantization）
