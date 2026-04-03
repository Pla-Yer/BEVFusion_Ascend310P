## PERF-008: 模型推理并行化优化

### Date
2026-03-20

### Configuration
- Hardware: Ascend 310P1
- Software: Ascend Toolkit 8.3.RC1
- Model: BEVFusion Parallel NPU (三OM架构)
- Input Size: 6 cameras, 256x704, dynamic voxels

### Metrics

**优化前性能（单OM）：**
| Component | Latency (ms) | % of Total | Status |
|-----------|--------------|------------|--------|
| sweep_loading | 36.14 | 10.2% | ✅ |
| voxelization | 67.01 | 18.9% | ⚠️ |
| image_loading | 32.64 | 9.2% | ✅ |
| precompute_depth | 24.85 | 7.0% | ✅ |
| precompute_geometry | 2.36 | 0.7% | ✅ |
| npu_inference | 185.03 | 52.1% | ⚠️ |
| decode | 0.69 | 0.2% | ✅ |
| nms | 2.80 | 0.8% | ✅ |
| transform | 2.61 | 0.7% | ✅ |
| **total_per_sample** | **355.01** | **100%** | ✅ |

**优化后性能（三OM并行）：**
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

### Bottleneck Analysis

**识别的瓶颈：**
- 单OM模型中LiDAR分支与Camera分支串行执行
- 无法显式控制两分支并行
- 静态输入和中间结果难做精细生命周期管理

### Optimization

**三OM并行架构设计：**

**1. 模型边界划分**
- **LiDAR Branch OM**：输入voxels/num_points/coords，输出lidar_bev
- **Camera Branch OM**：输入imgs/depth/pool_lookup/pool_mask，输出camera_bev
- **Fusion Head OM**：输入camera_bev/lidar_bev，输出检测头结果

**2. Runtime并行执行**
```python
# 创建3个stream：lidar_stream, camera_stream, fusion_stream
# LiDAR和Camera分支分别在自己的stream中execute_async
# 两路都完成后，Fusion Head在第三个stream中执行
```

**3. H2D过程优化**
- 将计算的查找表LUT固定在device上，减少copy时间
- 将lidar branch与camera branch的输出直接接到head的输入上
- 避免无谓的H2D、D2H或D2D操作

**优化效果：**
- LiDAR分支先下发，减少wall time
- 两分支真正并行执行

### Results

**性能提升：**
- NPU推理时间：185.03ms → 135.18ms（**提升26.9%**）
- 总流程时间：355.01ms → 270.11ms（**提升23.9%**）

**NPU推理详细性能：**
```
Stage              Total(s)   Mean(ms)    Std(ms)    Min(ms)    Max(ms)
lut_build             0.132     131.93       0.00     131.93     131.93
lidar_h2d             0.325       4.01       0.99       2.57      10.24
camera_h2d            0.808       9.97       0.74       9.11      13.23
branch_stage          7.114      87.83       3.37      71.06      97.42
fusion_h2d            0.000       0.00       0.00       0.00       0.01
fusion_infer          3.695      45.61       0.22      45.20      46.28
fusion_d2h            0.078       0.96       0.07       0.78       1.25
total                10.945     135.12       3.39     118.34     144.64
```

**关键成果：**
- ✅ 实现LiDAR与Camera分支真正并行
- ✅ 显式控制分支生命周期
- ✅ 减少数据传输开销
- ✅ 总流程时间降至270ms

### Notes

**技术要点：**
- ATC编译时模型已支持10路并行计算流（stream_num=10）
- 业务侧显式控制两条分支的开始、结束、中间结果转移
- LiDAR分支耗时更长，优先下发

**下一步优化方向：**
- 流水线优化：预处理与推理并行
- 双缓冲机制：下一帧数据预加载
