## PERF-006: BEVPool Scatter优化

### Date
2026-03-20

### Configuration
- Hardware: Ascend 310P1
- Software: Ascend Toolkit 8.3.RC1
- Model: BEVFusion Full NPU
- Input Size: 6 cameras, 256x704, dynamic voxels

### Metrics

**优化前性能：**
| Component | Latency (ms) | % of Total | Status |
|-----------|--------------|------------|--------|
| sweep_loading | 42.41 | 1.8% | ⚠️ |
| voxelization | 64.68 | 2.8% | ⚠️ |
| image_loading | 245.62 | 10.5% | ⚠️ |
| precompute_depth | 120.54 | 5.2% | ⚠️ |
| precompute_geometry | 50.78 | 2.2% | ⚠️ |
| npu_inference | 1800.77 | 77.2% | ❌ |
| decode | 0.66 | 0.03% | ✅ |
| nms | 2.73 | 0.1% | ✅ |
| transform | 2.51 | 0.1% | ✅ |
| **total_per_sample** | **2334.00** | **100%** | ❌ |

**优化后性能：**
| Component | Latency (ms) | % of Total | Status |
|-----------|--------------|------------|--------|
| sweep_loading | 64.19 | 8.1% | ⚠️ |
| voxelization | 72.74 | 9.2% | ⚠️ |
| image_loading | 296.56 | 37.4% | ❌ |
| precompute_depth | 164.65 | 20.8% | ❌ |
| precompute_geometry | 6.23 | 0.8% | ✅ |
| npu_inference | 182.22 | 23.0% | ✅ |
| decode | 0.67 | 0.08% | ✅ |
| nms | 2.77 | 0.3% | ✅ |
| transform | 2.54 | 0.3% | ✅ |
| **total_per_sample** | **792.10** | **100%** | ⚠️ |

### Bottleneck Analysis

通过profiling发现模型大量时间花在ScatterElements算子上，这是BEVPool环节的随机写过程带来的巨大延时。

**问题根因：**
- ScatterElements操作导致大量随机内存写操作
- 写冲突和串行操作导致性能瓶颈
- NPU推理耗时从预期的200ms左右飙升至1800ms

### Optimization

**优化策略：**
将scatter操作替换为gather + reduceSum组合：
- 原过程：直接scatter写入目标位置
- 优化后：先gather收集数据，再通过reduceSum聚合

**技术细节：**
```python
# 优化前：ScatterElements随机写
# 优化后：Gather + ReduceSum
# 大大减少写冲突与串行操作
```

### Results

**性能提升：**
- NPU推理时间：1800.77ms → 182.22ms（**提升89.9%**）
- 总流程时间：2334.00ms → 792.10ms（**提升66.1%**）
- 模型精度：mAP=0.2137（几乎不变）

**关键成果：**
- ✅ NPU推理性能接近预期水平
- ✅ 消除了ScatterElements瓶颈
- ⚠️ 发现新的瓶颈：image_loading和precompute_depth

### Notes

**下一步优化方向：**
1. image_loading 296ms：6张图串行disk I/O
2. precompute_depth 164ms：torch + Python per-camera循环，每帧重算逆矩阵
3. sweep/voxelization ~135ms：可以并行预读下一帧

**参考文档：**
- [深入理解硬件计算性能——以BEVPOOL为例.md](/home/ttt/PY080313/BEVFusion_Ascend310P/深入理解硬件计算性能——以BEVPOOL为例.md)
