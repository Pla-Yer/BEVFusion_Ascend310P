## PERF-002: CPU Optimization

### Date
TBD

### Configuration
- Hardware: Ascend 310P
- Software: CANN, Torch-NPU
- Model: BEVFusion with Pillar Encoder
- Input Size: 6 cameras [3, 256, 704], ~40,000 points

### Baseline Performance

| Component | Before (ms) | After (ms) | Improvement |
|-----------|-------------|------------|-------------|
| Voxelization | TBD | TBD | TBD |
| Pillar Encoder | TBD | TBD | TBD |
| NMS | TBD | TBD | TBD |
| **CPU Total** | **TBD** | **TBD** | **TBD** |

### Optimization Strategies

#### Voxelization Optimization
- [ ] 多线程并行处理
- [ ] 向量化优化
- [ ] 内存预分配

#### Pillar Encoder Optimization
- [ ] 使用 torch.jit.script
- [ ] 批量矩阵运算
- [ ] 特征缓存

#### NMS Optimization
- [ ] 使用 torchvision.ops.nms
- [ ] 预过滤低分检测
- [ ] 分类别并行 NMS

### Bottleneck Analysis
待优化后分析

### Results
待测试

### Notes
- 需要对比优化前后的性能
- 需要评估优化对精度的影响
- 需要记录优化实施细节
