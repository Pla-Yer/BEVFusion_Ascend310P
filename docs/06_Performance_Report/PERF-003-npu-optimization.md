## PERF-003: NPU Optimization

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
| Image Backbone | TBD | TBD | TBD |
| BEV Fusion | TBD | TBD | TBD |
| Detection Head | TBD | TBD | TBD |
| **NPU Total** | **TBD** | **TBD** | **TBD** |

### Optimization Strategies

#### Model Optimization
- [ ] 算子融合
- [ ] 模型量化
- [ ] 图优化

#### Memory Optimization
- [ ] 减少中间特征
- [ ] 内存复用
- [ ] FP16精度

#### Inference Optimization
- [ ] Dynamic batch size
- [ ] 异步推理
- [ ] 多流并行

### Bottleneck Analysis
待优化后分析

### Results
待测试

### Notes
- 需要使用ATC工具进行模型优化
- 需要评估优化对精度的影响
- 需要监控NPU利用率
