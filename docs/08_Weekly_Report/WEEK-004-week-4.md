## WEEK-004: Week 4 (2026-03-17 ~ 2026-03-21)

### Goals
- [x] 学习AscendC算子开发
- [x] 完成BEVPool性能优化
- [x] 实现预处理并行化
- [x] 实现模型推理并行化
- [x] 实现流水线优化

### Completed

1. **AscendC算子开发学习**
   - 实现add算子
   - 实现sigmoid算子
   - 理解AscendC算子开发流程

2. **BEVPool Scatter优化** (见PERF-006)
   - 通过profiling发现ScatterElements瓶颈
   - 将scatter替换为gather + reduceSum
   - NPU推理时间：1800ms → 182ms（提升89.9%）
   - 总流程时间：2334ms → 792ms

3. **预处理并行化优化** (见PERF-007)
   - 实现图像加载并行化：296ms → 32ms
   - 实现深度投影向量化：164ms → 24ms
   - 缓存传感器标定矩阵
   - 总流程时间：792ms → 355ms

4. **模型推理并行化优化** (见PERF-008)
   - 采用三OM架构：LiDAR/Camera/Fusion
   - 实现LiDAR与Camera分支真正并行
   - 优化H2D过程，减少数据传输
   - NPU推理时间：185ms → 135ms
   - 总流程时间：355ms → 270ms

5. **流水线与双缓冲优化** (见PERF-009)
   - 实现Host端双缓冲机制
   - 实现预处理与推理流水化
   - 预取线程与主线程并行工作
   - 总流程时间：270ms → 166ms
   - FPS达到6.01

6. **技术决策记录**
   - 完成OM模型集成策略决策 (见D-006)
   - 完成并行化优化策略决策 (见D-007)

### Blockers
- 无重大阻塞项

### Next Week Plan
1. 模型量化：使用PTQ（Post_Train Quantization）进行量化优化
2. 进一步性能优化：探索更多并行化和优化机会
3. 精度优化：在保证性能的前提下提升模型精度

### Metrics

| Metric | Value |
|--------|-------|
| Performance Reports | 4 (PERF-006~009) |
| Decision Logs | 2 (D-006~007) |
| Performance Improvement | 2334ms → 166ms (92.9%) |
| FPS | 0.43 → 6.01 (14x) |
| Model Accuracy | mAP=0.2137 (保持不变) |

### Key Achievements

**性能优化成果：**
- 总流程时间从2334ms降至166ms，提升92.9%
- FPS从0.43提升至6.01，提升14倍
- 模型精度保持不变：mAP=0.2137

**技术突破：**
- 成功实现全流程并行化优化
- 建立了完整的性能优化方法论
- 为后续量化优化奠定基础
