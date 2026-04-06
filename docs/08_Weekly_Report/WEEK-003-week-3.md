## WEEK-003: Week 3 (2026-03-10 ~ 2026-03-14)

### Goals
- [x] 完成各模块性能测试和精度测试
- [x] 优化stageA_encoder性能瓶颈
- [x] 提升模型精度
- [x] 完成模型量化
- [x] 完成融合模型初步评估
- [ ] 优化bev_pool性能瓶颈

### Completed
1. **性能优化**
   - 完成各模块性能测试，识别stageA_encoder为主要瓶颈(178.9ms, 72.69%)
   - 通过msprof工具分析，发现ScatterNdUpdate算子占比80%以上
   - 实施优化：将索引赋值操作改为拼接操作(见PERF-005)
   - 优化效果：stageA_encoder从178.9ms降至41ms，NPU总耗时从246ms降至108ms

2. **精度提升**
   - 决策：将体素化数量上限从6000增加到10000(见D-004)
   - 效果：mAP从0.189提升到0.219，提升15.9%
   - 代价：推理耗时从108ms增加到132ms

3. **模型量化**
   - 决策：使用FP16量化策略(见D-005)
   - 效果：推理耗时从132ms降至102ms，mAP仅从0.219降至0.218
   - 全流程pipeline从0.25s降至0.22s

4. **LiDAR分支完整评估**
   - 完成全流程计时统计(见EVAL-004)
   - mAP: 0.2186
   - 单样本总耗时: 18.559s (Mean: 229.1ms)

5. **融合模型初步评估**
   - 完成LiDAR+Camera融合模型端到端推理
   - 识别bev_pool为主要性能瓶颈(134.97ms)
   - 发现融合推理耗时高达1427ms，主要由于数据拷入/拷出频繁
   - mAP: 0.217

6. **其他尝试**
   - 尝试MatMul替代Scatter：未实现更优
   - 尝试index_add → ScatterAdd：未实现更优
   - 尝试AOE工具调优：难以提高性能

### Blockers
- **bev_pool性能瓶颈**：融合模型推理耗时高达1427ms，bev_pool单独作为OM模型推理耗时达1300ms
- **NPU不擅长bev_pool计算**：bev_pool主要计算是将特征撒在BEV网格中，这种计算NPU不擅长
- **融合模型精度略降**：融合后mAP(0.217)略低于LiDAR单独(0.2186)

### Next Week Plan
1. 学习AscendC算子设计
2. 将bev_pool设计为自定义NPU算子
3. 优化融合模型数据传输流程
4. 进一步提升融合模型精度

### Metrics
| Metric | Value |
|--------|-------|
| Performance Reports | 1 (PERF-005) |
| Decision Logs | 2 (D-004, D-005) |
| Evaluation Reports | 1 (EVAL-004) |
| LiDAR mAP | 0.2186 |
| Fusion mAP | 0.217 |
| LiDAR Inference | 102ms |
| Fusion Inference | 1427ms |
