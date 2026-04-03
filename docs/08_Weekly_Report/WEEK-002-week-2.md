## WEEK-002: Week 2 (2026-03-03 ~ 2026-03-07)

### Goals
- [x] 完成Voxelization CPU实现
- [x] 完成Pillar Encoder OM模型转换
- [x] 完成端到端推理流程搭建
- [x] 完成初步性能分析
- [x] 完成初步精度评估

### Completed
1. **Voxelization CPU实现**
   - 完成CPU版本的体素化算子实现
   - 解决段错误问题(见ERR-003)
   - 完成性能优化

2. **Pillar Encoder OM模型转换**
   - 完成ONNX导出
   - 解决ONNX兼容性问题(见ERR-005, ERR-007)
   - 完成OM模型转换
   - 解决动态输入问题(见ERR-008)

3. **端到端推理流程搭建**
   - 完成雷达分支全流程推理
   - 解决点云数据维度问题(见ERR-009)
   - 解决坐标映射问题(见ERR-006)
   - 实现CPU+NPU混合推理

4. **性能分析**
   - 完成初步性能profiling(见PERF-001)
   - NPU推理时间约515.987ms
   - FPS约1.9
   - 识别性能瓶颈:stageB_scatter占用61.66%

5. **精度评估**
   - 完成单帧推理评估(见EVAL-003)
   - 单帧mAP: 0.1193 (PyTorch: 0.1171)
   - 多帧mAP: 0.19 (PyTorch: 0.23)
   - 发现LoadPointsFromMultiSweeps参数重要性

6. **问题解决**
   - 解决setuptools版本问题(见ERR-001)
   - 解决ONNX导出失败问题(见ERR-002)
   - 解决模型输出不匹配问题(见ERR-004)

### Blockers
- 多帧推理精度仍有差距: OM多帧mAP(0.19)低于PyTorch(0.23),需要后续分析原因
- 其他评估指标(mATE, mASE, mAOE, mAVE, mAAE)存在差异,需要优化

### Next Week Plan
1. 分析多帧推理精度差距原因
2. 优化后处理流程
3. 完善per-class结果分析
4. 开始CPU模块性能优化
5. 准备集成测试

### Metrics
| Metric | Value |
|--------|-------|
| Commits | 15+ |
| Documents | 9 (ERR-001~009) |
| Evaluation Reports | 1 (EVAL-003) |
| Performance Reports | 1 (PERF-001) |
| Decision Logs | 1 (D-003) |
| mAP (单帧) | 0.1193 |
| mAP (多帧) | 0.19 |
| FPS | 1.9 |
