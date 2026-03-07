## R-004: 性能不达标

### Level
Medium

### Impact
实时性要求无法满足

### Probability
Medium

### Description
端到端延迟可能超过 200ms 目标,无法满足实时性要求。

### Mitigation Strategy
1. 性能分析和瓶颈识别
2. CPU 模块优化(多线程、向量化)
3. NPU 模块优化(模型优化、算子融合)
4. 数据传输优化(共享内存、异步传输)

### Contingency Plan
- 如果性能不达标:
  - 降低输入分辨率
  - 简化模型结构
  - 调整性能目标

### Status
⏸️ Monitoring

### Last Update
2026-02-28
