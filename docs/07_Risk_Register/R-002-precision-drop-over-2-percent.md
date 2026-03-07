## R-002: 精度下降超过 2%

### Level
Medium

### Impact
部署风险,用户接受度降低

### Probability
Medium

### Description
由于 SparseConv3d → Pillar 的替换、FP32 → FP16 的精度转换、ONNX/OM 转换等因素,可能导致精度下降超过 10%。

### Mitigation Strategy
1. 逐模块验证精度
2. 使用混合精度策略
3. 对敏感层保持高精度
4. 进行模型微调

### Contingency Plan
- 如果精度下降 > 10%:
  - 分析误差来源
  - 针对性优化
  - 调整精度目标

### Status
⏸️ Monitoring

### Last Update
2026-02-28
