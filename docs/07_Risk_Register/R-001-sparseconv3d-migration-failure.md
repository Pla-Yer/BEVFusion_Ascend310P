## R-001: SparseConv3d 迁移失败

### Level
High

### Impact
阻塞整个 pipeline

### Probability
Medium

### Description
Ascend 310P 不支持稀疏卷积,需要使用替代方案。如果替代方案精度损失过大或无法实现,将阻塞项目。

### Mitigation Strategy
1. 使用 PointPillars 作为替代方案
2. 评估精度损失,确保在可接受范围内
3. 准备备选方案(如自定义算子开发)

### Contingency Plan
- 如果 Pillar 方案精度损失 > 2%,考虑:
  - 微调模型
  - 开发自定义稀疏算子
  - 降低精度目标

### Status
🔄 Mitigating

### Last Update
2026-02-28
