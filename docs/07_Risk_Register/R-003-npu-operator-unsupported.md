## R-003: NPU 算子不支持

### Level
High

### Impact
部分模块无法在 NPU 上运行

### Probability
Medium

### Description
BEVFusion 使用了一些自定义算子(如 bev_pool),这些算子可能在 Ascend 310P 上没有原生支持。

### Mitigation Strategy
1. 调研 Ascend 算子支持列表
2. 使用等价的标准算子组合替代
3. 开发自定义 NPU 算子
4. 将不支持算子迁移到 CPU

### Contingency Plan
- 如果无法在 NPU 上实现:
  - 迁移到 CPU 实现
  - 接受性能损失
  - 寻找替代算法

### Status
🔄 Mitigating

### Last Update
2026-02-28
