## R-006: ONNX 转换失败

### Level
Medium

### Impact
无法生成 OM 模型

### Probability
Medium

### Description
部分 PyTorch 算子可能无法正确导出为 ONNX,或导出后在 Ascend 上转换失败。

### Mitigation Strategy
1. 使用标准算子,避免自定义算子
2. 分模块导出,逐步验证
3. 使用 ONNX Simplifier 简化模型
4. 参考 Ascend 支持的算子列表

### Contingency Plan
- 如果 ONNX 转换失败:
  - 重写不支持算子
  - 使用等价算子组合
  - 寻求华为技术支持

### Status
⏸️ Monitoring

### Last Update
2026-02-28
