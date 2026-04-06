## ERR-006: coords列顺序错误导致BEV特征图崩溃

### Date
2026-03-05

### Environment
- OS: Linux
- Component: MiddleEncoder scatter操作
- Tool: PyTorch, ONNX

### Symptom
修复PointPillarsScatter后,发现scatter操作在torch与onnx间仍存在巨大差异:
```
max|Δ|=5.23e+02
mean|Δ|=6.23e-02
```

### Debug Process
1. 验证scatter操作实现,确认使用scatter_add方法正确
2. 怀疑coords的列顺序问题
3. 创建`debug/inspect_coords.py`脚本打印torch模型与onnx模型的coords顺序
4. 对比发现coords列顺序确实不同
5. 定位到错误的重排代码

### Root Cause

**错误的重排代码**:
```python
# BEVFusionFullDeploy.forward中的错误代码
coords[:, [0, 3, 1, 2]]  # 错误重排: 把(b,z,y,x)变成(b,x,z,y)
```

**问题分析**:
- 原始coords格式: (batch, z, y, x)
- 错误重排后: (batch, x, z, y)
- y维度全变成0,导致BEV特征图崩溃
- 所有特征被错误地放置到错误位置

### Fix

**修复方案**:
```python
# 删除错误的重排操作
# coords保持原始顺序: (batch, z, y, x)
# 不需要重排
```

**关键修改**:
- 从deploy代码中删除`coords[:, [0, 3, 1, 2]]`重排
- 保持coords原始顺序
- 确保与模型预期一致

### Verification
修复后:
- middle_encoder输出与torch模型对应
- BEV特征图正确生成
- coords顺序一致

### Lessons & Notes
- **维度顺序至关重要**: 不同模块对维度顺序的约定可能不同
- **避免不必要的重排**: 除非明确需要,否则不要改变维度顺序
- **调试方法**:
  - 打印中间结果的维度和数值
  - 对比不同实现间的数据格式
  - 使用inspect脚本检查数据流
- **文档重要性**: 明确记录每个模块的输入输出格式约定
