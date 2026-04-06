## ERR-007: ScatterElements算子索引越界

### Date
2026-03-05

### Environment
- OS: Linux
- Component: stageB_scatter (MiddleEncoder)
- Tool: ATC converter, msame
- Hardware: Ascend 310P

### Symptom
使用msame工具推理stageB_scatter的OM模型时,出现Aicpu kernel执行失败错误:
```
E39999: Aicpu kernel execute failed, device_id=0, stream_id=16, task_id=34
Op execute failed. origin_op_name [/ScatterElements_ascend_mbatch_batch_1]
errorCode=0x91
```

错误提示ScatterElements算子的index越界。

### Debug Process
1. 分析错误信息,确认是ScatterElements算子索引越界
2. 检查scatter操作的索引计算
3. 发现索引计算公式问题
4. 修改索引计算方式
5. 重新导出ONNX并转换为OM
6. 使用msame验证

### Root Cause

**错误的索引计算**:
```python
# 原始错误代码
index = batch * HW + y * nx + x  # 索引计算错误
```

**问题分析**:
- 索引计算使用了错误的维度顺序
- 导致索引值超出有效范围
- ScatterElements算子在NPU上执行时检测到越界

### Fix

**修复后的索引计算**:
```python
# 修复后的代码
indices = (
    coords[:, 2].long() * nx
    + coords[:, 3].long()
).to(torch.int32)
```

**关键修改**:
- 使用正确的coords列索引
- 确保索引值在有效范围内
- 转换为int32类型

### Verification
修复后测试结果:
```
Inference average time: 17.257000 ms
Max diff: 0.0
Mean diff: 0.0
```

每个模块都可以保证OM与ONNX在相同输入的情况下得到近似的输出。

### Lessons & Notes
- **索引计算要精确**: 确保索引值在有效范围内
- **维度顺序一致性**: 索引计算要与数据维度顺序一致
- **NPU算子限制**: NPU对索引越界检查更严格
- **调试方法**:
  - 使用msame工具单独测试每个模块
  - 对比ONNX和OM的输出
  - 检查索引计算公式
- **数据类型**: 注意索引的数据类型要求(int32)
