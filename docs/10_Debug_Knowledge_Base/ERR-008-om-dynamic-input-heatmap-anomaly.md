## ERR-008: OM动态输入导致heatmap异常集中在(0,0)

### Date
2026-03-05

### Environment
- OS: Linux
- Component: OM模型推理
- Tool: ATC converter
- Hardware: Ascend 310P
- Reference: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/devaids/atctool/atlasatcparam_16_0020.html

### Symptom
对比OM与ONNX的heatmap输出,发现:
- OM的heatmap完全集中在(0,0)处
- OM的heatmap数值(100+)比ONNX的heatmap数值(10+)大得多
- 其他位置的特征被掩盖

### Debug Process
1. 对比OM与ONNX的heatmap输出,发现异常
2. 初步怀疑是精度问题,尝试在ATC时增加`--precision_mode=force_fp32`,但未解决
3. 研究昇腾资料,了解OM模型动态输入的机制
4. 分析动态输入对scatter操作的影响
5. 设计掩码方案解决问题

### Root Cause

**OM动态输入机制**:
- OM模型的动态输入本质上是帮助自动实现补0到确定大小
- 这会导致在scatter时,有许多无效特征被放在(0,0)位置
- (0,0)位置累积了大量无效特征,数值异常大
- 掩盖了其他位置的有效特征

**问题分析**:
```
动态输入 → 自动补0 → 无效特征集中在(0,0) → heatmap异常
```

### Fix

**修复方案: 在encoder后增加掩码操作**:
```python
# 将坐标为(0,0)处的特征全置为0
# 这样便不会影响后面的过程
mask = (coords[:, 2] == 0) & (coords[:, 3] == 0)
features[mask] = 0
```

**权衡考虑**:
- 这样意味着OM模型做了一些没用的计算
- 需要后续优化考虑
- 但可以保证功能正确性

### Verification
修复后:
- OM与ONNX的heatmap输出一致
- 特征分布正确
- 不再集中在(0,0)位置

### Lessons & Notes
- **动态输入机制**: 理解框架的动态输入实现机制很重要
- **无效数据处理**: 动态补0可能引入无效数据,需要处理
- **性能权衡**: 功能正确性优先,性能优化可以后续进行
- **调试方法**:
  - 对比不同框架的输出
  - 分析异常数据的来源
  - 查阅官方文档理解机制
- **优化方向**: 后续可以考虑优化动态输入的处理方式,减少无效计算
