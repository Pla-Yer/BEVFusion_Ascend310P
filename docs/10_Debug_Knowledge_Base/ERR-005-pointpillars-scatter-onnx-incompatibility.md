## ERR-005: PointPillarsScatter ONNX不兼容

### Date
2026-03-05

### Environment
- OS: Linux
- Component: MiddleEncoder (PointPillarsScatter)
- Source: mmdet3d原始实现
- Tool: ONNX export

### Symptom
MiddleEncoder在ONNX导出后,输出与PyTorch模型存在巨大差异(max|Δ|=21),导致BEV特征图错误。

### Debug Process
1. 通过`debug/diagnose_layer.py`定位到MiddleEncoder问题
2. 创建`debug/inspect_middle_encoder.py`脚本获取MiddleEncoder实现
3. 分析PointPillarsScatter源码
4. 识别ONNX不兼容的代码模式
5. 设计ONNX兼容的替代方案

### Root Cause

**mmdet3d原始实现 - ONNX不兼容**:
```python
# mmdet3d 原始代码 —— 两个ONNX致命问题
canvas = torch.zeros(...)
for i in range(batch_size):                          # ❌ 问题1: for循环,batch_size被trace为常量
    mask = (coors[:, 0] == i)
    canvas[i, :, coors_y[mask], coors_x[mask]] = \
        voxel_features[mask].T                       # ❌ 问题2: boolean mask + in-place index put
```

**问题分析**:
1. **for循环问题**: batch_size在ONNX trace时被固化,无法动态调整
2. **in-place操作**: boolean mask + in-place index赋值在ONNX中不支持
3. 这两点导致ONNX Runtime拿到的是构建逻辑被固化错误的BEV图

### Fix

**修复方案: 使用scatter_add重写**:
```python
# 新实现 —— ONNX opset11原生支持
linear_idx = batch_id * H * W + y * W + x          # 全局线性索引,无for循环
canvas_flat.scatter_add(0, idx_expanded, voxel_features)  # 单次操作完成所有batch
```

**关键改进**:
- 消除for循环,使用全局线性索引
- 使用scatter_add替代in-place操作
- ONNX opset11原生支持scatter操作

### Verification
修复后:
- MiddleEncoder输出与PyTorch模型匹配
- Stage2验证结果变为✅ PASS
- BEV特征图正确生成

### Lessons & Notes
- **ONNX兼容性原则**:
  - 避免for循环,使用向量化操作
  - 避免in-place操作,使用函数式操作
  - 使用ONNX原生支持的算子
- **scatter操作**: scatter_add是ONNX兼容的等价写法
- **调试方法**: 使用inspect脚本分析具体实现,识别不兼容模式
- **参考**: ONNX opset11及以上版本支持更多操作
