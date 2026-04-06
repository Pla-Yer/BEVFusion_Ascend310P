## ERR-009: 点云第5维数据错误导致mAP为0

### Date
2026-03-05

### Environment
- OS: Linux
- Component: 点云数据加载
- Data: NuScenes .bin激光雷达文件
- Model: BEVFusion

### Symptom
将OM模型放到bevfusion_evaluator中推理,结果全为0:
```
mAP: 0.0000
NDS: 0.0000
```

所有检测结果的置信度都极低,无法检测到任何目标。

### Debug Process
1. 验证OM模型各模块输出正确
2. 验证decoder输出基本一致
3. 检查点云数据加载流程
4. 分析NuScenes数据格式
5. 对比MMDetection3D/BEVFusion的数据要求
6. 发现第5维数据不匹配

### Root Cause

**NuScenes原始数据格式**:
```
NuScenes .bin激光雷达文件的5个维度: [x, y, z, intensity, ring_index]
```

**MMDetection3D/BEVFusion要求**:
```
官方单帧训练中,第5维是timestamp(单帧时全部为0.0)
```

**问题分析**:
- 直接截取了前5维送入网络
- 网络读到了ring_index(取值0~31的大整数)
- 被误当作极其异常的timestamp伪影
- 破坏了特征分布
- 导致网络全部输出极低置信度

### Fix

**修复方案: 强制将点云第5列置为0.0**:
```python
# 加载点云数据
points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5)
# 强制将第5列(timestamp)置为0.0
points[:, 4] = 0.0
```

**关键修改**:
- 确保第5维数据符合模型预期
- 单帧场景下timestamp应为0.0
- 避免异常数据破坏特征分布

### Verification
修复后成功完成推理,结果如下:
```
mAP: 0.1193
mATE: 0.7489
mASE: 0.8755
mAOE: 1.1282
mAVE: 1.4359
mAAE: 1.0000
NDS: 0.0972
```

### Lessons & Notes
- **数据格式至关重要**: 必须确保数据格式与模型预期完全一致
- **理解数据语义**: 不同数据集的维度语义可能不同
- **单帧vs多帧**: 单帧场景下某些维度(如timestamp)需要特殊处理
- **调试方法**:
  - 检查数据加载流程
  - 对比数据格式要求
  - 验证数据取值范围
- **文档重要性**: 明确记录数据格式要求和处理方式
- **异常值影响**: 异常数据会严重破坏模型性能
