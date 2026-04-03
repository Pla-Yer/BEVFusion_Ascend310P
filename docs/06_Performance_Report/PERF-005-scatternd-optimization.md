## PERF-005: ScatterNdUpdate算子优化

### Date
2026-03-11

### Configuration
- Hardware: Ascend 310P
- Software: CANN 8.3.RC1
- Model: BEVFusion LiDAR Branch (Pillar Encoder)
- Input Size: 6000 voxels * 32 points

### Initial Metrics
| Component       | Latency (ms) | % of Total | Status |
| --------------- | ------------ | ---------- | ------ |
| stageE (head)   | 40.752490    | 16.56%     | ✅      |
| stageD_neck     | 2.630        | 1.07%      | ✅      |
| stageC_backbone | 5.755        | 2.34%      | ✅      |
| stageB_scatter  | 18.04        | 7.33%      | ✅      |
| stageA_encoder  | 178.9        | **72.69%** | ⚠️     |
| **NPU Total**   | **246.077**  | **100%**   | ✅      |

### Bottleneck Analysis

通过msprof工具对stageA进行分析，运行10次，发现ScatterNdUpdate算子是主要瓶颈：

| opType          | accCore | count | totalTime  | avgTime | maxTime  | minTime  |
| --------------- | ------- | ----- | ---------- | ------- | -------- | -------- |
| ScatterNdUpdate | AI_CORE | 33    | 1665701.46 | 50475.8 | 52459.08 | 48658.56 |
| Transpose       | AI_CORE | 66    | 130345.8   | 1974.94 | 3864.48  | 1141.18  |
| Cast            | AI_CORE | 99    | 39629.31   | 400.3   | 1703.56  | 2.08     |

**ScatterNdUpdate算子占比超过80%**

**根因分析**：在添加特征pillar center时，索引赋值操作被映射为ScatterNd算子：

```python
# 这几行是罪魁祸首
f_center[:, :, 0] = features[:, :, 0] - (...)  # → ScatterND
f_center[:, :, 1] = features[:, :, 1] - (...)  # → ScatterND
f_center[:, :, 2] = features[:, :, 2] - (...)  # → ScatterND
```

这里的"="会被映射为ScatterNd算子，会对6000*32的位置进行3次scatter操作。

### Optimization

**优化策略**：将索引赋值操作变为拼接操作

```python
# 优化前：索引赋值 → ScatterNd
f_center[:, :, 0] = features[:, :, 0] - (...)
f_center[:, :, 1] = features[:, :, 1] - (...)
f_center[:, :, 2] = features[:, :, 2] - (...)

# 优化后：拼接操作 → Sub + Stack
f_center_x = features[:, :, 0] - (...)
f_center_y = features[:, :, 1] - (...)
f_center_z = features[:, :, 2] - (...)
f_center = torch.stack([f_center_x, f_center_y, f_center_z], dim=-1)
```

将ScatterNd算子替换为`Sub` + `Stack`（`Unsqueeze/Concat`）

### Results

**优化后算子耗时**：

| opType                  | accCore        | count | totalTime | avgTime |
| ----------------------- | -------------- | ----- | --------- | ------- |
| Transpose               | AI_CORE        | 66    | 130412.7  | 1975.95 |
| Cast                    | AI_CORE        | 99    | 39541.39  | 399.41  |
| BNInferenceD            | AI_CORE        | 33    | 34090.27  | 1033.04 |
| ConcatD                 | AI_CORE        | 44    | 20147.3   | 457.89  |

**ScatterNdUpdate算子已消除**

**优化后模型各阶段耗时**：

| Component       | Latency (ms) | % of Total | Status |
| --------------- | ------------ | ---------- | ------ |
| stageE (head)   | 40.752490    | 37.67%     | ✅      |
| stageD_neck     | 2.630        | 2.43%      | ✅      |
| stageC_backbone | 5.755        | 5.32%      | ✅      |
| stageB_scatter  | 18.04        | 16.68%     | ✅      |
| stageA_encoder  | 41           | 37.90%     | ✅      |
| **NPU Total**   | **108.177**  | **100%**   | ✅      |

**性能提升**：
- stageA_encoder: 178.9ms → 41ms (降低77%)
- NPU Total: 246.077ms → 108.177ms (降低56%)

**精度影响**：无精度下降

### Other Attempts

尝试了以下优化方法，但未实现更优：
1. MatMul替代Scatter
2. index_add → ScatterAdd

### Notes
- 该优化是NPU算子优化的典型案例，展示了索引操作在NPU上的性能问题
- 优化后精度保持不变，性能提升显著
- 后续可考虑将类似索引赋值操作统一优化为拼接操作
