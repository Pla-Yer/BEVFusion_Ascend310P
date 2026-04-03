## D-005: 模型量化策略

### Date
2026-03-12

### Background
在增加体素化数量上限后，模型推理耗时从108ms增加到132ms。为了弥补性能损失，考虑对模型进行量化，将精度从force_32降低到force_16。

### Options Considered
| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A. 保持FP32 | 维持force_32精度 | 精度最高，无精度损失 | 推理速度较慢(132ms) |
| B. 量化到FP16 | 使用force_fp16精度 | 推理速度快(102ms)，精度损失极小 | 可能存在微小精度损失 |

### Choice
**Option B: 量化到FP16**

### Reason
1. 性能提升显著：推理耗时从132ms降低到102ms，提升22.7%
2. 精度损失极小：mAP从0.219仅下降到0.218，损失可忽略
3. 全流程pipeline优化：从0.25s降低到0.22s
4. FP16在Ascend 310P上有良好的硬件支持

### Risk
- FP16可能引入数值精度问题
- 某些算子可能对FP16支持不完善

### Mitigation
- 使用`--precision_mode_v2=force_fp16`参数确保量化正确执行
- 对比量化前后精度，确保损失在可接受范围
- 监控模型运行稳定性

### Outcome
| Metric | Before (FP32) | After (FP16) | Change |
|--------|---------------|--------------|--------|
| 模型推理耗时 | 132ms | 102ms | -30ms |
| 全流程pipeline | 0.25s | 0.22s | -0.03s |
| mAP | 0.219 | 0.218 | -0.001 |

量化效果显著，性能提升明显，精度损失可忽略，决策有效。

### Notes
- ATC命令参数：`--precision_mode_v2=force_fp16`
- 参考文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/devaids/atctool/atlasatcparam_16_0069.html
