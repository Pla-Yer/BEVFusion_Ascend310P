# Roadmap

> 本文档记录项目路线图和里程碑。
> 
> 最后更新: 2026-04-02

## Version Milestones

### v0.1 - Architecture Design (2026-02-28)

**Status**: ✅ Completed

**Goals**:

- [x] 项目架构设计
- [x] 技术选型确定
- [x] 文档框架搭建
- [x] 开发环境配置

**Deliverables**:

- 系统架构文档
- 决策日志
- 风险登记册
- 模块状态追踪

**Acceptance Criteria**:

- ✅ 架构设计文档评审通过
- ✅ 技术选型确定
- ✅ 开发环境可用

---

### v0.2 - CPU Modules (2026-03-07)

**Status**: ✅ Completed

**Goals**:

- [x] Voxelization CPU实现
- [x] Pillar Encoder OM模型转换
- [x] NMS CPU实现
- [x] 端到端推理流程搭建

**Deliverables**:

- Voxelization CPU实现
- Pillar Encoder OM模型
- NMS CPU实现
- 端到端推理流程
- 性能分析报告(PERF-001)
- 精度评估报告(EVAL-003)

**Acceptance Criteria**:

- ✅ Voxelization功能正确
- ✅ Pillar Encoder精度验证通过
- ✅ 端到端推理流程可用
- ✅ 性能满足要求(FPS > 5)
- ⚠️ 精度待优化(mAP: 0.19 vs 0.23)

---

## Detailed Timeline

### March 2026

| Week | Tasks                            | Status |
| ---- | -------------------------------- | ------ |
| W1   | Voxelization 实现                  | ✅ 完成   |
| W2   | Voxelization 测试 + Pillar Encoder | ✅ 完成   |
| W3   | Pillar Encoder 完成 + NMS          | ✅ 完成   |
| W4   | 全流程性能优化（BEVPool+并行化+流水线）  | ✅ 完成 |
| W5   | 模型量化实验（AMCT PTQ）  | ✅ 完成 |

### April 2026

| Week | Tasks                  | Status |
| ---- | ---------------------- | ------ |
| W1   | 量化策略优化与重新设计    | 🔄 进行中 |
| W2   | Image Backbone OM      | 待开始 |
| W3   | Detection Head ONNX/OM | 待开始 |
| W4   | BEV Fusion 模块          | 待开始 |

### May 2026

| Week | Tasks |
| ---- | ----- |
| W1   | 端到端集成 |
| W2   | 精度验证  |
| W3   | 精度调优  |
| W4   | 性能优化  |

### June 2026

| Week | Tasks |
| ---- | ----- |
| W1   | 文档完善  |
| W2   | 代码审查  |
| W3   | 最终测试  |
| W4   | 项目交付  |

---

## Dependencies

```
v0.1 ─────► v0.2 ─────► v0.3 ─────► v0.4 ─────► v0.5 ─────► v1.0
 │           │           │           │           │           │
 │           │           │           │           │           │
 ▼           ▼           ▼           ▼           ▼           ▼
架构设计    CPU模块     NPU模块    精度验证    性能优化    交付
```

**关键依赖**:

- v0.2 必须完成才能开始 v0.3（需要 CPU 模块输出）
- v0.3 必须完成才能开始 v0.4（需要 NPU 模块）
- v0.4 精度达标才能进入 v0.5

---

## Risk Mitigation Timeline

| Risk            | Mitigation   | Timeline |
| --------------- | ------------ | -------- |
| SparseConv3d 迁移 | Pillar 替代方案  | v0.2     |
| NPU 算子不支持       | CPU fallback | v0.3     |
| 精度下降            | 微调优化         | v0.4     |
| 性能不达标           | 专项优化         | v0.5     |

---

## Success Metrics

| Metric    | v0.2  | v0.3    | v0.4    | v0.5    | v1.0    |
| --------- | ----- | ------- | ------- | ------- | ------- |
| CPU 模块完成度 | 100%  | -       | -       | -       | -       |
| NPU 模块完成度 | -     | 100%    | -       | -       | -       |
| mAP       | 0.19  | -       | 0.2137  | > 0.50  | > 0.50  |
| NDS       | -     | -       | > 0.55  | > 0.55  | > 0.55  |
| Latency   | 425ms | < 300ms | 166ms   | < 150ms | < 150ms |
| FPS       | 1.9   | > 3     | 6.01    | > 6     | > 6     |

---

## Notes

- 时间线可能根据实际情况调整
- 每个版本结束后进行评审
- 风险触发时启动应急预案
