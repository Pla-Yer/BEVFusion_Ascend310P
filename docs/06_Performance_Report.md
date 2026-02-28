# Performance Report

> 本文档记录系统性能分析和优化结果。
>
> 最后更新: 2026-02-28
>
> 目标: 端到端延迟 < 200ms

---

## Performance Targets

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| End-to-End Latency | < 200ms | TBD | ⏸️ |
| CPU Latency | < 50ms | TBD | ⏸️ |
| NPU Latency | < 150ms | TBD | ⏸️ |
| Memory Usage | < 8GB | TBD | ⏸️ |
| FPS | > 5 | TBD | ⏸️ |

---

## Latency Breakdown

### Overall Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    End-to-End Latency                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ CPU Stage 1: Data Loading                           │    │
│  │ Latency: TBD ms                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ CPU Stage 2: Voxelization                           │    │
│  │ Latency: TBD ms                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ CPU Stage 3: Pillar Encoder                         │    │
│  │ Latency: TBD ms                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│              ─────────── Data Transfer ───────────          │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ NPU Stage 1: Image Backbone                         │    │
│  │ Latency: TBD ms                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ NPU Stage 2: BEV Fusion                             │    │
│  │ Latency: TBD ms                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ NPU Stage 3: Detection Head                         │    │
│  │ Latency: TBD ms                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│              ─────────── Data Transfer ───────────          │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ CPU Stage 4: NMS                                    │    │
│  │ Latency: TBD ms                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## CPU Latency Analysis

### Voxelization

| Metric | Value | Notes |
|--------|-------|-------|
| Input Size | ~40,000 points | nuScenes average |
| Output Voxels | ~16,000 | max_voxels |
| Latency | TBD ms | |
| CPU Usage | TBD % | |
| Memory | TBD MB | |

**Optimization Opportunities**:
- [ ] 多线程并行处理
- [ ] 向量化优化
- [ ] 内存预分配

### Pillar Encoder

| Metric | Value | Notes |
|--------|-------|-------|
| Input | Voxel Features | [V, 32, 4] |
| Output | Pillar Features | [P, C] |
| Latency | TBD ms | |
| CPU Usage | TBD % | |

**Optimization Opportunities**:
- [ ] 使用 torch.jit.script
- [ ] 批量矩阵运算
- [ ] 特征缓存

### NMS

| Metric | Value | Notes |
|--------|-------|-------|
| Input Detections | ~1000 | before NMS |
| Output Detections | ~50 | after NMS |
| Latency | TBD ms | |
| CPU Usage | TBD % | |

**Optimization Opportunities**:
- [ ] 使用 torchvision.ops.nms
- [ ] 预过滤低分检测
- [ ] 分类别并行 NMS

---

## NPU Latency Analysis

### Image Backbone

| Metric | Value | Notes |
|--------|-------|-------|
| Input | [6, 3, 256, 704] | 6 cameras |
| Output | Multi-scale features | |
| Latency | TBD ms | |
| NPU Usage | TBD % | |

### BEV Fusion

| Metric | Value | Notes |
|--------|-------|-------|
| Input | Pillar + Image Features | |
| Output | Fused BEV Features | |
| Latency | TBD ms | |
| NPU Usage | TBD % | |

### Detection Head

| Metric | Value | Notes |
|--------|-------|-------|
| Input | BEV Features | |
| Output | Raw Detections | |
| Latency | TBD ms | |
| NPU Usage | TBD % | |

---

## Data Transfer Analysis

### CPU → NPU

| Data | Size | Transfer Time | Bandwidth |
|------|------|---------------|-----------|
| Pillar Features | TBD MB | TBD ms | TBD GB/s |
| Camera Images | TBD MB | TBD ms | TBD GB/s |

### NPU → CPU

| Data | Size | Transfer Time | Bandwidth |
|------|------|---------------|-----------|
| Detection Output | TBD MB | TBD ms | TBD GB/s |

**Optimization Opportunities**:
- [ ] 使用共享内存
- [ ] 异步传输
- [ ] 数据预取

---

## Memory Analysis

### CPU Memory

| Component | Size | Notes |
|-----------|------|-------|
| Point Cloud | ~2 MB | Float32 |
| Voxel Features | ~10 MB | Float32 |
| Pillar Features | ~5 MB | Float32 |
| Camera Images | ~50 MB | Float32 |
| Detection Results | ~1 MB | Float32 |
| **Total** | **~70 MB** | |

### NPU Memory

| Component | Size | Notes |
|-----------|------|-------|
| Model Weights | TBD MB | FP16 |
| Intermediate Features | TBD MB | FP16 |
| Input Buffer | TBD MB | FP16 |
| Output Buffer | TBD MB | FP16 |
| **Total** | **TBD MB** | |

---

## Bottleneck Analysis

### Current Bottlenecks

| Rank | Component | Latency | % of Total | xxxxxxxxxx ### R-XXX: [Risk Title]​**Level**: High / Medium / Low​**Impact**: [Describe impact]​**Probability**: High / Medium / Low​**Description**:[Detailed description]​**Mitigation Strategy**:1. [Strategy 1]2. [Strategy 2]​**Contingency Plan**:- [Plan if risk materializes]​**Status**: ⏸️ Monitoring / 🔄 Mitigating / ✅ Resolved​**Last Update**: YYYY-MM-DDmarkdown |
|------|-----------|---------|------------|----------|
| 1 | TBD | TBD ms | TBD % | High |
| 2 | TBD | TBD ms | TBD % | High |
| 3 | TBD | TBD ms | TBD % | Medium |

### Optimization Priority

1. **高优先级**
   - [ ] 识别最大延迟组件
   - [ ] 针对性优化

2. **中优先级**
   - [ ] 数据传输优化
   - [ ] 内存优化

3. **低优先级**
   - [ ] 代码重构
   - [ ] 日志优化

---

## Performance Comparison

### vs GPU Baseline

| Component | GPU Latency | 310P Latency | Ratio |
|-----------|-------------|--------------|-------|
| Voxelization | TBD ms | TBD ms | TBD |
| Backbone | TBD ms | TBD ms | TBD |
| Fusion | TBD ms | TBD ms | TBD |
| Head | TBD ms | TBD ms | TBD |
| NMS | TBD ms | TBD ms | TBD |
| **Total** | **TBD ms** | **TBD ms** | **TBD** |

---

## Profiling Tools

### CPU Profiling

```bash
# Python profiler
python -m cProfile -o profile.stats inference.py

# Memory profiler
python -m memory_profiler inference.py
```

### NPU Profiling

```bash
# Ascend profiling
msprof --output=./profiling_data python inference.py
```

---

## Performance Log

### Perf-001: Initial Profiling
- **Date**: TBD
- **Config**: Baseline
- **Result**: TBD
- **Notes**: TBD

### Perf-002: After CPU Optimization
- **Date**: TBD
- **Config**: Optimized CPU modules
- **Result**: TBD
- **Notes**: TBD

---

## Next Steps

1. [ ] 完成初始性能测试
2. [ ] 识别性能瓶颈
3. [ ] 优化 CPU 模块
4. [ ] 优化 NPU 模块
5. [ ] 优化数据传输
6. [ ] 达成性能目标
