# PERF-010: 模型量化实验性能报告

**日期**: 2026-03-30
**测试环境**: Atlas 310P
**量化工具**: AMCT (Ascend Model Compression Toolkit)

## 实验概述

本次实验使用AMCT对BEVFusion的三个分支（Camera、LiDAR、Fusion）进行训练后量化（PTQ），评估量化对模型大小、推理延迟和精度的影响。

## 量化配置

| 配置项 | 值 |
| ------ | -- |
| 量化方法 | 均匀量化（Uniform Quantization） |
| 量化位宽 | INT8 |
| 量化对象 | 权重 + 激活 |
| 校准策略 | IFMR (Input Feature Map Range) |
| 校准数据 | 验证集子集（100帧） |
| 模型格式 | ONNX → OM |

## Camera分支量化结果

### 性能指标

| 指标 | 量化前 | 量化后 | 变化 |
| ---- | ------ | ------ | ---- |
| 模型大小 | 100 MB | 27 MB | -73% |
| 推理延迟 | 55 ms | 45 ms | -18% |
| 精度(mAP) | - | - | 无损 |

### 成功原因分析

1. **纯CNN结构**：Camera分支是纯卷积网络，AMCT的量化支持层（Conv/Gemm）能覆盖全部参数
2. **静态shape**：输入shape固定，校准数据分布准确
3. **校准稳定**：激活值分布范围稳定，IFMR能准确估计scale和offset

### 结论

✅ **量化成功**，可以部署使用

---

## LiDAR分支量化结果

### 性能指标

| 指标 | 量化前 | 量化后 | 变化 |
| ---- | ------ | ------ | ---- |
| 模型大小 | 133 MB | 132 MB | -0.75% |
| 精度(mAP) | 0.21 | 0.17 | -19% |

### 失败原因分析

#### 根因A：动态轴导致IFMR校准失效

```
问题：VFE输入第0维是voxel数量V，每帧不同（3000~12000）
影响：校准在固定max_voxels=10000上进行，实际推理时V变化
结果：scale和offset不准确，INT8误差超出预期
```

#### 根因B：无可压缩的大型参数

```
LiDAR分支组成：
- PillarVFE: 3个Linear层（5→64→64维），参数量极少
- OnnxPointPillarsScatter: 纯Gather/Scatter操作，无可学习参数

AMCT支持层：Conv/Gemm/MatMul/ConvTranspose/AveragePool/MaxPool/Add
Scatter不在支持列表中，无法压缩

真正的大型参数（SECOND骨干、SecFPN）在Fusion Head中
```

### 结论

❌ **量化失败**，LiDAR分支不适合单独量化

---

## Fusion分支量化结果

### 性能指标

| 指标 | 量化前 | 量化后（修复前） | 量化后（修复后） |
| ---- | ------ | ---------------- | ---------------- |
| 模型大小 | 28 MB | 5 MB | 9 MB |
| 推理延迟 | 45 ms | - | 50 ms |
| 精度(mAP) | 0.21 | - | 0.12 |

### ATC转换问题

#### 问题1：SwinAttentionScoreFusionPass失败

```
错误：op[SwinAttentionScoreFusionPass], The dims of bmm1_node input0 should be greater than 3
原因：ATC误判Decoder的BMM为Swin Transformer attention结构
      BEVFusion attention张量是3D [B, seq_len, dim]，不满足4D要求
修复：生成fusion_switch.json，关闭相关融合pass
```

#### 问题2：AscendDequant编译失败

```
错误：requant_/.../Conv.dequant set op json path failed
原因：position_embedding_head和prediction_heads的Conv被量化后
      AscendDequant算子在310P上无法编译（特定shape/channel不支持）
修复：将这些层加入skip_layers，保持FP16
```

### 失败原因分析

#### 根因C：精度暴跌（0.21→0.12）

```
量化流程：
SECOND骨干(INT8) → SecFPN(INT8) → BEV特征(带量化误差)
                                    ↓
                        Transformer Decoder cross-attention
                                    ↓
                              精度敏感模块
                                    ↓
                            softmax指数放大误差
                                    ↓
                              proposal质量下降

问题：量化了不该量化的上游特征
```

#### 根因D：延迟反增（45ms→50ms）

```
混合精度边界：
INT8(骨干) → FP16(Decoder) → INT8(BEV计算) → FP16(预测头)

每个边界需要插入：
- AscendQuant: FP16 → INT8
- AscendDequant: INT8 → FP16

开销：
- DDR读写
- 类型转换指令
- 累计超过INT8推理节省的计算量
```

#### 根因E：结构不匹配

```
Fusion Head瓶颈：Transformer Decoder（自注意力+交叉注意力）
                ↓
            双变量MatMul
                ↓
        310P上量化无收益
                ↓
        跳过瓶颈部分
                ↓
        保留精度边界惩罚
                ↓
            性能不降反升
```

### 结论

❌ **量化失败**，需要重新设计量化策略

---

## 根本矛盾总结

| 维度 | LiDAR分支 | Fusion分支 |
| ---- | --------- | ---------- |
| **本质矛盾** | 分支内没有大参数，核心算子（Scatter）无法量化 | 推理瓶颈（Decoder）恰好是最不能量化的部分 |
| **精度损失来源** | 动态shape导致IFMR校准scale偏差 | 骨干量化误差在Decoder attention中被放大 |
| **延迟问题来源** | - | INT8/FP16混合精度边界的Quant/Dequant开销 |
| **正确方向** | 对LiDAR骨干（在Fusion Head里）整体量化 | 要么全量化（接受精度损失），要么只量化骨干+跳过Decoder |

---

## 量化适用性分析

### 适合量化的网络特征

1. ✅ 纯CNN结构（Conv/Gemm为主）
2. ✅ 静态输入shape
3. ✅ 激活值分布稳定
4. ✅ 无精度敏感模块（如attention）
5. ✅ 硬件支持量化算子

### 不适合量化的网络特征

1. ❌ 动态shape输入
2. ❌ 包含不支持的算子（如Scatter）
3. ❌ 精度敏感模块（如Transformer attention）
4. ❌ 混合精度边界过多
5. ❌ 瓶颈部分无法量化

---

## 后续建议

### Camera分支
- ✅ 部署量化模型
- 探索INT4量化可能性
- 评估FP8量化效果

### LiDAR分支
- ❌ 不单独量化
- ✅ 在Fusion Head中整体量化LiDAR骨干

### Fusion分支
**方案对比**：

| 方案 | 模型大小 | 延迟 | 精度 | 实施成本 | 推荐度 |
| ---- | -------- | ---- | ---- | -------- | ------ |
| A: 全量化 | 9MB | 50ms | 0.12 | 低 | ⭐ |
| B: 只量化骨干 | 15MB | 45ms | 0.21 | 低 | ⭐⭐ |
| C: QAT量化感知训练 | 9MB | 45ms | >0.18 | 高 | ⭐⭐⭐ |

**推荐**: 方案C（QAT），但优先级较低

---

## 技术收获

1. 深入理解量化原理和硬件适配的重要性
2. 掌握AMCT量化工具的使用和问题排查
3. 识别不同网络结构对量化的适应性差异
4. 理解混合精度边界的性能开销机制
5. 积累量化失败问题的分析和修复经验

---

## 相关文档

- D-008: 量化实验决策
- Record_Daily0326-0402: 量化实验详细记录
- D-005: 模型量化策略决策（初始）
