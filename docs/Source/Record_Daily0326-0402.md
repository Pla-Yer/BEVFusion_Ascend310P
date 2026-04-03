# 日常记录 (2026-03-26 ~ 2026-04-02)

## 2026-03-30 量化实验总结

### 什么是量化

量化就是指对模型的权重（weight）和数据（activation）进行低比特处理，让最终生成的网络模型更加轻量化，从而达到节省网络模型存储空间、降低传输时延、提高计算效率，达到性能提升与优化的目标。

量化的本质就是**把连续实数映射到有限个离散值**。

量化的对象主要是权重（weight），数据（activation），对于LLM，也常见KV量化等。

**权重量化**：将模型参数从高精度（FP32，FP16）转化为低精度（INT8，INT4）

**数据量化**：将模型的中间输出进行量化，也叫激活量化，但数据量化关注数据在模型中运行时的变化，激活量化更聚焦于层输出；也就是说激活量化是数据量化的一个部分。

### 量化的关键

量化是为部署落地服务的，那么一定会与硬件高度关联。量化的关键就在于**硬件适配**。量化是数学近似，但能否真正变成性能收益，是硬件问题。

模型数据在硬件的流程：

1. **存储**：权重怎么排布在显存/内存里
2. **搬运**：怎么从内存搬到寄存器/计算单元
3. **计算**：乘加单元能不能直接算 INT8/INT4/BF16
4. **输出**：结果要不要反量化、重排、再转换精度

对于存储，主要需要关注硬件对数据格式的支持：

- CPU：FP32、FP16、INT8 常较成熟，INT4 往往要看指令集和内核实现
- GPU：BF16/FP16 支持通常很好，INT8 常有专门加速，INT4/FP8 要看代际
- NPU/ASIC：往往专门为 INT8、INT4、混合精度设计
- DSP/MCU：更偏定点数和低功耗整数运算

对于搬运，主要需要关注硬件对数据搬运时的实际策略，例如大多硬件为了减少搬运，会将多数据拼在一起，这也是硬件的适配性。

对于计算，需要关注硬件方提供的高性能算子是否支持量化的格式，如果不支持，那么可能导致直接不能推理，或者从高性能算子变为低性能算子的组合导致性能反而下降

### ASCEND AMCT量化

AMCT（Ascend Model Compression Toolkit，简称AMCT）是一个针对昇腾芯片亲和的深度学习模型压缩工具包，提供量化、张量分解等多种模型压缩特性，压缩后模型体积变小，部署到昇腾AI处理器上后可使能低比特运算，提高计算效率，达到性能提升的目标。

训练后量化流程：

```
原始模型 -> 校准数据生成 -> 量化配置 -> 量化模型 -> 精度验证 -> 部署
```

这里选择onnx量化，onnx是使用范围最广的IR，在很多地方都用应用于适配，而且在本项目中，模型转化路径为pth->onnx->om，既有先验的实验与结果可以更好地使用。

AMCT支持的量化层：

| 支持的层类型              | 约束                                                                                                                                                                                                                                                  |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Conv：卷积层            | 支持4维或5维输入，且权重类型为constant或Initializer的量化。                                                                                                                                                                                                            |
| Gemm：广义矩阵乘          | transA=false、alpha=beta=1.0；权重类型为constant或Initializer。                                                                                                                                                                                              |
| MatMul：全连接层         | - 当第二路输入为常量const时，仅支持2-3维；权重类型为constant或Initializer。<br>- 当两路输入都为变量tensor时，仅支持INT8对称量化。两路输入都为变量tensor量化场景，只在如下产品型号能获得收益，其他产品型号量化后精度会下降： Atlas 200I/500 A2 推理产品 Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件 Atlas A3 训练系列产品/Atlas A3 推理系列产品 |
| ConvTranspose：转置卷积层 | 仅支持4维输入，且权重类型为constant或Initializer的量化。                                                                                                                                                                                                              |
| AveragePool：平均池化层   | 仅支持4维输入场景下的量化。                                                                                                                                                                                                                                      |
| MaxPool：最大池化层       | 只做tensor量化。                                                                                                                                                                                                                                         |
| Add：按元素求和层          | 只做tensor量化。                                                                                                                                                                                                                                         |

### 实验结果

通过脚本将3个onnx模型进行均匀量化，并进行分析（将量化后的onnx模型通过atc工具转化为om模型，并将单个量化后的om模型加入推理）

#### camera分支

camera分支的量化是很成功的，模型大小从100M->27M，推理耗时从55ms->45ms，而且精度几乎没变。

#### lidar分支

lidar分支大小没有变（133MB->132MB），精度反而下降(0.21->0.17)

#### fusion分支

fusion分支在atc时出现下面的问题：

1. op[SwinAttentionScoreFusionPass], The dims of bmm1_node input0 should be greater than 3.[FUNC:Fusion][FILE:swin_attention_score_fusion_pass.cc][LINE:263]

2. requant_/decoder.0/self_posembed/position_embedding_head/position_embedding_head.0/Conv.dequantset op json path failed[FUNC:ProcessSuccCompileTask][FILE:tbe_op_store_adapter.cc][LINE:1527]

**问题分析**：

- **`SwinAttentionScoreFusionPass` 触发失败**：ATC 的 Swin Attention 融合 pass 尝试融合 Decoder 里的 BMM 算子，但该 pass 要求输入维度 >3，而 BEVFusion Decoder 的 attention 张量是 3D 的，导致 pass 直接报错并中止编译。
- **`requant_/.../Conv.dequant` 编译失败**：Decoder 中 `position_embedding_head` 和 `prediction_heads` 的 Conv 被量化后，对应的 `AscendDequant` 算子在 310P 上无法编译（特定 shape/channel 组合不支持）。

**根因分析**：

**报错 1：`SwinAttentionScoreFusionPass` — dims should be greater than 3**

ATC 的图融合 pass 在优化时，把 Decoder 里的 BMM（批量矩阵乘）误判为 Swin Transformer 的 attention score 结构，尝试执行融合，但该 pass 要求输入必须是 4D 张量。BEVFusion Decoder 的 attention 张量是 3D 的 `[B, seq_len, dim]`，维度不满足，直接报错并终止整个编译。

**修复**：生成 `fusion_switch.json`，通过 ATC 的 `--fusion_switch_file` 参数关闭三个相关 pass。

**报错 2：`requant_/.../Conv.dequant` — Json path empty**

`position_embedding_head` 和 `prediction_heads` 里的 Conv 被量化后，AMCT 插入了 `AscendDequant` 算子，但 TBE 编译器在 310P 上无法为这些特定 shape/channel 组合生成对应的 kernel。

**修复**：量化时将这些层自动加入 `skip_layers`，保持 FP16，避免生成问题算子。

fusion则是在没有修复atc失败前，模型大小优化较大（28MB->5mb），修复之后（25MB->9MB），但是精度下降了一半（0.21->0.12），推理耗时反而增加（45ms->50ms）。

### 问题分析

**Camera** 是纯 CNN 结构，AMCT 的量化支持层（Conv/Gemm）能覆盖绝大部分参数，校准也稳定，所以压缩率高、精度几乎无损。**LiDAR** 和 **Fusion** 的失败原因则完全不同，需要分开拆解。

#### LiDAR 分支：两个独立根因叠加

**根因 A：动态轴导致 IFMR 校准失效**

VFE 的输入第 0 维是 voxel 数量 V，每帧推理时 V 不同（3000 ～ 12000 不等）。AMCT 的 IFMR 激活校准算子在统计激活分布时，是在固定的 `--max-voxels=10000` 的 dummy 数据上跑的。但实际 NPU 推理时 V 会变化，激活的绝对数值范围随 V 变化明显，导致校准时确定的 `scale` 和 `offset` 在实际推理中并不准确，INT8 误差超出预期，精度从 0.21 跌到 0.17。

**根因 B：这个分支根本没有可压缩的大型参数**

整个 LiDAR 分支只包含两个子模块：PillarVFE（3 个 Linear 层，参数量极少，5→64→64 维）和 OnnxPointPillarsScatter（纯 Gather/Scatter 索引操作，**没有任何可学习参数**）。AMCT 的量化支持层列表是 Conv/Gemm/MatMul/ConvTranspose/AveragePool/MaxPool/Add，而 Scatter 完全不在其中，无法压缩。实际上 BEVFusion LiDAR 管线里的"重头"——SECOND 骨干网络（多个 SparseConv）和 SecFPN——全都在 Fusion Head 里，LiDAR 分支本身几乎没有大型 Conv 参数，所以 133MB → 132MB 是正常的，不是 bug。

---

#### Fusion Head：三个根因

**根因 C：精度暴跌至 0.12 的真正原因**

量化脚本跳过了（skip）Decoder 里的双变量 MatMul 和 prediction_heads 的 Conv，这没有问题。问题在于被量化的部分：SECOND 骨干和 SecFPN 被成功量化为 INT8，它们的输出——也就是送入 Transformer Decoder 做 cross-attention 的 BEV 特征——已经带有 INT8 量化误差。Decoder 的 cross-attention 对输入 key/value 的精度极其敏感，轻微的特征扰动会被 softmax 指数放大，导致 attention weight 分布发生偏移，最终 proposal 质量大幅下降，mAP 减半。简而言之：**量化了不该量化的上游特征，而不是不该量化的下游头部**。

**根因 D：延迟反增的原因是混合精度边界惩罚**

跳过 Decoder 和 prediction_heads 后，图里存在大量 `INT8（骨干）→ FP16（Decoder）→ INT8（某些 BEV 计算）→ FP16（预测头）` 的精度切换边界。每个边界都需要 ATC 插入 `AscendQuant`/`AscendDequant` 算子，这些算子在 310P 的 AICore 上需要额外的 DDR 读写和类型转换指令。当精度边界数量很多、转换开销累计超过 INT8 推理节省的计算量时，总延迟不降反升。

**根因 E：这个分支的结构决定了碎片化量化必然失败**

Fusion Head 的推理瓶颈不在 SECOND/SecFPN 的 Conv，而在 Transformer Decoder 的自注意力和交叉注意力（多层 decoder 迭代）。恰好这部分在 310P 上最不适合量化（双变量 MatMul 无收益），跳过它等于放弃了最关键的加速目标，却保留了引入精度边界惩罚的代价。

---

#### 根本矛盾总结

|         | LiDAR                                          | Fusion Head                              |
| ------- | ---------------------------------------------- | ---------------------------------------- |
| 本质矛盾    | 分支内没有大参数，核心算子（Scatter）无法量化                     | 推理瓶颈（Decoder）恰好是最不能量化的部分                 |
| 精度损失来源  | 动态 shape 导致 IFMR 校准 scale 偏差                   | 骨干量化误差在 Decoder attention 中被放大           |
| 延迟问题来源  | —                                              | INT8/FP16 混合精度边界的 Quant/Dequant 开销       |
| 正确的解决方向 | 对 LiDAR 骨干（在 Fusion Head 里）整体量化；LiDAR 分支本身意义不大 | 要么全量化（接受精度损失），要么只量化骨干+跳过 Decoder 并接受延迟不变 |

### 结论与后续方向

1. **Camera分支量化成功**：可以继续优化，探索更低位宽（INT4）量化
2. **LiDAR分支不适合单独量化**：应该将LiDAR骨干网络（在Fusion Head中）整体量化
3. **Fusion分支需要重新设计量化策略**：
   - 方案A：全量化，接受精度损失
   - 方案B：只量化骨干网络，跳过Decoder，接受延迟不变
   - 方案C：探索其他量化方法（如QAT量化感知训练）

### 技术收获

1. 深入理解了量化原理和硬件适配的重要性
2. 掌握了AMCT量化工具的使用方法
3. 识别了不同网络结构对量化的适应性差异
4. 理解了混合精度边界的性能开销机制
5. 积累了量化失败问题的排查和修复经验
