## D-003: Voxel Encoder Om

### Date
2026-03-02

### Background
voxel的数量是不确定,而om模型需要确定的输入大小。通过实验发现,voxel数量通常在6000,多的话8000-1000。

### Options Considered

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A. onnx | onnx在cpu上推理 | 精度最高,不需要将模型转为om | 存在内存搬运增加耗时,并且cpu推理也会增加耗时 |
| B. singal large model | 一个输入为10000的模型,不足补0 | 只需要一个om | 增加大量无效计算 |
| C. multi-model | 多个不同dim的模型,通过判断使用哪个 | 在npu上计算 | 需要将所有模型都加载,但只用了一个 |
| D. dynamic_dims | 动态维度om模型 | 只要一个om模型 | 模型本身比较大,且使用该参数的模型比较复杂 |

### Choice
**Option D: dynamic_dims** (当前阶段)

### Reason
1. 只需要加载单个om模型
2. 保证数据都在npu上
3. 动态适应输入,减少计算量

### Risk
- 模型更大
- 应用复杂
- 精度下降

### Mitigation
- 暂无

### Outcome
待验证
