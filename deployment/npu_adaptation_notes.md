# NPU Adaptation Notes

> 本文档记录 Ascend 310P NPU 适配过程中的问题和解决方案。

---

## Environment Setup

### Software Stack

```
┌─────────────────────────────────────┐
│         Application Layer           │
│         (BEVFusion Pipeline)        │
├─────────────────────────────────────┤
│         Framework Layer             │
│         (Torch-NPU)                 │
├─────────────────────────────────────┤
│         Runtime Layer               │
│         (CANN ACL)                  │
├─────────────────────────────────────┤
│         Driver Layer                │
│         (Ascend Driver)             │
├─────────────────────────────────────┤
│         Hardware Layer              │
│         (Ascend 310P)               │
└─────────────────────────────────────┘
```

### Version Compatibility

| Component | Version | Notes |
|-----------|---------|-------|
| Ascend Driver | 23.0.0+ | NPU 驱动 |
| CANN | 7.0.0+ | 计算架构 |
| PyTorch | 1.11.0 | 深度学习框架 |
| Torch-NPU | 1.11.0 | PyTorch NPU 扩展 |
| Python | 3.8+ | Python 版本 |

### Installation

```bash
# 1. 安装 Ascend 驱动
sudo ./Ascend-driver.run --install

# 2. 安装 CANN
sudo ./Ascend-cann-toolkit.run --install

# 3. 配置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 4. 安装 Torch-NPU
pip install torch-npu==1.11.0
```

---

## Torch-NPU Usage

### Basic Usage

```python
import torch
import torch_npu

# 检查 NPU 可用性
print(f"NPU available: {torch.npu.is_available()}")
print(f"NPU count: {torch.npu.device_count()}")

# 设置设备
device = torch.device("npu:0")

# 张量创建
x = torch.randn(2, 3).npu()

# 模型迁移
model = MyModel().to(device)
```

### Memory Management

```python
# 清空缓存
torch.npu.empty_cache()

# 获取内存使用情况
memory_allocated = torch.npu.memory_allocated()
memory_reserved = torch.npu.memory_reserved()

# 设置内存限制
torch.npu.set_per_process_memory_fraction(0.8, 0)
```

### Synchronization

```python
# 同步操作
torch.npu.synchronize()

# 异步操作
with torch.npu.stream(torch.npu.Stream()):
    # 异步计算
    pass
```

---

## Data Format Conversion

### NCHW vs NHWC

Ascend 310P 原生支持 NCHW 格式，但某些算子可能需要 NHWC。

```python
# NCHW to NHWC
def nchw_to_nhwc(x):
    return x.permute(0, 2, 3, 1)

# NHWC to NCHW
def nhwc_to_nchw(x):
    return x.permute(0, 3, 1, 2)
```

### FP32 to FP16

```python
# 自动混合精度
from torch_npu.npu.amp import autocast

with autocast():
    output = model(input)

# 手动转换
x_fp16 = x.half()
x_fp32 = x.float()
```

### Host-Device Transfer

```python
# CPU to NPU
x_npu = x_cpu.npu()

# NPU to CPU
x_cpu = x_npu.cpu()

# 非阻塞传输
x_npu = x_cpu.npu(non_blocking=True)
```

---

## Memory Alignment

### Problem
Ascend 310P 对内存对齐有特殊要求。

### Requirements

| Data Type | Alignment |
|-----------|-----------|
| FP32 | 32 bytes |
| FP16 | 16 bytes |
| INT8 | 16 bytes |

### Solution

```python
def align_tensor(x, alignment=32):
    """对齐张量内存"""
    # 计算填充大小
    size = x.numel() * x.element_size()
    padding = (alignment - (size % alignment)) % alignment

    if padding > 0:
        # 创建对齐的张量
        aligned = torch.empty(
            size + padding,
            dtype=x.dtype,
            device=x.device
        )
        aligned[:x.numel()] = x.flatten()
        return aligned.view(x.shape)
    return x
```

---

## Shape Constraints

### Problem
某些算子对输入 shape 有特殊限制。

### Common Constraints

| Operator | Constraint |
|----------|------------|
| Conv2d | 输入通道数需为 4 的倍数 |
| MatMul | 维度需对齐 |
| Reshape | 总元素数需一致 |

### Solution

```python
def pad_to_multiple(x, divisor=4, dim=1):
    """填充到指定倍数"""
    size = x.shape[dim]
    padding = (divisor - (size % divisor)) % divisor

    if padding > 0:
        pad_shape = list(x.shape)
        pad_shape[dim] = padding
        padding_tensor = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x = torch.cat([x, padding_tensor], dim=dim)

    return x
```

---

## Operator Support

### Supported Operators

大部分标准 PyTorch 算子在 Torch-NPU 中有支持：

- Conv2d, Conv3d
- BatchNorm, LayerNorm
- ReLU, GELU, SiLU
- MaxPool, AvgPool
- MatMul, Linear
- Softmax, LogSoftmax

### Unsupported Operators

以下算子可能不支持或有限制：

| Operator | Status | Workaround |
|----------|--------|------------|
| SparseConv3d | ❌ | 使用 Pillar 替代 |
| bev_pool | ❌ | 自定义实现 |
| KNN | ⚠️ | CPU 实现 |
| NMS (3D) | ⚠️ | CPU 实现 |

### Custom Operator Development

```python
# 注册自定义算子
from torch_npu.npu.utils import npu_custom_op

@npu_custom_op
def my_custom_op(x, y):
    # 自定义算子实现
    pass
```

---

## Performance Optimization

### 1. Kernel Fusion

```python
# 使用 JIT 编译融合算子
@torch.jit.script
def fused_op(x, y):
    return torch.relu(x + y)
```

### 2. Memory Optimization

```python
# 使用梯度检查点
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    return checkpoint(self._forward_impl, x)
```

### 3. Batch Processing

```python
# 批量处理提高吞吐量
def batch_inference(model, inputs, batch_size=4):
    outputs = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        output = model(batch)
        outputs.append(output)
    return torch.cat(outputs)
```

---

## Debugging

### Logging

```python
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG)
torch.npu.set_compile_mode(jit_compile=False)
```

### Error Analysis

```python
try:
    output = model(input)
except RuntimeError as e:
    print(f"NPU Error: {e}")
    print(f"Input shape: {input.shape}")
    print(f"Input dtype: {input.dtype}")
    print(f"Memory allocated: {torch.npu.memory_allocated()}")
```

### Profiling

```python
# 使用 NPU profiler
with torch.npu.profile():
    output = model(input)

# 分析结果
# 使用 msprof 工具分析
```

---

## Common Issues

### Issue 1: NPU Not Found

**错误信息**:
```
RuntimeError: NPU device not found
```

**解决方案**:
```bash
# 检查驱动
npu-smi info

# 检查设备
ls /dev/davinci*
```

### Issue 2: Out of Memory

**错误信息**:
```
RuntimeError: NPU out of memory
```

**解决方案**:
```python
# 减小 batch size
batch_size = 1

# 清空缓存
torch.npu.empty_cache()

# 使用梯度检查点
```

### Issue 3: Operator Not Implemented

**错误信息**:
```
NotImplementedError: Could not run 'xxx' with arguments from the 'NPU' backend
```

**解决方案**:
1. 检查算子是否支持
2. 使用 CPU fallback
3. 开发自定义算子

### Issue 4: Precision Mismatch

**错误信息**:
```
Output differs between CPU and NPU
```

**解决方案**:
```python
# 使用 FP32
model = model.float()

# 或使用混合精度
with autocast(enabled=False):
    output = model(input)
```

---

## Best Practices

1. **版本一致性**: 确保驱动、CANN、Torch-NPU 版本匹配
2. **内存管理**: 及时释放不再使用的张量
3. **错误处理**: 捕获并记录 NPU 错误
4. **性能监控**: 使用 profiler 分析性能瓶颈
5. **文档记录**: 记录遇到的问题和解决方案

---

## References

- [Torch-NPU Documentation](https://gitee.com/ascend/pytorch)
- [CANN Developer Guide](https://www.hiascend.com/document)
- [Ascend Operator Support List](https://www.hiascend.com/document)
