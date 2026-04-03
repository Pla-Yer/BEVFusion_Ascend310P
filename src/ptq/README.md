# BEVFusion PTQ 训练后量化指南

本目录包含将 BEVFusion 三段并行 ONNX 模型量化为 INT8 并部署至昇腾 NPU 的完整工具链。

---

## 目录结构

```
src/ptq/
├── amct_session_utils.py        # 公共工具：AMCT 自定义 op 注册 / ORT Session 创建
├── quantize_lidar_branch.py     # LiDAR 分支量化脚本
├── quantize_camera_branch.py    # Camera 分支量化脚本
├── quantize_fusion_head.py      # Fusion Head 量化脚本
├── compare_lidar_branch.py      # LiDAR 分支精度对比脚本
├── compare_camera_branch.py     # Camera 分支精度对比脚本
└── compare_fusion_head.py       # Fusion Head 精度对比脚本
```

量化产物默认输出至：

```
tmp/
├── lidar/                       # 量化中间文件（config、modified model、record）
├── camera/
└── fusion/

results/
├── lidar/
│   ├── lidar_branch_quant_deploy_model.onnx      # 送 ATC 编译的部署模型
│   └── lidar_branch_quant_fake_quant_model.onnx  # CPU 精度仿真模型
├── camera/
│   ├── camera_branch_quant_deploy_model.onnx
│   └── camera_branch_quant_fake_quant_model.onnx
└── fusion/
    ├── fusion_head_quant_deploy_model.onnx
    └── fusion_head_quant_fake_quant_model.onnx
```

---

## 环境依赖

| 依赖 | 版本要求 | 说明 |
|---|---|---|
| Python | ≥ 3.8 | 推荐在 `amct_onnx` conda 环境中运行 |
| amct_onnx | CANN 配套版本 | 华为昇腾 AMCT 量化工具包 |
| onnxruntime | ≥ 1.14 | CPU 推理 / 校准 |
| onnx | ≥ 1.13 | Fusion Head skip_layers 扫描 |
| numpy | ≥ 1.21 | 数据构造与指标计算 |

激活环境：

```bash
conda activate amct_onnx
```

---

## 背景：为什么量化

BEVFusion 的全精度（FP32/FP16）模型在昇腾 310P 上推理延迟较高。通过 AMCT 训练后量化（PTQ）将 Conv、Gemm、MatMul 等算子的权重与激活量化至 INT8，可在精度损失极小的前提下：

- 模型体积减少约 **75%**（FP32 → INT8）
- NPU 推理吞吐量提升 **2～4×**
- 片上带宽占用显著降低

三段模型并行执行示意：

```
voxels/coords ──► [ LiDAR Branch ] ──────────────────────────────────┐
                                                                       ▼
imgs/depth    ──► [ Camera Branch ] ──► [ Fusion Head ] ──► detections
                         ▲
              pool_lookup/pool_mask
```

LiDAR 与 Camera 分支可在两个 NPU Stream 上并行执行，Fusion Head 等待二者完成后再运行。三段分别量化后即可独立送 ATC 编译为 OM 模型。

---

## 量化流程详解

AMCT 均匀量化（手工量化）标准四步流程如下：

```
原始 ONNX
    │
    ▼
[Step 1] create_quant_config()
    │  生成 config.json，记录哪些层参与量化、校准 batch 数等配置
    │
    ▼
[Step 2] quantize_model()
    │  图优化（Conv+BN 融合等）→ 插入权重量化算子（完成权重量化）
    │  → 插入激活校准算子（IFMR 等）→ 生成 modified_model.onnx
    │
    ▼
[Step 3] ORT 校准推理（batch_num 次）
    │  用校准集驱动 modified_model 前向，IFMR 算子统计激活分布，
    │  将量化因子（scale/offset）写入 record.txt
    │
    ▼
[Step 4] save_model()
    │  读取 record.txt 中的量化因子，将校准算子替换为
    │  AscendQuant / AscendDeQuant，生成两个模型：
    │    ├── *_deploy_model.onnx   → 送 ATC 编译，部署至 NPU
    │    └── *_fake_quant_model.onnx → CPU 精度仿真，验证精度损失
    ▼
deploy_model ──ATC──► OM 模型 ──► 昇腾 NPU 推理
```

> **fake_quant 模型**：在 ORT 环境中模拟 INT8 量化效果（量化→反量化→FP32 计算），其精度与实际 NPU 部署精度高度一致，是量化精度的可靠预估手段。

---

## 量化脚本使用

### 公共工具：`amct_session_utils.py`

所有量化与对比脚本均依赖此模块，需与其他脚本放在**同一目录**下。

它解决了以下问题：AMCT 向 `modified_model.onnx` 插入了 `amct.customop` 域的校准算子（IFMR 等），原生 ORT 不识别，必须先注册 AMCT 的 `.so` 动态库。模块会自动在 `amct_onnx` 包目录下搜索正确的库（通过验证 `RegisterCustomOps` 符号存在性），也可手动指定：

```bash
export AMCT_CUSTOM_OP_LIB=/path/to/libamct_onnx_ops.so
```

手动搜索正确库的方法：

```bash
find $CONDA_PREFIX -name '*.so' | xargs -I{} sh -c \
  'nm -D {} 2>/dev/null | grep -q RegisterCustomOps && echo {}'
```

---

### Step 1：量化 LiDAR 分支

**模型特点**：输入含动态轴（voxel 数量 V 可变），ATC 编译需指定 `--dynamic_dims`。

```bash
python src/ptq/quantize_lidar_branch.py \
    --onnx      models/onnx_parallel_npu/bevfusion_lidar_branch.onnx \
    --max-voxels 10000 \
    --batch-num  8
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--onnx` | 见上 | 原始 ONNX 路径 |
| `--max-voxels` | 8000 | 校准时喂入的 voxel 数量 |
| `--batch-num` | 1 | 校准推理次数，建议 8～32 |
| `--skip-layers` | 空 | 跳过量化的算子名（精度敏感层） |

---

### Step 2：量化 Camera 分支

**模型特点**：所有输入静态；`pool_mask` 为 uint8（v2 导出改动），脚本已自动处理 dtype cast。

```bash
python src/ptq/quantize_camera_branch.py \
    --onnx     models/onnx_parallel_npu/bevfusion_camera_branch.onnx \
    --batch-num 8
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--onnx` | 见上 | 原始 ONNX 路径 |
| `--batch-num` | 1 | 校准推理次数，建议 8～32 |
| `--skip-layers` | 空 | 跳过量化的算子名 |

> **生产建议**：Camera 分支对图像纹理特征敏感，校准集建议使用真实 nuScenes val 集图像，而非 dummy 数据，否则量化精度可能明显下降。

---

### Step 3：量化 Fusion Head

**模型特点**：Transformer Decoder 内部含双变量 tensor MatMul（两路输入均非常量）。根据 AMCT 文档，此类 MatMul 在 **Ascend 310P** 上量化无收益（精度会下降），脚本会自动扫描并将其加入 `skip_layers`。

```bash
python src/ptq/quantize_fusion_head.py \
    --onnx     models/onnx_parallel_npu/bevfusion_fusion_head.onnx \
    --batch-num 8 \
    --soc      Ascend310P1
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--onnx` | 见上 | 原始 ONNX 路径 |
| `--batch-num` | 1 | 校准推理次数 |
| `--soc` | Ascend310P1 | 目标 SOC 型号，影响 skip_layers 策略 |
| `--skip-layers` | 自动检测 | 手动指定时覆盖自动检测结果 |

> 若目标 SOC 为 Atlas A2/A3 系列，双变量 MatMul 量化有收益，可将 `--skip-layers` 显式设为空：`--skip-layers`（不带参数值）。

---

### 量化常见问题

#### 问题 1：`No opset import for domain 'amct.customop'`

ORT 未注册 AMCT 自定义算子。`amct_session_utils.py` 已解决此问题（自动搜索 `.so`），如仍报错请手动设置环境变量：

```bash
export AMCT_CUSTOM_OP_LIB=/path/to/libamct_onnx_ops.so
```

#### 问题 2：`undefined symbol: RegisterCustomOps`

找到的 `.so` 不是 ORT custom op 库（如 `libamct_ncx.so`）。`amct_session_utils.py` 通过 `ctypes` 验证 `RegisterCustomOps` 符号存在性来过滤，如自动搜索仍失败，用上面的搜索命令手动定位。

#### 问题 3：`Unexpected input data type. Actual: (tensor(int32)), expected: (tensor(float))`

AMCT 图优化 pass 可能将 `int32` 输入节点重定向至 `float32` 中间节点。所有量化脚本均会在 `session.run()` 前从 Session 读取期望 dtype 并自动 cast，已内置处理。

#### 问题 4：校准推理时内存 OOM（`std::bad_alloc`）

减小 `--max-voxels` 或 `--batch-num`，或参考 AMCT 文档设置 `OMP_NUM_THREADS`：

```bash
export OMP_NUM_THREADS=8
```

---

## 精度对比脚本使用

量化完成后，用 `fake_quant_model.onnx` 与原始 ONNX 做精度对比，**无需 NPU 硬件**，在 CPU 上即可完成。

### LiDAR 分支精度对比

```bash
python src/ptq/compare_lidar_branch.py \
    --ori-onnx  models/onnx_parallel_npu/bevfusion_lidar_branch.onnx \
    --qnt-onnx  results/lidar/lidar_branch_quant_fake_quant_model.onnx \
    --max-voxels 10000 \
    --num-iters  10
```

### Camera 分支精度对比

```bash
python src/ptq/compare_camera_branch.py \
    --ori-onnx  models/onnx_parallel_npu/bevfusion_camera_branch.onnx \
    --qnt-onnx  results/camera/camera_branch_quant_fake_quant_model.onnx \
    --num-iters  10
```

### Fusion Head 精度对比

```bash
python src/ptq/compare_fusion_head.py \
    --ori-onnx  models/onnx_parallel_npu/bevfusion_fusion_head.onnx \
    --qnt-onnx  results/fusion/fusion_head_quant_fake_quant_model.onnx \
    --num-iters  10
```

| 参数 | 说明 |
|---|---|
| `--ori-onnx` | 原始 ONNX 路径 |
| `--qnt-onnx` | 量化后 fake_quant ONNX 路径 |
| `--num-iters` | 对比推理次数（取平均），建议 ≥ 5 |

---

## 精度指标说明

每个输出张量均打印以下指标：

| 指标 | 全称 | 计算方式 | 参考阈值 |
|---|---|---|---|
| **CosSim** | 余弦相似度 | `dot(a,b) / (‖a‖·‖b‖)` | ≥ 0.999 优秀，≥ 0.99 可接受 |
| **MaxAE** | 最大绝对误差 | `max(|a-b|)` | 越小越好 |
| **MAE** | 平均绝对误差 | `mean(|a-b|)` | 越小越好 |
| **RelErr** | 相对误差 | `MAE / mean(|origin|)` | < 0.01 优秀 |
| **SNR** | 信噪比 | `10·log₁₀(signal/noise)` dB | > 40 dB 通常可接受 |

### Camera 分支额外指标

**逐通道余弦相似度分布**（共 80 个 BEV 特征通道）：

```
逐通道余弦相似度统计（共 80 通道）：
  min=0.99312  max=0.99998  mean=0.99821  <0.99 通道数: 2
```

若 `<0.99 通道数` 偏多，可将对应通道关联的卷积层名加入 `--skip-layers`。

### Fusion Head 额外指标

**检测头专项指标**（第 1 次 iter 打印）：

```
── 检测头专项指标 ──────────────────────────────────────
Top-K index 命中率 : 197/200 = 98.50%
center   平均 L2 偏差 : 0.003241
dim      平均 L2 偏差 : 0.001872
rot      平均 L2 偏差 : 0.000934
dense_heatmap sigmoid 统计：
  ori  mean=0.04123  max=0.91234
  qnt  mean=0.04118  max=0.90987
  MAE=0.000312
```

| 专项指标 | 含义 | 参考阈值 |
|---|---|---|
| Top-K 命中率 | 量化前后 Top-200 proposal index 重合比例 | ≥ 95% |
| center L2 | 已选 proposal 的中心点坐标偏差 | < 0.01 |
| dim L2 | 尺寸预测偏差 | < 0.01 |
| rot L2 | 朝向预测偏差 | < 0.01 |

---

## ATC 编译（量化模型 → OM）

量化脚本执行完毕后会自动打印对应的 ATC 命令，也可参考以下模板：

```bash
# LiDAR 分支（动态 voxel 数量）
atc --model="results/lidar/lidar_branch_quant_deploy_model.onnx" \
    --framework=5 \
    --output="models/om/bevfusion_lidar_branch_quant_dynamic" \
    --input_format=ND \
    --input_shape="voxels:-1,32,5;num_points:-1;coords:-1,4" \
    --dynamic_dims="6000,6000,6000;8000,8000,8000;10000,10000,10000" \
    --soc_version=Ascend310P1 \
    --op_select_implmode=high_precision \
    --precision_mode=allow_fp32_to_fp16 \
    --log=warning

# Camera 分支（静态形状）
atc --model="results/camera/camera_branch_quant_deploy_model.onnx" \
    --framework=5 \
    --output="models/om/bevfusion_camera_branch_quant" \
    --input_format=ND \
    --input_shape="imgs:1,6,3,256,704;depth:1,6,1,256,704;pool_lookup:129600,16;pool_mask:129600,16" \
    --soc_version=Ascend310P1 \
    --op_select_implmode=high_precision \
    --precision_mode=allow_fp32_to_fp16 \
    --log=warning

# Fusion Head（静态形状）
atc --model="results/fusion/fusion_head_quant_deploy_model.onnx" \
    --framework=5 \
    --output="models/om/bevfusion_fusion_head_quant" \
    --input_format=ND \
    --input_shape="camera_bev:1,80,360,360;lidar_bev:1,64,360,360" \
    --soc_version=Ascend310P1 \
    --op_select_implmode=high_precision \
    --precision_mode=allow_fp32_to_fp16 \
    --log=warning
```

---

## 完整执行流程

```bash
# 0. 激活环境
conda activate amct_onnx
cd /path/to/BEVFusion_Ascend310P

# 1. 量化三个分支（建议 batch-num ≥ 8，生产环境换用真实校准数据）
python src/ptq/quantize_lidar_branch.py  --max-voxels 10000 --batch-num 8
python src/ptq/quantize_camera_branch.py --batch-num 8
python src/ptq/quantize_fusion_head.py   --batch-num 8 --soc Ascend310P1

# 2. 精度对比（验证量化损失）
python src/ptq/compare_lidar_branch.py  --num-iters 10
python src/ptq/compare_camera_branch.py --num-iters 10
python src/ptq/compare_fusion_head.py   --num-iters 10

# 3. 精度满足要求后，执行 ATC 编译
# （各量化脚本结尾已自动打印具体 ATC 命令，复制执行即可）

# 4. 部署运行（配合推理脚本使用量化 OM 模型）
python src/bevfusion_parallel_npu_evaluator.py \
    --lidar-om  models/om/bevfusion_lidar_branch_quant_dynamic.om \
    --camera-om models/om/bevfusion_camera_branch_quant.om \
    --fusion-om models/om/bevfusion_fusion_head_quant.om \
    ...
```

---

## 精度调优建议

若对比脚本输出的余弦相似度不达标，按以下顺序排查：

**1. 使用真实校准数据**

dummy 数据的激活分布与真实推理差异很大，会导致量化因子不准确。建议从 nuScenes val 集中随机抽取 **16～32 帧**作为校准集：

```python
# 在量化脚本 build_dummy_inputs() 位置替换为：
# LiDAR：加载真实点云 → voxelization → 输出 voxels/num_points/coords
# Camera：加载真实图像 → 归一化 → 输出 imgs/depth/pool_lookup/pool_mask
# Fusion：由上两支的真实 BEV 特征输出获得
```

**2. 增大校准 batch 数**

```bash
--batch-num 32    # 更多帧 → 激活统计更准确
```

**3. 跳过精度敏感层**

对比脚本会打印余弦相似度 < 0.99 的输出及相关建议。找到对应的算子节点名（通过 Netron 可视化 ONNX 图），加入 `--skip-layers`：

```bash
# 示例：跳过某个 BEV 池化后的第一个 Conv
python src/ptq/quantize_camera_branch.py \
    --skip-layers /view_transform/bev_pool/Conv_0
```

**4. 升级为基于精度的自动量化**

如手动调整仍无法满足精度要求，可改用 AMCT 的 `accuracy_based_auto_calibration` 接口，自动搜索最优量化配置，参考 [AMCT 文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/devaids/amct/atlasamct_16_0141.html)。

---

## 参考文档

- [AMCT ONNX 均匀量化](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/devaids/amct/atlasamct_16_0144.html)
- [AMCT 基于精度的自动量化](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/devaids/amct/atlasamct_16_0141.html)
- [ATC 工具使用指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/devaids/amct/atlasamct_16_0141.html)
- `src/export_bevfusion_parallel_npu_v2.py` — 三段 ONNX 导出脚本
