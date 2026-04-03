# BEVFusion 推理优化过程

本目录包含了 BEVFusion 模型在 Ascend 310P 上的推理优化演进过程，通过多轮优化将全流程耗时从 **2334ms** 降低至 **166ms**，性能提升约 **14 倍**。

## 优化演进路线

### 1. 原始全流程推理 (`bevfusion_full_npu_evaluator.py`)

**文件**: `bevfusion_full_npu_evaluator.py`

**特点**:
- 单个 OM 模型完成全流程推理
- 包含完整的 LiDAR 分支、Camera 分支和 Fusion Head
- 所有操作串行执行

**性能表现**:
```
Task                         Total(s)   Mean(ms)
sweep_loading                    3.44      42.41
voxelization                     5.24      64.68
image_loading                   19.90     245.62
precompute_depth                 9.76     120.54
precompute_geometry              4.11      50.78
npu_inference                  145.86    1800.77
total_per_sample               189.05    2334.00
```

**主要瓶颈**:
- NPU 推理耗时 1800ms，其中 ScatterElements 算子（BEVPool）造成大量随机写冲突
- 图像串行加载耗时 245ms
- 深度图预计算耗时 120ms

---

### 2. 单 OM 到三 OM 并行 (`bevfusion_parallel_evaluator.py`)

**文件**: `bevfusion_parallel_evaluator.py`

**核心优化**:
1. **BEVPool 算子优化**: 将 ScatterElements（随机写）改为 Gather + ReduceSum（顺序读），大幅减少写冲突
2. **模型拆分**: 将单 OM 拆分为三个 OM：
   - LiDAR Branch OM: 处理点云特征提取
   - Camera Branch OM: 处理图像特征提取
   - Fusion Head OM: 融合与检测头
3. **双流并行**: LiDAR 和 Camera 分支在不同 stream 上并发执行

**性能表现**:
```
Task                         Total(s)   Mean(ms)
sweep_loading                    5.20      64.19
voxelization                     5.89      72.74
image_loading                   24.02     296.56
precompute_depth                13.50     164.65
npu_inference                   14.76     182.22
total_per_sample                64.16     792.10
```

**优化效果**:
- NPU 推理: 1800ms → 182ms（**提升 9.9 倍**）
- 全流程: 2334ms → 792ms（**提升 2.9 倍**）

**新瓶颈**:
- 图像加载仍为串行，耗时 296ms
- 深度图预计算耗时 164ms

---

### 3. 预处理并行优化 (`bevfusion_parallel_npu_evaluator.py`)

**文件**: `bevfusion_parallel_npu_evaluator.py`

**核心优化**:
1. **图像并行加载**: 使用 ThreadPoolExecutor 并行加载 6 张相机图像
   ```python
   # 旧: 串行，总时间 ≈ 6 × 50ms = 300ms
   for cam_key in self.cam_keys:
       img = load_image(img_path)

   # 新: 并行，总时间 ≈ max(单张) = 50ms
   with ThreadPoolExecutor(max_workers=6) as exe:
       imgs = list(exe.map(load_image, img_paths))
   ```

2. **深度图向量化计算**: 将 torch + Python 循环改为纯 numpy 向量化
   ```python
   # 旧: torch + 6 次 Python 循环，每次 ~20ms
   for c in range(N):
       valid_pix = pix[c, on_img[c]].long()
       depth[b, c, 0, valid_pix[:,0], valid_pix[:,1]] = valid_dist

   # 新: 纯 numpy，一次性完成所有相机
   pts_cam = lidar2image_np[:, :3, :] @ pts_hom.T
   cam_idx, pt_idx = np.where(valid)
   depth_out[linear] = d[order]
   ```

3. **标定矩阵缓存**: 传感器标定矩阵固定不变，首帧计算后缓存复用

4. **CPU 任务并行**: 点云加载与图像加载并行；体素化与深度图计算并行

**性能表现**:
```
Task                         Total(s)   Mean(ms)
sweep_loading                    2.93      36.14
voxelization                     5.43      67.01
image_loading                    2.64      32.64
precompute_depth                 2.04      24.85
npu_inference                   14.99     185.03
total_per_sample                28.76     355.01
```

**优化效果**:
- 图像加载: 296ms → 32ms（**提升 9.3 倍**）
- 深度预计算: 164ms → 24ms（**提升 6.8 倍**）
- 全流程: 792ms → 355ms（**提升 2.2 倍**）
- 精度: mAP = 0.2137（几乎不变）

---

### 4. Pipeline 流水线优化 (`bevfusion_parallel_npu_evaluator_v2.py`)

**文件**: `bevfusion_parallel_npu_evaluator_v2.py`

**核心优化**:
1. **Host 双缓冲（Double Buffering）**
   - 预分配两个 host slot，分别存储 voxels/num_points/coords/imgs/depth
   - 当前帧推理时，下一帧预处理结果写入另一个 slot
   - 避免内存踩踏和反复分配

2. **预处理-推理流水化**
   - 通过 prefetch executor 将下一帧的预处理与当前帧的 NPU 推理重叠
   - 点云加载、图像加载、体素化、depth 预计算与推理并行

3. **持久化线程池**
   - 图像读取复用固定线程池，避免每帧创建 6 个线程
   - Stage-A/Stage-B CPU 任务复用固定预处理线程池

4. **设备内存优化**
   - LUT 固定在 device 上，减少 H2D 拷贝
   - LiDAR/Camera 分支输出直接连接到 Head 输入，避免 D2H/H2D

**性能表现**:
```
Task                         Total(s)   Mean(ms)
sweep_loading                    4.33      53.40
voxelization                     1.47      18.14
image_loading                    1.96      24.21
precompute_depth                 6.87      84.86
npu_inference                   10.83     133.71
total_per_sample                13.45     166.09
```

**优化效果**:
- NPU 推理: 185ms → 133ms（**提升 1.4 倍**）
- 全流程: 355ms → 166ms（**提升 2.1 倍**）
- **吞吐量: 6.01 samples/s（约 6 FPS）**

---

## 性能对比总结

| 版本 | 全流程耗时 | NPU 推理耗时 | 提升倍数 | 累计提升 |
|------|-----------|-------------|---------|---------|
| 原始版本 | 2334ms | 1800ms | - | - |
| BEVPool 优化 | 792ms | 182ms | 2.9x | 2.9x |
| 预处理并行 | 355ms | 185ms | 2.2x | 6.6x |
| Pipeline 优化 | 166ms | 133ms | 2.1x | **14.1x** |

---

## 文件说明

### 评估器（Evaluator）
- `bevfusion_full_npu_evaluator.py`: 原始全流程推理评估器
- `bevfusion_parallel_evaluator.py`: 三 OM 并行推理评估器
- `bevfusion_parallel_npu_evaluator.py`: 预处理并行优化评估器
- `bevfusion_parallel_npu_evaluator_v2.py`: Pipeline 流水线优化评估器（最终版本）

### 网络封装（Net）
- `bevfusion_full_npu_net.py`: 单 OM 模型封装
- `bevfusion_parallel_net.py`: 三 OM 并行模型封装
- `bevfusion_parallel_npu_net.py`: 预处理并行模型封装
- `bevfusion_parallel_npu_net_v2.py`: Pipeline 优化模型封装（最终版本）

### 模型导出（Export）
- `export_bevfusion_full_npu.py`: 导出单 OM 模型
- `export_bevfusion_parallel.py`: 导出三 OM 模型
- `export_bevfusion_parallel_npu.py`: 导出预处理并行模型
- `export_bevfusion_parallel_npu_v2.py`: 导出 Pipeline 优化模型（最终版本）

---

## 关键技术点

### 1. BEVPool 算子优化
- **问题**: ScatterElements 造成大量随机写冲突，硬件串行化严重
- **方案**: 改为 Gather + ReduceSum，将随机写变为顺序读
- **效果**: NPU 推理从 1800ms 降至 182ms

### 2. 模型拆分与并行
- **方案**: 将单 OM 拆分为 LiDAR/Camera/Fusion-Head 三个 OM
- **执行**: LiDAR 和 Camera 在不同 stream 上并发，Fusion 等待两者完成后执行
- **效果**: 充分利用 NPU 多流并行能力

### 3. 预处理并行化
- **图像加载**: 6 张相机图像并行读取
- **深度计算**: numpy 向量化替代 torch 循环
- **标定缓存**: 固定矩阵首帧计算后复用
- **效果**: 预处理从 450ms 降至 90ms

### 4. Pipeline 流水线
- **双缓冲**: 当前帧推理与下一帧预处理重叠
- **持久化线程池**: 避免线程创建销毁开销
- **设备内存优化**: 减少 H2D/D2H 拷贝
- **效果**: 全流程从 355ms 降至 166ms

---

## 使用方法

### 运行最终优化版本
```bash
python bevfusion_parallel_npu_evaluator_v2.py \
    --dataroot data/nuscenes-mini \
    --version v1.0-mini \
    --om-dir models/om \
    --out-dir results
```

### 导出优化后的 OM 模型
```bash
python export_bevfusion_parallel_npu_v2.py \
    --onnx-dir models/onnx \
    --om-dir models/om \
    --soc-version Ascend310P1
```

---

## 后续优化方向

1. **量化**: 使用 PTQ（Post-Training Quantization）进行 INT8 量化
2. **算子融合**: 将 BEVPool 算子通过 AscendC 开发并插入 OM 模型
3. **动态 batch**: 支持多帧批处理推理
4. **模型编译优化**: 探索 ATC 更多编译优化选项

---

## 参考文献

优化过程的详细记录请参考: `docs/Source/Record_Daily0316-0323.md`
