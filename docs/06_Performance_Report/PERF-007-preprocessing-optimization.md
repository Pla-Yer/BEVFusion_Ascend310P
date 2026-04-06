## PERF-007: 预处理并行化优化

### Date
2026-03-20

### Configuration
- Hardware: Ascend 310P1
- Software: Ascend Toolkit 8.3.RC1
- Model: BEVFusion Full NPU
- Input Size: 6 cameras, 256x704, dynamic voxels

### Metrics

**优化前性能：**
| Component | Latency (ms) | % of Total | Status |
|-----------|--------------|------------|--------|
| sweep_loading | 64.19 | 8.1% | ⚠️ |
| voxelization | 72.74 | 9.2% | ⚠️ |
| image_loading | 296.56 | 37.4% | ❌ |
| precompute_depth | 164.65 | 20.8% | ❌ |
| precompute_geometry | 6.23 | 0.8% | ✅ |
| npu_inference | 182.22 | 23.0% | ✅ |
| decode | 0.67 | 0.08% | ✅ |
| nms | 2.77 | 0.3% | ✅ |
| transform | 2.54 | 0.3% | ✅ |
| **total_per_sample** | **792.10** | **100%** | ⚠️ |

**优化后性能：**
| Component | Latency (ms) | % of Total | Status |
|-----------|--------------|------------|--------|
| sweep_loading | 36.14 | 10.2% | ✅ |
| voxelization | 67.01 | 18.9% | ⚠️ |
| image_loading | 32.64 | 9.2% | ✅ |
| precompute_depth | 24.85 | 7.0% | ✅ |
| precompute_geometry | 2.36 | 0.7% | ✅ |
| npu_inference | 185.03 | 52.1% | ✅ |
| decode | 0.69 | 0.2% | ✅ |
| nms | 2.80 | 0.8% | ✅ |
| transform | 2.61 | 0.7% | ✅ |
| **total_per_sample** | **355.01** | **100%** | ✅ |

### Bottleneck Analysis

**识别的瓶颈：**
1. **image_loading 296ms**：6张图串行disk I/O
2. **precompute_depth 164ms**：torch + Python per-camera循环，每帧重算逆矩阵
3. **sweep/voxelization ~135ms**：可以并行预读下一帧
4. **pose/calib nusc.get()**：每帧都lookup，但sensor calib完全固定

### Optimization

**三步优化策略：**

**1. 并行图像加载**
```python
# 旧：串行，总时间 = sum(单张时间) ≈ 6 × 50ms
for cam_key in self.cam_keys:
    img = load_and_preprocess_image(img_path, ...)

# 新：并行，总时间 = max(单张时间) ≈ 50ms
with ThreadPoolExecutor(max_workers=N) as exe:
    imgs_list = list(exe.map(_load, img_paths))
```

**2. 纯numpy向量化深度投影**
```python
# 旧：torch + 6次Python循环
for c in range(N):           # Python loop，每次 ~20ms
    valid_pix = pix[c, on_img[c]].long()
    depth[b, c, 0, valid_pix[:,0], valid_pix[:,1]] = valid_dist

# 新：纯numpy，一次性完成所有相机
pts_cam = lidar2image_np[:, :3, :] @ pts_hom.T   # [N,3,M] 单次 matmul
cam_idx, pt_idx = np.where(valid)                  # 一次 where
depth_out[linear] = d[order]                       # 一次 fancy-index scatter
```

**3. 缓存传感器标定矩阵**
- 避免每帧重复lookup
- 缓存固定的sensor calib矩阵

### Results

**性能提升：**
- image_loading：296.56ms → 32.64ms（**提升89.0%**）
- precompute_depth：164.65ms → 24.85ms（**提升84.9%**）
- 总流程时间：792.10ms → 355.01ms（**提升55.2%**）
- 模型精度：mAP=0.2137（保持不变）

**关键成果：**
- ✅ 图像加载实现并行化
- ✅ 深度投影实现向量化
- ✅ 消除了Python循环瓶颈
- ✅ 总流程时间降至355ms

### Notes

**NPU推理详细性能：**
```
Stage            Total(s)   Mean(ms)    Std(ms)    Min(ms)    Max(ms)
lut_build           0.116     116.15       0.00     116.15     116.15
h2d                 2.438      30.10       4.27      27.03      58.77
infer              12.241     151.12       2.94     133.06     153.53
d2h                 0.079       0.97       0.15       0.81       2.09
total              15.055     185.86       6.00     164.08     214.29
```

**下一步优化方向：**
- 模型推理并行化：LiDAR分支与Camera分支并行
- 进一步降低总流程时间
