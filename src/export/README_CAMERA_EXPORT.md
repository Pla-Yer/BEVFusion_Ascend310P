# BEVFusion 图像分支导出说明

## 概述

本脚本用于导出BEVFusion的图像分支，包括Backbone、Neck和ViewTransform三个部分。由于ViewTransform中的BEV Pool涉及复杂的scatter操作，在ONNX导出时可能存在兼容性问题，因此采用分阶段导出的方式。

## 导出的模型

### Stage 1: Backbone (ResNet50)
- **输入**: `[B, N, 3, 256, 704]` (B=batch, N=6 cameras)
- **输出**: 3个特征层
  - `feat1`: `[B*N, 512, 32, 88]`
  - `feat2`: `[B*N, 1024, 16, 44]`
  - `feat3`: `[B*N, 2048, 8, 22]`
- **文件**: `models/onnx_camera/stage1_backbone.onnx` (90MB)

### Stage 2: Neck (GeneralizedLSSFPN)
- **输入**: 3个backbone特征
  - `bb_feat1`: `[B*N, 512, 32, 88]`
  - `bb_feat2`: `[B*N, 1024, 16, 44]`
  - `bb_feat3`: `[B*N, 2048, 8, 22]`
- **输出**: 2个特征层
  - `neck_feat1`: `[B*N, 256, 32, 88]`
  - `neck_feat2`: `[B*N, 256, 16, 44]`
- **文件**: `models/onnx_camera/stage2_neck.onnx` (8.3MB)

### Stage 3: ViewTransform - DepthNet
- **输入**:
  - `img_feat`: `[B, N, 256, 32, 88]`
  - `depth`: `[B, N, 1, 256, 704]` (从点云生成的深度图)
- **输出**:
  - `cam_feats`: `[B, N, 80, 118, 32, 88]`
- **文件**: `models/onnx_camera/stage3_depthnet.onnx` (5.5MB)

### Stage 4: ViewTransform - BEV Pool (CPU实现)
- **输入**:
  - `cam_feats`: `[B, N, 80, 118, 32, 88]`
  - `geom_feats`: `[B, N, 118, 32, 88, 3]` (几何坐标)
- **输出**:
  - `bev_feat`: `[B, 80, 360, 360]`
- **说明**: 由于scatter操作在ONNX导出时的兼容性问题，建议在CPU上实现

## 使用方法

### 1. 导出ONNX模型

```bash
# 导出所有阶段
python src/export/export_bevfusion_camera.py --stage all

# 导出指定阶段
python src/export/export_bevfusion_camera.py --stage 1  # 只导出Backbone
python src/export/export_bevfusion_camera.py --stage 2  # 只导出Neck
python src/export/export_bevfusion_camera.py --stage 3  # 只导出DepthNet
```

### 2. 转换为OM模型

使用ATC工具将ONNX转换为OM：

```bash
# Stage 1: Backbone
atc --model="models/onnx_camera/stage1_backbone.onnx" \
    --framework=5 \
    --output="models/om/stage1_backbone" \
    --input_format=NCHW \
    --input_shape="img:6,3,256,704" \
    --soc_version=Ascend310P1 \
    --precision_mode=allow_fp32_to_fp16 \
    --log=warning

# Stage 2: Neck
atc --model="models/onnx_camera/stage2_neck.onnx" \
    --framework=5 \
    --output="models/om/stage2_neck" \
    --input_format=NCHW \
    --input_shape="bb_feat1:6,512,32,88;bb_feat2:6,1024,16,44;bb_feat3:6,2048,8,22" \
    --soc_version=Ascend310P1 \
    --precision_mode=allow_fp32_to_fp16 \
    --log=warning

# Stage 3: DepthNet
atc --model="models/onnx_camera/stage3_depthnet.onnx" \
    --framework=5 \
    --output="models/om/stage3_depthnet" \
    --input_format=NCHW \
    --input_shape="img_feat:1,6,256,32,88;depth:1,6,1,256,704" \
    --soc_version=Ascend310P1 \
    --precision_mode=allow_fp32_to_fp16 \
    --log=warning
```

## 部署流程

### 1. CPU预计算部分

以下部分建议在CPU上预计算：

#### 1.1 深度图生成 (depth)
从点云投影到图像平面，参考 `src/bevfusion/depth_lss.py` 中的 `BaseDepthTransform.forward` 方法：

```python
# 伪代码
for b in range(batch_size):
    cur_coords = points[b][:, :3]
    # inverse aug
    cur_coords -= cur_lidar_aug_matrix[b][:3, 3]
    cur_coords = torch.inverse(cur_lidar_aug_matrix[b][:3, :3]).matmul(cur_coords.T)
    # lidar2image
    cur_coords = lidar2image[b][:, :3, :3].matmul(cur_coords)
    cur_coords += lidar2image[b][:, :3, 3].reshape(-1, 3, 1)
    # get 2d coords
    cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]
    # imgaug
    cur_coords = img_aug_matrix[b][:, :3, :3].matmul(cur_coords)
    cur_coords += img_aug_matrix[b][:3, 3].reshape(-1, 3, 1)
    # 投影到图像平面
    ...
```

#### 1.2 几何特征计算 (geom_feats)
通过 `get_geometry` 计算，涉及矩阵逆运算：

```python
# 伪代码
geom = get_geometry(
    camera2lidar_rots,
    camera2lidar_trans,
    intrins,
    post_rots,
    post_trans,
    extra_rots=extra_rots,
    extra_trans=extra_trans,
)
```

### 2. NPU推理部分

#### 2.1 Stage 1: Backbone
```python
# 输入: img [B, N, 3, 256, 704]
# 输出: feat1, feat2, feat3
```

#### 2.2 Stage 2: Neck
```python
# 输入: feat1, feat2, feat3
# 输出: neck_feat1, neck_feat2
```

#### 2.3 Stage 3: DepthNet
```python
# 输入: img_feat [B, N, 256, 32, 88], depth [B, N, 1, 256, 704]
# 输出: cam_feats [B, N, 80, 118, 32, 88]
```

#### 2.4 Stage 4: BEV Pool (CPU)
```python
# 输入: cam_feats [B, N, 80, 118, 32, 88], geom_feats [B, N, 118, 32, 88, 3]
# 输出: bev_feat [B, 80, 360, 360]
# 在CPU上实现bev_pool_scatter
```

## 注意事项

1. **Stage 4的BEV Pool**: 由于scatter操作在ONNX导出时的兼容性问题，建议在CPU上实现。可以参考 `src/bevfusion/depth_lss.py` 中的 `bev_pool_scatter` 方法。

2. **深度图生成**: 需要从点云投影到图像平面，涉及复杂的坐标变换，建议在CPU上预计算。

3. **几何特征计算**: 涉及矩阵逆运算，建议在CPU上预计算。

4. **内存优化**: Stage 1-3的输出可以作为Stage 4的输入，建议使用流水线方式处理，避免内存占用过大。

## 测试数据

导出脚本会自动生成测试数据，保存在 `camera_bins/` 目录下：

- `stage1_img.bin`: Stage 1输入
- `stage1_feat0/1/2.bin`: Stage 1输出
- `stage2_bb_feat0/1/2.bin`: Stage 2输入
- `stage2_neck_feat0/1.bin`: Stage 2输出
- `stage3_img_feat.bin`: Stage 3输入
- `stage3_depth.bin`: Stage 3输入
- `stage3_cam_feats.bin`: Stage 3输出

可以使用这些数据进行模型验证。

## 参考代码

- 图像分支实现: `src/bevfusion/bevfusion.py`
- ViewTransform实现: `src/bevfusion/depth_lss.py`
- Neck实现: `src/bevfusion/bevfusion_necks.py`
- 雷达分支导出: `src/export/export_bevfusion_full0306.py`
