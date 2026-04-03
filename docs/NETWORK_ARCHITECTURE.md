# BEVFusion 完整网络架构说明

## 概述

BEVFusion是一个多模态3D目标检测网络，融合了LiDAR点云和相机图像两种模态。本文档详细说明网络结构和各模块参数。

## 整体架构

```
输入数据
├── LiDAR点云 (points)
│   ├── Voxelization
│   ├── VoxelEncoder (PillarFeatureNet)
│   ├── Scatter (PointPillarsScatter)
│   ├── Backbone (SECOND)
│   └── Neck (SECONDFPN)
│   └──→ lidar_bev_feat [B, 256, 180, 180]
│
└── 相机图像 (imgs)
    ├── Backbone (ResNet50)
    ├── Neck (GeneralizedLSSFPN)
    └── ViewTransform (DepthLSSTransform)
        ├── DepthNet
        └── BEV Pool
    └──→ camera_bev_feat [B, 80, 360, 360]

特征融合
├── ConvFuser (Concat + Conv)
└──→ fused_feat [B, 256, 180, 180]

检测头
├── Backbone (SECOND)
├── Neck (SECONDFPN)
└── TransFusionHead
    └──→ 检测结果
```

## 详细模块说明

### 1. LiDAR分支

#### 1.1 Voxelization
```python
voxel_size = [0.30, 0.30, 8.0]  # 体素大小 (x, y, z)
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]  # 点云范围
max_num_points = 32  # 每个体素内最大点数
max_voxels = [8000, 10000]  # [训练, 测试] 最大体素数
```

**输出**: 
- `voxels`: [V, 32, 5] 体素特征
- `num_points`: [V] 每个体素的点数
- `coords`: [V, 4] 体素坐标 (batch_id, z, y, x)

#### 1.2 VoxelEncoder (PillarFeatureNet)
```python
in_channels = 5  # 输入特征维度 (x, y, z, intensity, time_lag)
feat_channels = [64, 128, 256]  # 特征通道数
```

**输出**: `voxel_features` [V, 256]

#### 1.3 Scatter (PointPillarsScatter)
```python
in_channels = 256
output_shape = [360, 360]  # BEV网格大小
```

**输出**: `bev_feat` [B, 256, 360, 360]

#### 1.4 Backbone (SECOND)
```python
in_channels = 256
out_channels = [128, 256]  # 两层输出通道
layer_nums = [5, 5]  # 每层卷积数
layer_strides = [2, 2]  # 每层步长
```

**输出**: 
- `feat1`: [B, 128, 180, 180]
- `feat2`: [B, 256, 90, 90]

#### 1.5 Neck (SECONDFPN)
```python
in_channels = [128, 256]
out_channels = [256, 256]
upsample_strides = [1, 2]
```

**输出**: 
- `neck_feat1`: [B, 256, 180, 180]
- `neck_feat2`: [B, 256, 180, 180]

**最终输出**: `lidar_bev_feat` [B, 256, 180, 180]

### 2. 相机分支

#### 2.1 Backbone (ResNet50)
```python
depth = 50
num_stages = 4
out_indices = (1, 2, 3)  # 输出第2,3,4阶段的特征
frozen_stages = 1  # 冻结第1阶段
```

**输入**: `imgs` [B, N, 3, 256, 704] (N=6 cameras)

**输出**: 
- `feat1`: [B*N, 512, 32, 88]
- `feat2`: [B*N, 1024, 16, 44]
- `feat3`: [B*N, 2048, 8, 22]

#### 2.2 Neck (GeneralizedLSSFPN)
```python
in_channels = [512, 1024, 2048]
out_channels = 256
start_level = 0
num_outs = 3
```

**输出**: 
- `neck_feat1`: [B*N, 256, 32, 88]
- `neck_feat2`: [B*N, 256, 16, 44]

#### 2.3 ViewTransform (DepthLSSTransform)

##### 2.3.1 DepthNet
```python
in_channels = 256
out_channels = 80  # BEV特征通道数
image_size = [256, 704]
feature_size = [32, 88]
dbound = [1.0, 60.0, 0.5]  # 深度范围 [dmin, dmax, dd]
# D = (60 - 1) / 0.5 = 118 个深度bin
```

**输入**: 
- `img_feat`: [B, N, 256, 32, 88]
- `depth`: [B, N, 1, 256, 704] (从点云生成)

**输出**: `cam_feats` [B, N, 80, 118, 32, 88]

##### 2.3.2 BEV Pool
```python
xbound = [-54.0, 54.0, 0.3]  # x范围和分辨率
ybound = [-54.0, 54.0, 0.3]  # y范围和分辨率
zbound = [-10.0, 10.0, 20.0]  # z范围和分辨率
# nx = [360, 360, 1]
```

**输入**: 
- `cam_feats`: [B, N, 80, 118, 32, 88]
- `geom_feats`: [B, N, 118, 32, 88, 3] (几何坐标)

**输出**: `camera_bev_feat` [B, 80, 360, 360]

### 3. 特征融合 (ConvFuser)

```python
in_channels = [80, 256]  # [camera, lidar]
out_channels = 256
```

**操作**:
1. 将 `camera_bev_feat` [B, 80, 360, 360] 下采样到 [B, 80, 180, 180]
2. Concat: [B, 80+256, 180, 180]
3. Conv: [B, 256, 180, 180]

**输出**: `fused_feat` [B, 256, 180, 180]

### 4. 检测头

#### 4.1 Backbone (SECOND)
```python
in_channels = 256
out_channels = [128, 256]
layer_nums = [5, 5]
layer_strides = [2, 2]
```

**输出**: 
- `feat1`: [B, 128, 90, 90]
- `feat2`: [B, 256, 45, 45]

#### 4.2 Neck (SECONDFPN)
```python
in_channels = [128, 256]
out_channels = [256, 256]
upsample_strides = [1, 2]
```

**输出**: `neck_feat` [B, 512, 90, 90]

#### 4.3 TransFusionHead

```python
num_proposals = 200  # K个候选
in_channels = 512
hidden_channel = 128
num_classes = 10
num_decoder_layers = 1
```

**输出**:
- `dense_heatmap`: [B, 10, 90, 90] 原始热力图
- `top_cls`: [B, K] 候选类别
- `query_heatmap_score`: [B, 10, K] 热力图分数
- `heatmap_q`: [B, 10, K] decoder热力图
- `center`: [B, 2, K] 中心坐标
- `height`: [B, 1, K] 高度
- `dim`: [B, 3, K] 尺寸 (w, l, h)
- `rot`: [B, 2, K] 旋转 (sin, cos)
- `vel`: [B, 2, K] 速度

## 关键参数总结

### BEV参数
```python
# LiDAR BEV
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.30, 0.30, 8.0]
grid_size = [360, 360, 1]
out_size_factor = 2  # 下采样因子

# Camera BEV
xbound = [-54.0, 54.0, 0.3]
ybound = [-54.0, 54.0, 0.3]
zbound = [-10.0, 10.0, 20.0]
dbound = [1.0, 60.0, 0.5]
```

### 图像参数
```python
image_size = [256, 704]  # 输入图像大小
feature_size = [32, 88]  # 特征图大小
num_cameras = 6
```

### 网络参数
```python
# LiDAR分支
lidar_channels = 256
lidar_backbone_out = [128, 256]
lidar_neck_out = [256, 256]

# 相机分支
camera_backbone_out = [512, 1024, 2048]
camera_neck_out = 256
camera_bev_channels = 80

# 融合
fusion_out_channels = 256

# 检测头
head_in_channels = 512
head_hidden_channels = 128
num_proposals = 200
num_classes = 10
```

## 数据流总结

### LiDAR分支
```
points [N, 5]
  ↓ Voxelization
voxels [V, 32, 5], coords [V, 4]
  ↓ VoxelEncoder
voxel_features [V, 256]
  ↓ Scatter
bev_feat [B, 256, 360, 360]
  ↓ Backbone
feats [[B,128,180,180], [B,256,90,90]]
  ↓ Neck
lidar_bev_feat [B, 256, 180, 180]
```

### 相机分支
```
imgs [B, 6, 3, 256, 704]
  ↓ Backbone
feats [[B*6,512,32,88], [B*6,1024,16,44], [B*6,2048,8,22]]
  ↓ Neck
neck_feats [[B*6,256,32,88], [B*6,256,16,44]]
  ↓ DepthNet (with depth)
cam_feats [B, 6, 80, 118, 32, 88]
  ↓ BEV Pool (with geom_feats)
camera_bev_feat [B, 80, 360, 360]
```

### 融合+检测
```
lidar_bev_feat [B, 256, 180, 180]
camera_bev_feat [B, 80, 360, 360] → downsample → [B, 80, 180, 180]
  ↓ Concat + Conv
fused_feat [B, 256, 180, 180]
  ↓ Backbone + Neck
neck_feat [B, 512, 90, 90]
  ↓ TransFusionHead
detections
```

## 类别定义

```python
CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
```

## NMS阈值

```python
NMS_THRESHOLDS = {
    "car": 0,
    "truck": 0,
    "construction_vehicle": 0,
    "bus": 0,
    "trailer": 0,
    "barrier": 0,
    "motorcycle": 0,
    "bicycle": 0,
    "pedestrian": 0.175,
    "traffic_cone": 0.175
}
```

## 参考

- 配置文件: `src/configs/bevfusion_lidar-cam_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d_resnet50.py`
- 模型实现: `src/bevfusion/bevfusion.py`
- ViewTransform: `src/bevfusion/depth_lss.py`
- 检测头: `mmdet3d/models/dense_heads/transfusion_head.py`
