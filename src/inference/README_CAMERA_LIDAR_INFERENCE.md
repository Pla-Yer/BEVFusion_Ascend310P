# BEVFusion Camera+Lidar 推理框架

## 概述

本框架实现了BEVFusion的雷达+相机融合推理，采用NPU+CPU混合计算策略：

- **NPU部分**：雷达分支、相机分支（Backbone、Neck、DepthNet）、融合+检测头
- **CPU部分**：深度图生成、几何特征计算、BEV Pool

## 架构

```
输入数据
├── 点云数据 (points)
│   ├── Voxelization (CPU)
│   └── 雷达分支 (NPU)
│       ├── VoxelEncoder
│       ├── Scatter
│       ├── Backbone
│       └── Neck
│       └──→ lidar_bev_feat [B, C_l, H, W]
│
└── 图像数据 (imgs)
    └── 相机分支 (NPU)
        ├── Backbone (ResNet50)
        ├── Neck (FPN)
        └── DepthNet
        └──→ cam_feats [B, N, C, D, H, W]
    
CPU计算部分
├── 深度图生成 (depth generation)
│   └── 从点云投影到图像平面
├── 几何特征计算 (geometry calculation)
│   └── 计算每个像素的lidar坐标
└── BEV Pool
    └── 将cam_feats投影到BEV空间
    └──→ camera_bev_feat [B, C_c, H, W]

融合+检测头 (NPU)
├── 特征融合 (Fusion)
│   └── Concat([lidar_bev_feat, camera_bev_feat])
├── Backbone
├── Neck
└── Detection Head
    └──→ 检测结果
```

## 文件结构

```
src/inference/
├── bevfusion_camera_lidar_net.py       # 推理网络基类
├── bevfusion_camera_lidar_evaluator.py # 评估脚本
└── README_CAMERA_LIDAR_INFERENCE.md    # 本文档
```

## 使用方法

### 1. 准备OM模型

需要准备以下OM模型：

```bash
# 雷达分支模型
models/om/bevfusion_lidar.om

# 相机分支模型
models/om/camera_backbone.om      # Stage 1: Backbone
models/om/camera_neck.om          # Stage 2: Neck
models/om/camera_depthnet.om      # Stage 3: DepthNet

# 融合+检测头模型
models/om/fusion_head.om
```

### 2. 导出ONNX模型

```bash
# 导出相机分支
python src/export/export_bevfusion_camera.py --stage all

# 转换为OM模型
atc --model=models/onnx_camera/stage1_backbone.onnx \
    --framework=5 \
    --output=models/om/camera_backbone \
    --input_format=ND \
    --input_shape="img:6,3,256,704" \
    --soc_version=Ascend310P1
```

### 3. 运行推理

```python
from bevfusion_camera_lidar_net import BEVFusionCameraLidarNet, init_acl

# 初始化ACL
ctx = init_acl(0)

# 初始化网络
net = BEVFusionCameraLidarNet(
    lidar_model_path="models/om/bevfusion_lidar.om",
    camera_backbone_path="models/om/camera_backbone.om",
    camera_neck_path="models/om/camera_neck.om",
    camera_depthnet_path="models/om/camera_depthnet.om",
    fusion_head_path="models/om/fusion_head.om"
)

# 准备输入数据
voxels, num_points, coords = voxelizer(points)
imgs, metas = load_images_and_metas(sample_token)

# 运行推理
outputs = net.forward(voxels, num_points, coords, imgs, metas)
```

### 4. 运行评估

```bash
python src/inference/bevfusion_camera_lidar_evaluator.py
```

## 关键参数

### BEV参数

```python
xbound = [-54.0, 54.0, 0.3]  # x范围和分辨率
ybound = [-54.0, 54.0, 0.3]  # y范围和分辨率
zbound = [-10.0, 10.0, 20.0] # z范围和分辨率
dbound = [1.0, 60.0, 0.5]    # 深度范围和分辨率
```

### 图像参数

```python
image_size = [256, 704]      # 输入图像大小
feature_size = [32, 88]      # 特征图大小
```

### Voxelization参数

```python
voxel_size = [0.3, 0.3, 8.0]  # 体素大小
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]  # 点云范围
max_voxels = 10000            # 最大体素数
```

## CPU计算部分详解

### 1. 深度图生成

从点云投影到图像平面，生成每个相机视角的深度图：

```python
def generate_depth_map(points, img_aug_matrix, lidar_aug_matrix, lidar2image):
    """
    Args:
        points: [M, 3+] 点云
        img_aug_matrix: [N, 4, 4] 图像增强矩阵
        lidar_aug_matrix: [4, 4] lidar增强矩阵
        lidar2image: [N, 4, 4] lidar到图像的变换矩阵
    
    Returns:
        depth: [B, N, 1, H, W] 深度图
    """
```

### 2. 几何特征计算

计算每个像素在lidar坐标系下的位置：

```python
def get_geometry(camera2lidar_rots, camera2lidar_trans, intrins, post_rots, post_trans):
    """
    Args:
        camera2lidar_rots: [B, N, 3, 3] 相机到lidar的旋转
        camera2lidar_trans: [B, N, 3] 相机到lidar的平移
        intrins: [B, N, 3, 3] 相机内参
        post_rots: [B, N, 3, 3] 后处理旋转
        post_trans: [B, N, 3] 后处理平移
    
    Returns:
        geom_feats: [B, N, D, H, W, 3] 几何特征
    """
```

### 3. BEV Pool

将相机特征投影到BEV空间：

```python
def bev_pool(cam_feats, geom_feats):
    """
    Args:
        cam_feats: [B, N, C, D, H, W] 相机特征
        geom_feats: [B, N, D, H, W, 3] 几何特征
    
    Returns:
        bev_feat: [B, C*nx2, nx0, nx1] BEV特征
    """
```

## 性能优化建议

### 1. NPU部分

- 使用FP16精度减少内存占用和计算时间
- 合理设置dynamic batch gears以适应不同点云密度
- 使用多流并行执行雷达和相机分支

### 2. CPU部分

- 使用多线程并行处理多个相机视角
- 使用numpy向量化操作替代循环
- 考虑使用C++实现关键算子（BEV Pool）

### 3. 数据传输

- 减少NPU和CPU之间的数据传输
- 使用共享内存或零拷贝技术
- 批量处理多个样本以提高吞吐量

## 已知限制

1. **BEV Pool在CPU上实现**：由于scatter操作在ONNX导出时的兼容性问题，BEV Pool在CPU上实现，可能成为性能瓶颈。

2. **深度图生成**：从点云生成深度图涉及复杂的坐标变换，在CPU上计算较慢。

3. **几何特征计算**：涉及矩阵逆运算，在CPU上计算较慢。

## 未来改进方向

1. **自定义算子**：为BEV Pool、深度图生成、几何特征计算开发自定义NPU算子。

2. **模型优化**：将CPU计算部分融合到ONNX模型中，减少数据传输。

3. **流水线并行**：实现雷达分支和相机分支的流水线并行执行。

4. **量化**：使用INT8量化进一步减少计算量和内存占用。

## 参考

- 雷达分支导出：`src/export/export_bevfusion_full0306.py`
- 相机分支导出：`src/export/export_bevfusion_camera.py`
- 原始BEVFusion实现：`src/bevfusion/bevfusion.py`
- ViewTransform实现：`src/bevfusion/depth_lss.py`
