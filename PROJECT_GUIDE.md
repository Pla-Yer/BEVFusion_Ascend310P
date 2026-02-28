# BEVFusion 独立项目

这是从 mmdetection3d 仓库中提取出来的 BEVFusion 独立项目，支持完整的训练和推理功能。

## 项目结构

```
BEVFusion_standalone/
├── bevfusion/                    # 核心代码
│   ├── __init__.py
│   ├── bevfusion.py              # 主模型
│   ├── bevfusion_necks.py        # Neck 模块
│   ├── depth_lss.py              # 深度估计和 LSS 变换
│   ├── loading.py                # 数据加载
│   ├── sparse_encoder.py         # 稀疏编码器
│   ├── transformer.py            # Transformer 模块
│   ├── transforms_3d.py          # 3D 数据增强
│   ├── transfusion_head.py       # TransFusion 检测头
│   ├── utils.py                  # 工具函数
│   └── ops/                      # CUDA 扩展操作
│       ├── bev_pool/             # BEV Pooling
│       └── voxel/                # 体素化操作
├── configs/                      # 配置文件
│   ├── _base_/                   # 基础配置
│   ├── bevfusion_lidar_*.py      # 仅 LiDAR 配置
│   └── bevfusion_lidar-cam_*.py  # 多模态配置
├── tools/                        # 工具脚本
│   ├── train.py                  # 训练脚本
│   ├── test.py                   # 测试脚本
│   ├── inference.py              # 推理脚本
│   ├── dist_train.sh             # 分布式训练脚本
│   └── dist_test.sh              # 分布式测试脚本
├── demo/                         # 示例代码
├── setup.py                      # 安装脚本
├── requirements.txt              # 依赖
├── test_install.py               # 安装测试脚本
└── README.md                     # 原始 README
```

## 安装

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 编译 CUDA 扩展

```bash
# 确保已安装 CUDA 和 PyTorch
pip install -v -e .
```

### 3. 验证安装

```bash
python test_install.py
```

## 数据准备

请按照 mmdet3d 的数据准备流程准备 nuScenes 数据集：

```bash
# 数据目录结构
data/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-test/
└── v1.0-trainval/

# 生成数据信息文件
python -c "from nuscenes.nuscenes import NuScenes; nusc = NuScenes(dataroot='data/nuscenes/', version='v1.0-trainval')"
```

## 训练

### 单 GPU 训练

```bash
python tools/train.py configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
```

### 多 GPU 分布式训练

```bash
# 使用 4 个 GPU
bash tools/dist_train.sh configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py 4

# 使用 8 个 GPU
bash tools/dist_train.sh configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py 8
```

### 使用 AMP 混合精度训练

```bash
python tools/train.py configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py --amp
```

### 恢复训练

```bash
# 自动恢复
python tools/train.py configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py --resume auto

# 从指定 checkpoint 恢复
python tools/train.py configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py --resume work_dirs/bevfusion/epoch_10.pth
```

## 测试/评估

### 单 GPU 测试

```bash
python tools/test.py configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py work_dirs/bevfusion/epoch_20.pth
```

### 多 GPU 分布式测试

```bash
bash tools/dist_test.sh configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py work_dirs/bevfusion/epoch_20.pth 4
```

### 可视化结果

```bash
python tools/test.py configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py work_dirs/bevfusion/epoch_20.pth \
    --show \
    --show-dir results/vis \
    --task multi-modality_det
```

## 推理

### 使用推理脚本

```bash
python tools/inference.py \
    configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    work_dirs/bevfusion/epoch_20.pth \
    --points data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin \
    --out-dir results/
```

### 在代码中使用

```python
import torch
from mmengine.config import Config
from mmdet3d.registry import MODELS
import bevfusion  # 注册模块

# 加载配置
cfg = Config.fromfile('configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py')

# 构建模型
model = MODELS.build(cfg.model)
model = model.cuda()
model.eval()

# 加载权重
checkpoint = torch.load('work_dirs/bevfusion/epoch_20.pth')
model.load_state_dict(checkpoint['state_dict'])

# 推理
with torch.no_grad():
    # 准备输入数据
    # ...
    results = model.predict(inputs, data_samples)
```

## 配置说明

### 仅 LiDAR 配置

- `bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py`
  - 仅使用 LiDAR 点云
  - 体素大小: [0.075, 0.075, 0.2]
  - 训练 20 epochs

### 多模态配置

- `bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py`
  - 使用 LiDAR + Camera 多模态融合
  - 包含图像 backbone (Swin Transformer)
  - 训练 6 epochs

## 常见问题

### 1. CUDA 扩展编译失败

确保 CUDA 和 PyTorch 版本兼容：
```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

### 2. 内存不足

- 减小 batch_size
- 使用 AMP 混合精度训练
- 减小体素数量

### 3. 数据加载问题

确保数据路径正确：
```python
# 在配置文件中检查
data_root = 'data/nuscenes/'
```

## 注意事项

1. **CUDA 扩展**：本项目包含 CUDA 扩展，需要正确安装 CUDA 工具链
2. **依赖版本**：确保 mmcv、mmdet、mmdet3d 版本兼容
3. **配置文件**：配置文件中的路径可能需要根据实际情况调整

## 原始项目

本项目提取自 mmdetection3d 仓库的 BEVFusion 实现。

原始论文：BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's Eye View Representation
