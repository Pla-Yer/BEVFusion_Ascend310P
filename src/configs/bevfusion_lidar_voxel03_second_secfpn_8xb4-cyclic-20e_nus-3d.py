# 基础配置文件
_base_ = ['_base_/default_runtime.py']

# 自定义导入模块，导入BEVFusion相关组件
custom_imports = dict(
    imports=['bevfusion'], allow_failed_imports=False)

# 模型设置
# 体素大小，用于体素编码器
# 通常体素大小与点云范围一致变化
# 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
voxel_size = [0.30, 0.30, 8.0]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

# grid size: (x,y,z) = (360,360,1)
grid_size = [360, 360, 1]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


# 修改元信息以使用NuScenes mini版本
metainfo = dict(classes=class_names, version='v1.0-mini')
# metainfo = dict(classes=class_names, version='v1.0')
dataset_type = 'NuScenesDataset'
# 数据根目录
data_root = 'data/nuscenes-mini/'
# 数据前缀配置，指定各类数据的存储路径
data_prefix = dict(
    pts='samples/LIDAR_TOP',           # LiDAR点云数据路径
    CAM_FRONT='samples/CAM_FRONT',     # 前摄像头图像路径
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',   # 前左摄像头图像路径
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT', # 前右摄像头图像路径
    CAM_BACK='samples/CAM_BACK',       # 后摄像头图像路径
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',   # 后右摄像头图像路径
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',     # 后左摄像头图像路径
    sweeps='sweeps/LIDAR_TOP')         # sweep LiDAR数据路径
# 输入模态配置，这里只使用LiDAR数据，不使用相机数据
input_modality = dict(use_lidar=True, use_camera=False)
# 后端参数，设为None表示使用本地文件系统
backend_args = None

# 模型配置
model = dict(
    # 模型类型为BEVFusion
    type='BEVFusion',
    # 数据预处理器配置
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        pad_size_divisor=32,  # 填充尺寸除数，确保特征图尺寸可被整除
        # 体素化配置
        voxelize_cfg=dict(
            max_num_points=32,  # 每个体素内的最大点数
            point_cloud_range=point_cloud_range,  # 点云范围
            voxel_size=voxel_size,  # 体素大小
            max_voxels=[8000, 10000],     # 训练和测试时的最大体素数
            voxelize_reduce=False)),          # 是否减少体素点数
    # 点云体素编码器
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=5,  # 你点特征是5维
        feat_channels=[64,128,256],
        with_distance=False,
        legacy=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
    ),
    # BEV特征编码器
    pts_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=256,
        output_shape=[360, 360],            # 108 / 0.3 = 360
    ),
    # 点云骨干网络(SecondNet)
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,            # 输入通道数
        out_channels=[128, 256],    # 输出通道数列表
        layer_nums=[5, 5],          # 每层卷积层数
        layer_strides=[2, 2],       # 每层步长
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),  # 批归一化配置
        conv_cfg=dict(type='Conv2d', bias=False)),  # 卷积配置
    # 点云颈部网络(SECOND FPN)
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],     # 输入通道数列表
        out_channels=[256, 256],    # 输出通道数列表
        upsample_strides=[1, 2],    # 上采样步长
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),  # 归一化配置
        upsample_cfg=dict(type='deconv', bias=False),  # 上采样配置
        use_conv_for_no_stride=True),  # 对于无步长情况是否使用卷积
    # 检测头配置(TransFusionHead)
    bbox_head=dict(
        type='TransFusionHead',
        num_proposals=200,          # 建议框数量
        auxiliary=True,             # 是否使用辅助任务
        in_channels=512,            # 输入通道数
        hidden_channel=128,         # 隐藏层通道数
        num_classes=10,             # 类别数
        nms_kernel_size=3,          # NMS核大小
        bn_momentum=0.1,            # BN动量
        num_decoder_layers=1,       # 解码器层数
        # 解码器层配置
        decoder_layer=dict(
            type='TransformerDecoderLayer',
            # 自注意力配置
            self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            # 交叉注意力配置
            cross_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            # 前馈网络配置
            ffn_cfg=dict(
                embed_dims=128,                    # 嵌入维度
                feedforward_channels=256,          # 前馈网络通道数
                num_fcs=2,                         # 全连接层数
                ffn_drop=0.1,                      # Dropout率
                act_cfg=dict(type='ReLU', inplace=True),  # 激活函数配置
            ),
            norm_cfg=dict(type='LN'),              # 归一化配置
            # 位置编码配置
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128)),
        # 训练配置
        train_cfg=dict(
            dataset='nuScenes',                    # 数据集名称
            point_cloud_range=point_cloud_range,  # 点云范围
            grid_size=grid_size,            # 网格大小
            voxel_size=voxel_size,        # 体素大小
            out_size_factor=2,                     # 输出尺寸因子
            gaussian_overlap=0.1,                  # 高斯重叠阈值
            min_radius=2,                          # 最小半径
            pos_weight=-1,                         # 正样本权重
            # 损失权重配置
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            # 分配器配置
            assigner=dict(
                type='HungarianAssigner3D',
                # IoU计算器配置
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                # 分类成本配置
                cls_cost=dict(
                    type='mmdet.FocalLossCost',
                    gamma=2.0,
                    alpha=0.25,
                    weight=0.15),
                # 回归成本配置
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                # IoU成本配置
                iou_cost=dict(type='IoU3DCost', weight=0.25))),
        # 测试配置
        test_cfg=dict(
            dataset='nuScenes',                    # 数据集名称
            grid_size=grid_size,            # 网格大小
            out_size_factor=2,                     # 输出尺寸因子
            voxel_size=[0.30, 0.30],             # 体素大小(x,y方向)
            pc_range=[-54.0, -54.0],               # 点云范围
            nms_type=None),                        # NMS类型
        # 通用头部配置
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2]),
        # 边界框编码器配置
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[-54.0, -54.0],               # 点云范围
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],  # 后处理中心范围
            score_threshold=0.0,                   # 分数阈值
            out_size_factor=2,                     # 输出尺寸因子
            voxel_size=[0.30, 0.30],             # 体素大小
            code_size=10),                         # 编码尺寸
        # 分类损失配置(Focal Loss)
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,                      # 是否使用sigmoid
            gamma=2.0,                             # Focal Loss的gamma参数
            alpha=0.25,                            # Focal Loss的alpha参数
            reduction='mean',                      # 归约方式
            loss_weight=1.0),                      # 损失权重
        # 热力图损失配置(Gaussian Focal Loss)
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        # 边界框回归损失配置(L1 Loss)
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25)))

# 数据库采样器配置，用于数据增强
db_sampler = dict(
    data_root=data_root,                       # 数据根目录
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',  # 数据库信息路径
    rate=1.0,                                  # 采样率
    # 准备配置
    prepare=dict(
        filter_by_difficulty=[-1],             # 按难度过滤
        # 按最小点数过滤各类别
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,                       # 类别名称
    # 各类别的采样组配置
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    # 点加载器配置
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',                    # 坐标类型
        load_dim=5,                            # 加载维度
        use_dim=[0, 1, 2, 3, 4],               # 使用的维度
        backend_args=backend_args))

# 训练数据处理流水线
train_pipeline = [
    # 加载点云文件
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    # 从多个扫描中加载点云
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,                          # 扫描数量
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,                 # 是否填充空扫描
        remove_close=True,                     # 是否移除近处点
        backend_args=backend_args),
    # 加载3D标注
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,                     # 是否包含3D边界框
        with_label_3d=True,                    # 是否包含3D标签
        with_attr_label=False),                # 是否包含属性标签
    # 对象采样增强
    dict(type='ObjectSample', db_sampler=db_sampler),
    # 全局旋转缩放平移增强
    dict(
        type='GlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],          # 缩放比例范围
        rot_range=[-0.78539816, 0.78539816],   # 旋转角度范围(弧度)
        translation_std=0.5),                  # 平移标准差
    # BEVFusion随机翻转增强
    dict(type='BEVFusionRandomFlip3D'),
    # 点范围过滤
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # 对象范围过滤
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # 对象名称过滤
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    # 点云混洗
    dict(type='PointShuffle'),
    # 打包3D检测输入
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ],
        # 元数据键
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix'
        ])
]

# 测试数据处理流水线
test_pipeline = [
    # 加载点云文件
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    # 从多个扫描中加载点云
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    # 点范围过滤
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range),
    # 打包3D检测输入
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        # 元数据键
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats', 'num_views'
        ])
]

# 训练数据加载器配置
train_dataloader = dict(
    batch_size=1,                              # 批次大小从4改为2
    num_workers=1,                             # 工作线程数
    persistent_workers=True,                   # 是否保持工作线程
    sampler=dict(type='DefaultSampler', shuffle=True),  # 采样器配置
    # 数据集配置
    dataset=dict(
        type='CBGSDataset',                    # CBGS数据集类型(类别平衡分组采样)
        dataset=dict(
            type=dataset_type,                 # 数据集类型
            data_root=data_root,               # 数据根目录
            ann_file='nuscenes_infos_train.pkl',  # 标注文件
            pipeline=train_pipeline,            # 训练流水线
            metainfo=metainfo,                 # 元信息
            modality=input_modality,           # 输入模态
            test_mode=False,                   # 是否为测试模式
            data_prefix=data_prefix,           # 数据前缀
            use_valid_flag=True,               # 是否使用有效标志
            box_type_3d='LiDAR')))             # 3D框类型

# 验证数据加载器配置
val_dataloader = dict(
    batch_size=2,                              # 批次大小
    num_workers=4,                             # 工作线程数
    persistent_workers=True,                   # 是否保持工作线程
    drop_last=False,                           # 是否丢弃最后一个批次
    sampler=dict(type='DefaultSampler', shuffle=False),  # 采样器配置
    # 数据集配置
    dataset=dict(
        type=dataset_type,                     # 数据集类型
        data_root=data_root,                   # 数据根目录
        ann_file='nuscenes_infos_val.pkl',     # 标注文件
        pipeline=test_pipeline,                # 测试流水线
        metainfo=metainfo,                     # 元信息
        modality=input_modality,               # 输入模态
        data_prefix=data_prefix,               # 数据前缀
        test_mode=True,                        # 是否为测试模式
        box_type_3d='LiDAR',                   # 3D框类型
        backend_args=backend_args))            # 后端参数

# 测试数据加载器配置(与验证相同)
test_dataloader = val_dataloader

# 验证评估器配置
val_evaluator = dict(
    type='NuScenesMetric',                     # NuScenes评估指标
    data_root=data_root,                       # 数据根目录
    ann_file=data_root + 'nuscenes_infos_val.pkl',  # 标注文件
    metric='bbox',                             # 评估指标(bbox)
    backend_args=backend_args)                 # 后端参数

# 测试评估器配置(与验证相同)
test_evaluator = val_evaluator

# 可视化后端配置
vis_backends = [dict(type='LocalVisBackend')]
# 可视化器配置
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# 学习率设置
lr = 0.0001
# 参数调度器配置
param_scheduler = [
    # 学习率调度器
    # 在前8个epoch中，学习率从0增加到lr * 10
    # 在接下来的12个epoch中，学习率从lr * 10减少到lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=8,                               # 最大周期
        eta_min=lr * 10,                       # 最小学习率
        begin=0,                               # 开始epoch
        end=8,                                 # 结束epoch
        by_epoch=True,                         # 是否按epoch计算
        convert_to_iter_based=True),           # 是否转换为基于迭代的
    dict(
        type='CosineAnnealingLR',
        T_max=12,                              # 最大周期
        eta_min=lr * 1e-4,                     # 最小学习率
        begin=8,                               # 开始epoch
        end=20,                                # 结束epoch
        by_epoch=True,                         # 是否按epoch计算
        convert_to_iter_based=True),           # 是否转换为基于迭代的
    # 动量调度器
    # 在前8个epoch中，动量从0增加到0.85 / 0.95
    # 在接下来的12个epoch中，动量从0.85 / 0.95增加到1
    dict(
        type='CosineAnnealingMomentum',
        T_max=8,                               # 最大周期
        eta_min=0.85 / 0.95,                   # 最小动量
        begin=0,                               # 开始epoch
        end=8,                                 # 结束epoch
        by_epoch=True,                         # 是否按epoch计算
        convert_to_iter_based=True),           # 是否转换为基于迭代的
    dict(
        type='CosineAnnealingMomentum',
        T_max=12,                              # 最大周期
        eta_min=1,                             # 最小动量
        begin=8,                               # 开始epoch
        end=20,                                # 结束epoch
        by_epoch=True,                         # 是否按epoch计算
        convert_to_iter_based=True)
]

# 运行时设置
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=5)  # 训练配置
val_cfg = dict()                                                # 验证配置
test_cfg = dict()                                               # 测试配置

# 优化器包装器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),     # AdamW优化器
    clip_grad=dict(max_norm=35, norm_type=2))                   # 梯度裁剪

# 自动缩放学习率配置
#   - `enable` 表示是否启用自动缩放学习率
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)
# 日志处理器配置
log_processor = dict(window_size=50)

# 默认钩子配置
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),                # 日志钩子
    checkpoint=dict(type='CheckpointHook', interval=5))         # 检查点钩子
# 自定义钩子配置
custom_hooks = [dict(type='DisableObjectSampleHook', disable_after_epoch=15)]  # 禁用对象采样钩子