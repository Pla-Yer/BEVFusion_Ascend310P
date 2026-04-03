"""
BEVFusion Camera+Lidar OM Model Inference Base Class
This module provides the base class for running BEVFusion camera+lidar models on Ascend NPU.

Architecture:
    1. Lidar Branch (NPU): Voxelization -> VoxelEncoder -> Scatter -> Backbone -> Neck
    2. Camera Branch (NPU): Backbone -> Neck -> DepthNet
    3. CPU Parts: Depth generation, Geometry calculation, BEV Pool
    4. Fusion + Detection Head (NPU): Fusion -> Backbone -> Neck -> Head

Requirements:
    - acl (Ascend Computing Language)
    - numpy
    - torch (for CPU parts)

Usage:
    from bevfusion_camera_lidar_net import BEVFusionCameraLidarNet, init_acl

    ctx = init_acl(0)
    net = BEVFusionCameraLidarNet(
        lidar_model_path="./lidar_branch.om",
        camera_backbone_path="./camera_backbone.om",
        camera_neck_path="./camera_neck.om",
        camera_depthnet_path="./camera_depthnet.om",
        fusion_head_path="./fusion_head.om"
    )
    outputs = net.forward(points, imgs, metas)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import acl
import sys
import os
import time

# Error code constant
ACL_SUCCESS = 0


def check_ret(message, ret):
    """Check return value, print error and exit if failed."""
    if ret != ACL_SUCCESS:
        print(f"[Error] {message} failed, return code: {ret}")
        sys.exit(1)


class Net:
    """Base Ascend OM Model inference class."""

    def __init__(self, model_path, output_dtype=[np.float32], gears=None):
        """
        Initialize the OM model.

        Args:
            model_path: Path to the .om model file
            gears: List of supported dynamic batch sizes (optional)
        """
        self.gears = gears
        self.model_path = model_path
        self.output_dtype = output_dtype
        
        # Load model
        self.model_id, ret = acl.mdl.load_from_file(model_path)
        check_ret("acl.mdl.load_from_file", ret)

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)

        self.input_buffers = []
        self.output_buffers = []
        self._init_resource()

    def _init_resource(self):
        """Initialize input and output datasets with memory allocation."""
        # Create input dataset
        self.input_dataset = acl.mdl.create_dataset()
        input_count = acl.mdl.get_num_inputs(self.model_desc)
        print(f"[{self.model_path}] Input count: {input_count}")

        for i in range(input_count):
            size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            print(f"  Input {i} size: {size}")
            buf, ret = acl.rt.malloc(size, 0)
            check_ret(f"acl.rt.malloc input {i}", ret)
            db = acl.create_data_buffer(buf, size)
            _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, db)
            check_ret(f"acl.mdl.add_dataset_buffer input {i}", ret)
            self.input_buffers.append({"buffer": buf, "size": size})

        # Create output dataset
        self.output_dataset = acl.mdl.create_dataset()
        output_count = acl.mdl.get_num_outputs(self.model_desc)
        print(f"[{self.model_path}] Output count: {output_count}")

        for i in range(output_count):
            size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            print(f"  Output {i} size: {size}")
            buf, ret = acl.rt.malloc(size, 0)
            check_ret(f"acl.rt.malloc output {i}", ret)
            db = acl.create_data_buffer(buf, size)
            _, ret = acl.mdl.add_dataset_buffer(self.output_dataset, db)
            check_ret(f"acl.mdl.add_dataset_buffer output {i}", ret)
            self.output_buffers.append({"buffer": buf, "size": size})

    def forward(self, inputs, dynamic_dims=None):
        """
        Run inference on the model.

        Args:
            inputs: List of numpy arrays
            dynamic_dims: Dynamic dimensions dict (optional)

        Returns:
            List of numpy arrays (model outputs)
        """
        # Set dynamic dimensions if needed
        if dynamic_dims is not None:
            index, ret = acl.mdl.get_input_index_by_name(
                self.model_desc, "ascend_mbatch_shape_data")
            check_ret("get_dynamic_index", ret)
            ret = acl.mdl.set_input_dynamic_dims(
                self.model_id, self.input_dataset, index, dynamic_dims)
            check_ret("set_dynamic_dims", ret)

        # H2D data copy
        for i, data in enumerate(inputs):
            bytes_data = data.tobytes()
            ret = acl.rt.memcpy(self.input_buffers[i]["buffer"],
                                self.input_buffers[i]["size"],
                                acl.util.bytes_to_ptr(bytes_data),
                                len(bytes_data), 1)
            check_ret(f"memcpy H2D {i}", ret)

        # Execute inference
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        check_ret("execute", ret)

        # D2H result retrieval
        results = []
        for i in range(len(self.output_buffers)):
            size = self.output_buffers[i]["size"]
            host_ptr, ret = acl.rt.malloc_host(size)
            check_ret(f"malloc_host", ret)
            ret = acl.rt.memcpy(host_ptr, size, self.output_buffers[i]["buffer"], size, 2)
            check_ret(f"memcpy D2H", ret)
            data_bytes = acl.util.ptr_to_bytes(host_ptr, size)
            if self.output_dtype:
                outdtype = self.output_dtype[i] if i < len(self.output_dtype) else np.float32
            results.append(np.frombuffer(data_bytes, dtype=outdtype).copy())
            acl.rt.free_host(host_ptr)

        return results

    def __del__(self):
        """Release resources in destructor."""
        if hasattr(self, 'model_id'):
            acl.mdl.unload(self.model_id)
        if hasattr(self, 'model_desc') and self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)


class BEVPoolCPU:
    """CPU implementation of BEV Pool for camera features."""
    
    def __init__(self, bx, dx, nx):
        """
        Args:
            bx: [3] tensor, BEV grid origin
            dx: [3] tensor, BEV grid size
            nx: [3] tensor, BEV grid count
        """
        self.bx = bx
        self.dx = dx
        self.nx = nx
    
    def forward(self, x, geom_feats):
        """
        geom_feats: [B,N,D,H,W,3]  (float, lidar coords)
        x:         [B,N,D,H,W,C]  (float)
        return:    [B, C*nx2, nx0, nx1]
        """
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # ---- flatten feats ----
        feats = x.reshape(Nprime, C)  # [Nprime, C]
        coords = geom_feats.reshape(Nprime, 3)  # [Nprime, 3]

        # ---- grid index: floor((coord - (bx - dx/2)) / dx) ----
        # 这里严格对齐你原逻辑： (geom_feats - (bx - dx/2))/dx 然后 long()
        # 注意：self.bx/self.dx 建议是 shape=[3] 的 tensor（原代码就是）
        idx = ((coords - (self.bx - self.dx / 2.0)) / self.dx).to(torch.int64)  # [Nprime, 3]
        x_id = idx[:, 0]
        y_id = idx[:, 1]
        z_id = idx[:, 2]

        # ---- batch id（替代原来 python for + torch.full 拼接）----
        # 每个 batch 的点数固定 = Nprime // B
        pts_per_batch = Nprime // B
        b_id = torch.arange(B, device=feats.device, dtype=torch.int64).repeat_interleave(pts_per_batch)  # [Nprime]

        # ---- valid mask（替代 kept + 动态裁剪）----
        # self.nx 是 int tensor: [nx0, nx1, nx2]
        nx0 = self.nx[0].to(torch.int64)
        nx1 = self.nx[1].to(torch.int64)
        nx2 = self.nx[2].to(torch.int64)

        valid = (
                (x_id >= 0) & (x_id < nx0) &
                (y_id >= 0) & (y_id < nx1) &
                (z_id >= 0) & (z_id < nx2)
        )

        # mask 置零：等价于把 kept 外的点删掉再 sum
        feats = feats * valid.to(feats.dtype).unsqueeze(1)

        # clamp 防越界：无效点 feats=0 ，所以 clamp 到边界不会影响 sum
        x_id = x_id.clamp(0, nx0 - 1)
        y_id = y_id.clamp(0, nx1 - 1)
        z_id = z_id.clamp(0, nx2 - 1)

        # ---- linear index into (B, nx2, nx0, nx1) ----
        # lin = b*(nx2*nx0*nx1) + z*(nx0*nx1) + x*(nx1) + y
        stride_x = nx1
        stride_z = nx0 * nx1
        stride_b = nx2 * nx0 * nx1
        lin = b_id * stride_b + z_id * stride_z + x_id * stride_x + y_id  # [Nprime]

        # ---- scatter_reduce sum ----
        out_cells = B * nx2 * nx0 * nx1
        out = torch.zeros((out_cells, C), device=feats.device, dtype=feats.dtype)  # [cells, C]

        # scatter_reduce 需要 index 与 src 同 shape
        idx2 = lin.view(-1, 1).expand(-1, C)  # [Nprime, C]
        out = out.scatter_reduce(0, idx2, feats, reduce="sum", include_self=True)  # [cells, C]

        # ---- reshape back to [B, C*nx2, nx0, nx1] ----
        out = out.view(B, nx2, nx0, nx1, C).permute(0, 4, 1, 2, 3).contiguous()  # [B,C,nx2,nx0,nx1]
        final = out.reshape(B, C * nx2, nx0, nx1)  # [B,C*nx2,nx0,nx1]
        return final


class DepthGeometryCalculator:
    """CPU implementation of depth and geometry calculation."""
    
    def __init__(self, image_size, feature_size, xbound, ybound, zbound, dbound):
        """
        Args:
            image_size: [H, W] original image size
            feature_size: [fH, fW] feature map size
            xbound: [xmin, xmax, dx] x bounds
            ybound: [ymin, ymax, dy] y bounds
            zbound: [zmin, zmax, dz] z bounds
            dbound: [dmin, dmax, dd] depth bounds
        """
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound
        
        # Create frustum
        self.frustum = self._create_frustum()
        
        # Create BEV grid
        dx = torch.Tensor([xbound[2], ybound[2], zbound[2]])
        bx = torch.Tensor([xbound[0] + xbound[2]/2, ybound[0] + ybound[2]/2, zbound[0] + zbound[2]/2])
        nx = torch.LongTensor([(xbound[1]-xbound[0])/xbound[2],
                               (ybound[1]-ybound[0])/ybound[2],
                               (zbound[1]-zbound[0])/zbound[2]])
        self.dx = dx
        self.bx = bx
        self.nx = nx
    
    def _create_frustum(self):
        """Create frustum for depth estimation."""
        iH, iW = self.image_size
        fH, fW = self.feature_size
        
        # Depth bins
        ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D = ds.shape[0]
        
        # Image coordinates
        xs = torch.linspace(0, iW-1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, iH-1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        
        # Frustum: [D, fH, fW, 3] (x, y, d)
        frustum = torch.stack((xs, ys, ds), -1)
        return frustum
    
    def get_geometry(self, camera2lidar_rots, camera2lidar_trans, 
                     intrins, post_rots, post_trans,
                     extra_rots=None, extra_trans=None):
        """
        Calculate geometry features (lidar coordinates for each pixel).
        
        Args:
            camera2lidar_rots: [B, N, 3, 3]
            camera2lidar_trans: [B, N, 3]
            intrins: [B, N, 3, 3]
            post_rots: [B, N, 3, 3]
            post_trans: [B, N, 3]
            extra_rots: [B, 3, 3] (optional)
            extra_trans: [B, 3] (optional)
        
        Returns:
            geom_feats: [B, N, D, fH, fW, 3]
        """
        B, N, _ = camera2lidar_trans.shape
        
        # Undo post-transformation
        # points: [B, N, D, fH, fW, 3]
        # convert to tensor 
        post_trans = torch.tensor(post_trans, dtype=torch.float32)
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        
        # cam_to_lidar
        points = torch.cat([
            points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
            points[:, :, :, :, :, 2:3]
        ], dim=5)
        
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)
        
        # Apply extra transformation (lidar augmentation)
        if extra_rots is not None:
            points = extra_rots.view(B, 1, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        if extra_trans is not None:
            points += extra_trans.view(B, 1, 1, 1, 3)
        
        return points
    
    def generate_depth_map(self, points, img_aug_matrix, lidar_aug_matrix, 
                          lidar2image, B, N):
        """
        Generate depth map from point cloud.
        
        Args:
            points: list of [M, 3+] point clouds
            img_aug_matrix: [B, N, 4, 4]
            lidar_aug_matrix: [B, 4, 4]
            lidar2image: [B, N, 4, 4]
            B: batch size
            N: number of cameras
        
        Returns:
            depth: [B, N, 1, H, W]
        """
        iH, iW = self.image_size
        depth = torch.zeros(B, N, 1, iH, iW, dtype=torch.float32)
        
        for b in range(B):
            cur_coords = points[b][:, :3]  # [M, 3]
            cur_img_aug = img_aug_matrix[b]  # [N, 4, 4]
            cur_lidar_aug = lidar_aug_matrix[b]  # [4, 4]
            cur_lidar2img = lidar2image[b]  # [N, 4, 4]
            
            
            cur_coords -= cur_lidar_aug[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug[:3, :3]).matmul(
                cur_coords.transpose(1, 0))
            # lidar2image
            cur_coords = cur_lidar2img[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2img[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            cur_coords = cur_img_aug[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = ((cur_coords[..., 0] < self.image_size[0])
                      & (cur_coords[..., 0] >= 0)
                      & (cur_coords[..., 1] < self.image_size[1])
                      & (cur_coords[..., 1] >= 0))
            
            for c in range(N):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                if len(masked_coords) > 0:
                    depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist
        
        return depth


class BEVFusionCameraLidarNet:
    """
    BEVFusion Camera+Lidar inference class.
    
    Architecture (aligned with config):
        1. Lidar Branch (NPU): 
           - VoxelEncoder (PillarFeatureNet): [V,32,5] -> [V,256]
           - Scatter (PointPillarsScatter): [V,256] -> [B,256,360,360]
           - Backbone (SECOND): [B,256,360,360] -> [[B,128,180,180], [B,256,90,90]]
           - Neck (SECONDFPN): -> [B,256,180,180]
        
        2. Camera Branch (NPU):
           - Backbone (ResNet50): [B,N,3,256,704] -> [[B*N,512,32,88], [B*N,1024,16,44], [B*N,2048,8,22]]
           - Neck (GeneralizedLSSFPN): -> [[B*N,256,32,88], [B*N,256,16,44]]
           - DepthNet: [B,N,256,32,88] + depth -> [B,N,80,118,32,88]
        
        3. CPU Parts:
           - Depth generation: points -> [B,N,1,256,704]
           - Geometry calculation: -> [B,N,118,32,88,3]
           - BEV Pool: [B,N,80,118,32,88] -> [B,80,360,360]
        
        4. Fusion + Detection Head (NPU):
           - ConvFuser: [B,80,360,360] + [B,256,180,180] -> [B,256,180,180]
           - Backbone (SECOND): -> [[B,128,90,90], [B,256,45,45]]
           - Neck (SECONDFPN): -> [B,512,90,90]
           - TransFusionHead: -> detections
    """
    
    def __init__(self,
                 lidar_model_path,
                 camera_backbone_path,
                 camera_depthnet_path,
                 fusion_head_path,
                 # BEV parameters (from config)
                 xbound=[-54.0, 54.0, 0.3],
                 ybound=[-54.0, 54.0, 0.3],
                 zbound=[-10.0, 10.0, 20.0],
                 dbound=[1.0, 60.0, 0.5],
                 image_size=[256, 704],
                 feature_size=[32, 88],
                 # Voxelization parameters (from config)
                 voxel_size=[0.3, 0.3, 8.0],
                 point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
                 max_voxels=10000,
                 # Network parameters (from config)
                 lidar_channels=256,
                 camera_bev_channels=80,
                 fusion_channels=256,
                 head_in_channels=512,
                 num_proposals=200,
                 num_classes=10,
                 gears=[6000, 8000, 10000]):
        """
        Initialize BEVFusion Camera+Lidar inference.
        
        Args:
            lidar_model_path: Path to lidar branch OM model
            camera_backbone_path: Path to camera backbone OM model
            camera_neck_path: Path to camera neck OM model
            camera_depthnet_path: Path to camera depthnet OM model
            fusion_head_path: Path to fusion+head OM model
            xbound, ybound, zbound: BEV grid bounds
            dbound: Depth bounds
            image_size: [H, W] original image size
            feature_size: [fH, fW] feature map size
            voxel_size: [vx, vy, vz] voxel size
            point_cloud_range: [xmin, ymin, zmin, xmax, ymax, zmax]
            max_voxels: Maximum number of voxels
            gears: Dynamic batch sizes for lidar model
        """
        # Store parameters
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound
        self.image_size = image_size
        self.feature_size = feature_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_voxels = max_voxels
        self.gears = gears
        
        # Initialize timing statistics
        self.timing_stats = {
            'lidar_branch': [],
            'camera_branch': [],
            'depth_generation': [],
            'geometry_calculation': [],
            'bev_pool': [],
            'fusion_head': [],
            'total_inference': []
        }
        
        # Initialize OM models
        print("Loading lidar model...")
        self.lidar_net = Net(lidar_model_path,output_dtype=[np.float32], gears=gears)
        
        print("Loading camera backbone...")
        self.camera_backbone = Net(camera_backbone_path)
        
        print("Loading camera depthnet...")
        self.camera_depthnet = Net(camera_depthnet_path)
        
        print("Loading fusion+head model...")
        self.fusion_head = Net(fusion_head_path,output_dtype=[np.float32, np.int64, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.int64 ]) 
        # Initialize CPU calculators
        self.depth_geom_calc = DepthGeometryCalculator(
            image_size, feature_size, xbound, ybound, zbound, dbound)
        
        self.bev_pool = BEVPoolCPU(
            self.depth_geom_calc.bx,
            self.depth_geom_calc.dx,
            self.depth_geom_calc.nx
        )
        self.dowmsample = nn.Identity()
        
        print("BEVFusion Camera+Lidar initialized successfully!")
    
    def forward(self, voxels, num_points, coords, imgs, metas):
        """
        Forward pass.
        
        Args:
            voxels: [V, M, C] voxel features (M=32, C=5)
            num_points: [V] number of points per voxel
            coords: [V, 4] voxel coordinates (batch_id, z, y, x)
            imgs: [B, N, 3, H, W] images (N=6 cameras, H=256, W=704)
            metas: dict containing:
                - lidar2image: [B, N, 4, 4]
                - cam2img: [B, N, 4, 4]
                - cam2lidar: [B, N, 4, 4]
                - img_aug_matrix: [B, N, 4, 4]
                - lidar_aug_matrix: [B, 4, 4]
                - points: list of point clouds
        
        Returns:
            Detection outputs (same as TransFusionHead outputs)
        """
        start_time = time.time()
        
        # =====================================================================
        # 1. Lidar Branch (NPU)
        # =====================================================================
        lidar_start_time = time.time()
        print("Running lidar branch...")
        print(f"  Input: voxels {voxels.shape}, coords {coords.shape}")
        
        actual_voxel_num = voxels.shape[0]
        target_gear = next((g for g in self.gears if g >= actual_voxel_num), self.gears[-1])
        
        # Pad inputs
        def pad_to_gear(data, target_gear):
            pad_size = target_gear - data.shape[0]
            if pad_size <= 0:
                return data
            pad_shape = (pad_size,) + data.shape[1:]
            pad = np.zeros(pad_shape, dtype=data.dtype)
            return np.concatenate([data, pad], axis=0)
        
        voxels = pad_to_gear(voxels, target_gear)
        num_points = pad_to_gear(num_points, target_gear)
        coords = pad_to_gear(coords, target_gear)
        
        # Run lidar model
        # Input: [V, 32, 5], [V], [V, 4]
        # Output: [B, 256, 360, 360] (after Backbone + Neck)
        dynamic_dims = {
            'name': '',
            'dimCount': 6,
            'dims': [target_gear, 32, 5, target_gear, target_gear, 4]
        }
        lidar_outputs = self.lidar_net.forward(
            [voxels, num_points, coords],
            dynamic_dims
        )
        
        # lidar_outputs: [bev_feat] shape [B, 256, 360, 360]
        lidar_bev_feat = lidar_outputs[0]
        lidar_bev_feat = lidar_bev_feat.reshape(1, -1, 360, 360)  # reshape to [B, C, H, W]
        print(f"  Output: lidar_bev_feat {lidar_bev_feat.shape}")
        self.timing_stats['lidar_branch'].append(time.time() - lidar_start_time)
        
        # =====================================================================
        # 2. Camera Branch (NPU)
        # =====================================================================
        camera_start_time = time.time()
        print("Running camera branch...")
        B, N, C, H, W = imgs.shape
        print(f"  Input: imgs {imgs.shape}")
        
        # 2.1 Camera Backbone (ResNet50)
        # Input: [B, N, 3, 256, 704]
        # Output: [feat1, feat2, feat3]
        #   - feat1: [B*N, 512, 32, 88]
        #   - feat2: [B*N, 1024, 16, 44]
        #   - feat3: [B*N, 2048, 8, 22]
        camera_backbone_outputs = self.camera_backbone.forward([imgs])
        feat = camera_backbone_outputs[0]
        feat = feat.reshape(B,N, -1, *self.feature_size)
        print(f"  Camera backbone output feat1: {feat.shape}")
        self.timing_stats['camera_branch'].append(time.time() - camera_start_time)
        
        # =====================================================================
        # 3. CPU Parts: Depth + Geometry + BEV Pool
        # =====================================================================
        cpu_start_time = time.time()
        print("Running CPU parts...")
        
        # Convert to torch tensors for CPU computation
        # neck_feat1: [B*N, 256, 32, 88] -> [B, N, 256, 32, 88]
        neck_feat1_torch = torch.from_numpy(feat).view(B, N, -1, *self.feature_size)
        
        # 3.1 Generate depth map
        # Input: points [M, 3+]
        # Output: depth [B, N, 1, 256, 704]
        depth_gen_start_time = time.time()
        depth = self.depth_geom_calc.generate_depth_map(
            metas['points'],
            metas['img_aug_matrix'],
            metas['lidar_aug_matrix'],
            metas['lidar2image'],
            B, N
        )
        self.timing_stats['depth_generation'].append(time.time() - depth_gen_start_time)
        print(f"  Depth map: {depth.shape}")
        
        # 3.2 Run DepthNet
        # Input: img_feat [B, N, 256, 32, 88], depth [B, N, 1, 256, 704]
        # Output: cam_feats [B, N, 80, 118, 32, 88]
        depth_np = depth.numpy()
        camera_depthnet_outputs = self.camera_depthnet.forward([neck_feat1_torch.numpy(), depth_np])
        cam_feats = camera_depthnet_outputs[0]
        cam_feats = cam_feats.reshape(B, N,118, *self.feature_size,80)
        print(f"  DepthNet output: {cam_feats.shape}")
        
        # 3.3 Calculate geometry
        # Output: geom_feats [B, N, 118, 32, 88, 3]
        geom_calc_start_time = time.time()
        cam_feats_torch = torch.from_numpy(cam_feats)
        geom_feats = self.depth_geom_calc.get_geometry(
            metas['cam2lidar'][..., :3, :3],  # camera2lidar_rots
            metas['cam2lidar'][..., :3, 3],   # camera2lidar_trans
            metas['cam2img'][..., :3, :3],    # intrins
            metas['img_aug_matrix'][..., :3, :3],  # post_rots
            metas['img_aug_matrix'][..., :3, 3],   # post_trans
            metas['lidar_aug_matrix'][..., :3, :3],  # extra_rots
            metas['lidar_aug_matrix'][..., :3, 3]    # extra_trans
        )
        self.timing_stats['geometry_calculation'].append(time.time() - geom_calc_start_time)
        print(f"  Geometry features: {geom_feats.shape}")
        
        # 3.4 BEV Pool
        # Input: cam_feats [B, N, 80, 118, 32, 88], geom_feats [B, N, 118, 32, 88, 3]
        # Output: camera_bev_feat [B, 80, 360, 360]
        bev_pool_start_time = time.time()
        camera_bev_feat = self.bev_pool.forward(cam_feats_torch, geom_feats)
        self.timing_stats['bev_pool'].append(time.time() - bev_pool_start_time)
        print(f"  Camera BEV feature: {camera_bev_feat.shape}")
        
        self.timing_stats['cpu_parts'] = [time.time() - cpu_start_time]
        
        # =====================================================================
        # 4. Fusion + Detection Head (NPU)
        # =====================================================================
        fusion_start_time = time.time()
        print("Running fusion+head...")
        
        # 4.1 Feature Fusion (ConvFuser)
        # lidar_bev_feat: [B, 256, 180, 180]
        # camera_bev_feat: [B, 80, 360, 360] -> downsample to [B, 80, 180, 180]
        # Concat: [B, 336, 180, 180]
        # Conv: [B, 256, 180, 180]
        
        # Downsample camera_bev_feat
        camera_bev_feat_down = self.dowmsample(camera_bev_feat).numpy()

        
        # 4.2 Run fusion+head model
        # Input: fused_feat [B, 256, 180, 180]
        # Output: TransFusionHead outputs
        #   - dense_heatmap: [B, 10, 90, 90]
        #   - top_cls: [B, K]
        #   - query_heatmap_score: [B, 10, K]
        #   - heatmap_q: [B, 10, K]
        #   - center: [B, 2, K]
        #   - height: [B, 1, K]
        #   - dim: [B, 3, K]
        #   - rot: [B, 2, K]
        #   - vel: [B, 2, K]
        print("  Fusing features and running detection head...")
        print(f"  Lidar BEV feat: {lidar_bev_feat.shape}, Camera BEV feat (downsampled): {camera_bev_feat_down.shape}")

        fusion_outputs = self.fusion_head.forward([lidar_bev_feat,camera_bev_feat_down])
        
        self.timing_stats['fusion_head'].append(time.time() - fusion_start_time)
        self.timing_stats['total_inference'].append(time.time() - start_time)
        
        return fusion_outputs

    def print_timing_summary(self):
        """Print timing statistics summary."""
        print("\n" + "="*60)
        print("NET TIMING STATISTICS SUMMARY")
        print("="*60)

        stats = {}
        for task_name, times in self.timing_stats.items():
            if times:
                stats[task_name] = {
                    'total': sum(times),
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'count': len(times)
                }

        print(f"\n{'Task':<25} {'Total(s)':<10} {'Mean(ms)':<10} {'Std(ms)':<10} {'Min(ms)':<10} {'Max(ms)':<10} {'Count':<5}")
        print("-" * 80)

        for task_name, s in stats.items():
            print(f"{task_name:<25} {s['total']:<10.3f} {s['mean']*1000:<10.2f} "
                  f"{s['std']*1000:<10.2f} {s['min']*1000:<10.2f} {s['max']*1000:<10.2f} {s['count']:<5}")

        print("="*60)


def init_acl(device_id=0):
    """
    Initialize ACL runtime.

    Args:
        device_id: Ascend device ID

    Returns:
        ACL context
    """
    acl.init()
    acl.rt.set_device(device_id)
    context, _ = acl.rt.create_context(device_id)
    return context


def torch_dtype_to_numpy(dtype):
    """Convert PyTorch dtype to numpy dtype"""
    import torch
    if dtype == torch.float32:
        return np.float32
    elif dtype == torch.float64:
        return np.float64
    elif dtype == torch.float16:
        return np.float16
    elif dtype == torch.int32:
        return np.int32
    elif dtype == torch.int64:
        return np.int64
    elif dtype == torch.int16:
        return np.int16
    elif dtype == torch.int8:
        return np.int8
    elif dtype == torch.uint8:
        return np.uint8
    else:
        # 默认返回float32或其他合适的数据类型
        return np.float32