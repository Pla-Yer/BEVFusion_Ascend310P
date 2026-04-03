"""
BEVFusion 全NPU推理模块
================================================================================
新流程：一个 OM 模型完成全部 NPU 推理

  旧流程 (4个OM + CPU中间计算):
    lidar.om → [depth_gen(CPU) + geo_gen(CPU) + bevpool(CPU)] → camera_backbone.om
    → camera_depthnet.om → fusion_head.om

  新流程 (1个OM + 轻量预计算):
    [depth_gen(CPU)]               ← 每帧必做（点云不同）
    [geo_gen(CPU) → LUT(CPU)]      ← 固定标定时只做一次，net 自动缓存
    → bevfusion_full_npu.om        ← 全部NPU (含LiDAR+Camera+BEVPool+FusionHead)
                                      BEVPool 使用 Gather+Mul+ReduceSum，无 ScatterElements

OM模型输入 (7个张量):
    voxels     : [max_V, 32, 5]        体素特征（pad后）
    num_points : [max_V]                每体素点数
    coords     : [max_V, 4]            体素坐标 (b,z,y,x)
    imgs       : [B, N, 3, H, W]       相机图像
    depth      : [B, N, 1, H, W]       预计算深度图
    pool_lookup: [out_cells, max_pts]  int64   BEV pool Gather 查找表
    pool_mask  : [out_cells, max_pts]  float32 有效槽位掩码

Usage:
    from bevfusion_full_npu_net import BEVFusionFullNPUNet, init_acl

    ctx = init_acl(device_id=0)
    net = BEVFusionFullNPUNet(
        model_path='models/om/bevfusion_full_npu.om',
        gears=[6000, 8000, 10000],
    )

    # 固定标定时显式预热 LUT（可选，否则首帧自动构建并缓存）
    # net.precompute_lut(geom_feats_np)

    # 推理接口不变，geom_feats 第一次调用时自动转换为 LUT 并缓存
    outputs = net.forward(voxels, num_points, coords, imgs, depth, geom_feats)
"""

import time
import sys
import os
import numpy as np
import torch
import acl

# ─── ACL 常量 ────────────────────────────────────────────────────────────────
ACL_SUCCESS = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def check_ret(message: str, ret: int):
    """检查 ACL 返回值，非0则打印并退出。"""
    if ret != ACL_SUCCESS:
        print(f'[ACL Error] {message} failed, ret={ret}')
        sys.exit(1)


def init_acl(device_id: int = 0):
    """初始化 ACL 运行时，返回 context。"""
    acl.init()
    acl.rt.set_device(device_id)
    ctx, _ = acl.rt.create_context(device_id)
    print(f'[ACL] Initialized on device {device_id}')
    return ctx


# ═══════════════════════════════════════════════════════════════════════════
#  底层 OM 推理封装
# ═══════════════════════════════════════════════════════════════════════════

class _OmModel:
    """
    单个 .om 模型的加载与推理封装。
    支持静态 shape 和 dynamic_dims（通过 ascend_mbatch_shape_data）。
    """

    def __init__(self, model_path: str, output_dtypes=None, gears=None):
        """
        Args:
            model_path   : .om 文件路径
            output_dtypes: 输出张量 dtype 列表，默认全部 np.float32
            gears        : 动态体素档位列表（例如 [6000, 8000, 10000]）
        """
        self.model_path = model_path
        self.output_dtypes = output_dtypes or []
        self.gears = gears or []

        # 加载模型
        self.model_id, ret = acl.mdl.load_from_file(model_path)
        check_ret('acl.mdl.load_from_file', ret)

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret('acl.mdl.get_desc', ret)

        self.input_buffers = []
        self.output_buffers = []
        self._allocate_buffers()

    # ────────────────────────────────────────────────────────────────────
    def _allocate_buffers(self):
        """按模型描述分配 device 侧输入/输出缓冲区。"""
        # 输入
        self.input_dataset = acl.mdl.create_dataset()
        n_in = acl.mdl.get_num_inputs(self.model_desc)
        print(f'[{os.path.basename(self.model_path)}] inputs={n_in}')
        for i in range(n_in):
            size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            print(f'  input[{i}] size={size}')
            buf, ret = acl.rt.malloc(size, 0)
            check_ret(f'malloc input[{i}]', ret)
            db = acl.create_data_buffer(buf, size)
            _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, db)
            check_ret(f'add_dataset_buffer input[{i}]', ret)
            self.input_buffers.append({'buf': buf, 'size': size})

        # 输出
        self.output_dataset = acl.mdl.create_dataset()
        n_out = acl.mdl.get_num_outputs(self.model_desc)
        print(f'[{os.path.basename(self.model_path)}] outputs={n_out}')
        for i in range(n_out):
            size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            print(f'  output[{i}] size={size}')
            buf, ret = acl.rt.malloc(size, 0)
            check_ret(f'malloc output[{i}]', ret)
            db = acl.create_data_buffer(buf, size)
            _, ret = acl.mdl.add_dataset_buffer(self.output_dataset, db)
            check_ret(f'add_dataset_buffer output[{i}]', ret)
            self.output_buffers.append({'buf': buf, 'size': size})

    # ────────────────────────────────────────────────────────────────────
    def infer(self, inputs: list, dynamic_dims=None):
        """
        执行一次推理。

        Args:
            inputs      : list of np.ndarray，顺序与 ONNX 输入对应
            dynamic_dims: dict，动态维度描述，例如：
                          {'name': '', 'dimCount': 6,
                           'dims': [6000, 32, 5, 6000, 6000, 4]}

        Returns:
            list of np.ndarray（模型输出）
        """
        # 设置动态维度
        if dynamic_dims is not None:
            idx, ret = acl.mdl.get_input_index_by_name(
                self.model_desc, 'ascend_mbatch_shape_data')
            check_ret('get_input_index_by_name', ret)
            ret = acl.mdl.set_input_dynamic_dims(
                self.model_id, self.input_dataset, idx, dynamic_dims)
            check_ret('set_input_dynamic_dims', ret)

        # H2D 数据拷贝
        t0 = time.time()
        for i, data in enumerate(inputs):
            raw = data.tobytes()
            ret = acl.rt.memcpy(
                self.input_buffers[i]['buf'],
                self.input_buffers[i]['size'],
                acl.util.bytes_to_ptr(raw),
                len(raw),
                ACL_MEMCPY_HOST_TO_DEVICE
            )
            check_ret(f'memcpy H2D[{i}]', ret)
        t_h2d = time.time() - t0

        # NPU 推理
        t0 = time.time()
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        check_ret('acl.mdl.execute', ret)
        t_infer = time.time() - t0

        # D2H 结果拷贝
        t0 = time.time()
        results = []
        for i, ob in enumerate(self.output_buffers):
            size = ob['size']
            host_ptr, ret = acl.rt.malloc_host(size)
            check_ret('malloc_host', ret)
            ret = acl.rt.memcpy(
                host_ptr, size, ob['buf'], size, ACL_MEMCPY_DEVICE_TO_HOST)
            check_ret(f'memcpy D2H[{i}]', ret)
            raw_bytes = acl.util.ptr_to_bytes(host_ptr, size)
            dtype = (self.output_dtypes[i]
                     if i < len(self.output_dtypes) else np.float32)
            results.append(np.frombuffer(raw_bytes, dtype=dtype).copy())
            acl.rt.free_host(host_ptr)
        t_d2h = time.time() - t0

        return results, {'h2d': t_h2d, 'infer': t_infer, 'd2h': t_d2h}

    def __del__(self):
        if hasattr(self, 'model_id'):
            acl.mdl.unload(self.model_id)
        if hasattr(self, 'model_desc') and self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)


# ═══════════════════════════════════════════════════════════════════════════
#  BEVFusion 全NPU推理主类
# ═══════════════════════════════════════════════════════════════════════════

class BEVFusionFullNPUNet:
    """
    BEVFusion 全 NPU 推理类（单 OM 模型）。

    整合原来的 4 个 OM 模型为 1 个 OM。CPU 侧保留：
      - 深度图预计算（每帧）
      - geom_feats → BEV pool LUT 转换（首帧构建后自动缓存，固定标定只算一次）

    BEV pool LUT 缓存机制：
      - 首次调用 forward(geom_feats=...) 时自动构建并缓存
      - 后续帧若 geom_feats 相同（固定标定），直接复用缓存，CPU 开销 ≈ 0
      - 调用 precompute_lut(geom_feats) 可手动预热/刷新缓存

    参数:
        model_path   : bevfusion_full_npu.om 路径
        gears        : 动态体素档位列表（与 ATC 编译时 --dynamic_dims 对应）
        D            : depth bins（默认 118）
        feature_size : 特征图大小 [fH, fW]（默认 [32, 88]）
        image_size   : 原图大小 [H, W]（默认 [256, 704]）
        num_cameras  : 相机数量（默认 6）
        xbound/ybound/zbound : BEV 网格范围 (min, max, step)
        max_pts      : 每个 voxel LUT 槽位上限（默认 16，与 export 时保持一致）
    """

    # OM 模型输出 dtype（与 FusionHeadNetwork 输出顺序对应）
    OUTPUT_DTYPES = [
        np.float32,   # dense_heatmap
        np.int64,     # top_cls
        np.float32,   # query_heatmap_score
        np.float32,   # heatmap_q
        np.float32,   # center
        np.float32,   # height
        np.float32,   # dim
        np.float32,   # rot
        np.float32,   # vel
        np.int64,     # top_idx
    ]

    OUTPUT_NAMES = [
        'dense_heatmap', 'top_cls', 'query_heatmap_score',
        'heatmap_q', 'center', 'height', 'dim', 'rot', 'vel', 'top_idx'
    ]

    def __init__(self,
                 model_path: str,
                 gears: list = None,
                 D: int = 118,
                 feature_size: list = None,
                 image_size: list = None,
                 num_cameras: int = 6,
                 xbound: tuple = (-54.0, 54.0, 0.3),
                 ybound: tuple = (-54.0, 54.0, 0.3),
                 zbound: tuple = (-10.0, 10.0, 20.0),
                 max_pts: int = 16):

        self.gears = gears or [6000, 8000, 10000]
        self.D = D
        self.fH, self.fW = (feature_size or [32, 88])
        self.H,  self.W  = (image_size   or [256, 704])
        self.N = num_cameras
        self.max_pts = int(max_pts)

        # ── BEV 网格参数（用于 LUT 构建） ─────────────────────────────────
        self._bx = np.array([xbound[0] + xbound[2] / 2,
                              ybound[0] + ybound[2] / 2,
                              zbound[0] + zbound[2] / 2], dtype=np.float32)
        self._dx = np.array([xbound[2], ybound[2], zbound[2]], dtype=np.float32)
        self._nx0 = int(round((xbound[1] - xbound[0]) / xbound[2]))
        self._nx1 = int(round((ybound[1] - ybound[0]) / ybound[2]))
        self._nx2 = int(round((zbound[1] - zbound[0]) / zbound[2]))
        # B=1 固定（单样本推理）
        self._out_cells = 1 * self._nx2 * self._nx0 * self._nx1

        # ── LUT 缓存（首帧构建后不再重建） ────────────────────────────────
        self._lut_cache = None   # tuple(pool_lookup_np, pool_mask_np) | None

        print(f'[BEVFusionFullNPUNet] Loading OM: {model_path}')
        self._om = _OmModel(model_path,
                            output_dtypes=self.OUTPUT_DTYPES,
                            gears=self.gears)

        # 时延统计（新增 lut_build 桶）
        self.timing = {
            'lut_build': [], 'h2d': [], 'infer': [], 'd2h': [], 'total': []
        }

        print('[BEVFusionFullNPUNet] Initialized.')
        print(f'  BEV grid: nx0={self._nx0} nx1={self._nx1} nx2={self._nx2}'
              f'  out_cells={self._out_cells}  max_pts={self.max_pts}')

    # ────────────────────────────────────────────────────────────────────
    #  LUT 构建（CPU numpy 向量化，固定标定时只调用一次）
    # ────────────────────────────────────────────────────────────────────
    def _build_lut_numpy(self, geom_feats_np: np.ndarray):
        """
        从 geom_feats 向量化构建 BEV pool Gather 查找表。

        Args:
            geom_feats_np : np.ndarray [B, N, D, fH, fW, 3]  float32

        Returns:
            pool_lookup : np.ndarray [out_cells, max_pts]  int64
            pool_mask   : np.ndarray [out_cells, max_pts]  float32
        """
        B  = 1            # 单样本推理
        N  = self.N
        D  = self.D
        fH = self.fH
        fW = self.fW
        nx0, nx1, nx2 = self._nx0, self._nx1, self._nx2
        max_pts   = self.max_pts
        ppb       = N * D * fH * fW
        out_cells = self._out_cells

        coords = geom_feats_np.reshape(-1, 3).astype(np.float32)
        idx    = ((coords - (self._bx - self._dx * 0.5)) / self._dx).astype(np.int64)
        x_id, y_id, z_id = idx[:, 0], idx[:, 1], idx[:, 2]

        valid = (
            (x_id >= 0) & (x_id < nx0) &
            (y_id >= 0) & (y_id < nx1) &
            (z_id >= 0) & (z_id < nx2)
        )

        b_id = np.repeat(np.arange(B, dtype=np.int64), ppb)
        x_c  = np.clip(x_id, 0, nx0 - 1)
        y_c  = np.clip(y_id, 0, nx1 - 1)
        z_c  = np.clip(z_id, 0, nx2 - 1)

        lin = (b_id * (nx2 * nx0 * nx1)
               + z_c * (nx0 * nx1)
               + x_c * nx1
               + y_c)

        # ── 向量化槽位分配（argsort + segment-cumsum，无 Python 逐点循环） ─
        pool_lookup = np.zeros((out_cells, max_pts), dtype=np.int64)
        pool_mask   = np.zeros((out_cells, max_pts), dtype=np.float32)

        valid_pt_idx = np.where(valid)[0]           # [M]
        valid_vox    = lin[valid_pt_idx]             # [M]

        order      = np.argsort(valid_vox, kind='stable')
        sorted_pt  = valid_pt_idx[order]
        sorted_vox = valid_vox[order]

        # 组内偏移 = 全局位置 − 所在组的起始全局位置
        is_new          = np.concatenate([[True], sorted_vox[1:] != sorted_vox[:-1]])
        seg_start_pos   = np.where(is_new, np.arange(len(sorted_pt), dtype=np.int64), 0)
        seg_start_pos   = np.maximum.accumulate(seg_start_pos)
        slot_arr        = np.arange(len(sorted_pt), dtype=np.int64) - seg_start_pos

        keep       = slot_arr < max_pts
        final_pt   = sorted_pt[keep]
        final_vox  = sorted_vox[keep]
        final_slot = slot_arr[keep]

        pool_lookup[final_vox, final_slot] = final_pt
        pool_mask  [final_vox, final_slot] = 1.0

        return pool_lookup, pool_mask

    def precompute_lut(self, geom_feats_np: np.ndarray):
        """
        手动预构建并缓存 BEV pool LUT。

        推荐在首帧推理前显式调用（节省首帧额外延迟），
        也可作为标定更新时刷新缓存的入口。

        Args:
            geom_feats_np : np.ndarray [B, N, D, fH, fW, 3]  float32
        """
        t0 = time.time()
        pool_lookup, pool_mask = self._build_lut_numpy(geom_feats_np)
        self._lut_cache = (pool_lookup, pool_mask)
        elapsed = time.time() - t0
        self.timing['lut_build'].append(elapsed)
        print(f'[BEVFusionFullNPUNet] LUT precomputed in {elapsed*1000:.2f} ms  '
              f'(out_cells={self._out_cells}, max_pts={self.max_pts})')

    # ────────────────────────────────────────────────────────────────────
    def _select_gear(self, actual_v: int) -> int:
        """选择不小于 actual_v 的最小档位，若超出则取最大档。"""
        for g in sorted(self.gears):
            if g >= actual_v:
                return g
        return max(self.gears)

    @staticmethod
    def _pad_to(arr: np.ndarray, target: int) -> np.ndarray:
        """将 arr 在第 0 维 pad 到 target 长度。"""
        pad_n = target - arr.shape[0]
        if pad_n <= 0:
            return arr
        pad = np.zeros((pad_n,) + arr.shape[1:], dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)

    # ────────────────────────────────────────────────────────────────────
    def forward(self,
                voxels: np.ndarray,
                num_points: np.ndarray,
                coords: np.ndarray,
                imgs: np.ndarray,
                depth: np.ndarray,
                geom_feats: np.ndarray = None) -> list:
        """
        全 NPU 推理。接口与旧版兼容（仍接受 geom_feats）。

        Args:
            voxels     : [V, 32, 5]          float32，原始体素（未pad）
            num_points : [V]                  float32
            coords     : [V, 4]              float32，(b,z,y,x)
            imgs       : [B, N, 3, H, W]     float32，归一化后的图像
            depth      : [B, N, 1, H, W]     float32，预计算深度图
            geom_feats : [B, N, D, fH, fW, 3] float32，预计算几何坐标
                         首次调用时必须传入；后续帧可省略（使用缓存 LUT）。
                         固定标定场景下建议使用 precompute_lut() 显式预热后可省略。

        Returns:
            list of np.ndarray，顺序见 OUTPUT_NAMES
        """
        t_start = time.time()

        # ── 1. LUT：首帧构建并缓存，后续帧直接复用 ──────────────────────
        if self._lut_cache is None:
            if geom_feats is None:
                raise ValueError(
                    'LUT 尚未构建。首次调用 forward() 时必须传入 geom_feats，'
                    '或提前调用 precompute_lut(geom_feats)。')
            t0 = time.time()
            pool_lookup, pool_mask = self._build_lut_numpy(
                geom_feats.astype(np.float32))
            self._lut_cache = (pool_lookup, pool_mask)
            elapsed = time.time() - t0
            self.timing['lut_build'].append(elapsed)
            print(f'[BEVFusionFullNPUNet] LUT built in {elapsed*1000:.2f} ms '
                  f'(cached for subsequent frames)')
        pool_lookup, pool_mask = self._lut_cache

        # ── 2. 选择体素档位并 pad ────────────────────────────────────────
        actual_v = voxels.shape[0]
        gear = self._select_gear(actual_v)
        voxels_p    = self._pad_to(voxels.astype(np.float32),    gear)
        num_points_p = self._pad_to(num_points.astype(np.float32), gear)
        coords_p    = self._pad_to(coords.astype(np.float32),    gear)

        # ── 3. 准备输入列表（顺序必须与 ONNX export 的 input_names 一致） ─
        inputs = [
            np.ascontiguousarray(voxels_p),
            np.ascontiguousarray(num_points_p),
            np.ascontiguousarray(coords_p),
            np.ascontiguousarray(imgs.astype(np.float32)),
            np.ascontiguousarray(depth.astype(np.float32)),
            np.ascontiguousarray(pool_lookup),   # int64
            np.ascontiguousarray(pool_mask),     # float32
        ]

        # ── 4. 构造动态维度描述（与 ATC --dynamic_dims 对应） ─────────────
        #   dim 顺序：voxels(3) num_points(1) coords(2) imgs(5) depth(5)
        #             pool_lookup(2) pool_mask(2) = 20 dims
        out_cells = self._out_cells
        max_pts   = self.max_pts
        dynamic_dims = {
            'name': '',
            'dimCount': 20,
            'dims': [
                gear, 32, 5,             # voxels       [3]
                gear,                    # num_points   [1]
                gear, 4,                 # coords       [2]
                1, self.N, 3, self.H, self.W,   # imgs  [5]
                1, self.N, 1, self.H, self.W,   # depth [5]
                out_cells, max_pts,      # pool_lookup  [2]
                out_cells, max_pts,      # pool_mask    [2]
            ],
        }

        # ── 5. NPU 推理 ──────────────────────────────────────────────────
        results, timing_detail = self._om.infer(inputs, dynamic_dims=dynamic_dims)

        t_total = time.time() - t_start
        self.timing['h2d'].append(timing_detail['h2d'])
        self.timing['infer'].append(timing_detail['infer'])
        self.timing['d2h'].append(timing_detail['d2h'])
        self.timing['total'].append(t_total)

        print(f'[NPU] H2D={timing_detail["h2d"]*1000:.1f}ms  '
              f'Infer={timing_detail["infer"]*1000:.1f}ms  '
              f'D2H={timing_detail["d2h"]*1000:.1f}ms  '
              f'Total={t_total*1000:.1f}ms')

        return results

    # ────────────────────────────────────────────────────────────────────
    def print_timing_summary(self):
        """打印推理时延统计表。"""
        print('\n' + '=' * 62)
        print('BEVFusionFullNPUNet  TIMING SUMMARY')
        print('=' * 62)
        fmt = '{:<14} {:>10.3f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}'
        header = (f'{"Stage":<14} {"Total(s)":>10} {"Mean(ms)":>10}'
                  f' {"Std(ms)":>10} {"Min(ms)":>10} {"Max(ms)":>10}')
        print(header)
        print('-' * 62)
        for name, times in self.timing.items():
            if not times:
                continue
            arr = np.array(times)
            print(fmt.format(
                name,
                arr.sum(),
                arr.mean() * 1000,
                arr.std()  * 1000,
                arr.min()  * 1000,
                arr.max()  * 1000,
            ))
        print('=' * 62)