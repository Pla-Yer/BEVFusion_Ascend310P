"""
BEVFusion 并行推理模块
================================================================================
用两条 ACL Stream 并行执行 lidar_branch 和 camera_branch，
Barrier 后串行执行 fusion_head。

NPU 时延理论收益：
  串行旧版 ≈ 30(lidar) + 100(camera) + 52(fusion) = 182ms
  并行新版 ≈ max(30, 100)            + 52         = 152ms  (-30ms/帧)

关键 ACL API：
  acl.rt.create_stream()                                 创建异步流
  acl.rt.memcpy_async(dst,sz,src,sz,kind,stream)         异步 H2D
  acl.mdl.execute_async(mid, in_ds, out_ds, stream)      异步推理
  acl.rt.synchronize_stream(stream)                      等待流完成

接口兼容性：
  forward() 接口与旧版 BEVFusionFullNPUNet.forward() 完全兼容，
  evaluator 无需修改任何业务逻辑，只需替换 net 实例。

LUT 缓存策略：
  首次 forward(geom_feats=...) 时自动构建并缓存，
  固定标定时后续帧复用，CPU 开销 ≈ 0。
  可提前调用 precompute_lut(geom_np) 预热。
"""

import time
import sys
import os
import numpy as np
import acl

# ── ACL 常量 ────────────────────────────────────────────────────────────────
ACL_SUCCESS           = 0
ACL_MEMCPY_H2D        = 1
ACL_MEMCPY_D2H        = 2
ACL_MEMCPY_D2D        = 3   # device → device，不过 PCIe，极快
ACL_MEM_MALLOC_NORMAL = 0


def _chk(msg: str, ret: int):
    if ret != ACL_SUCCESS:
        print(f'[ACL Error] {msg} failed, ret={ret}')
        sys.exit(1)


def init_acl(device_id: int = 0):
    """初始化 ACL 运行时，返回 context。"""
    acl.init()
    acl.rt.set_device(device_id)
    ctx, _ = acl.rt.create_context(device_id)
    print(f'[ACL] Initialized on device {device_id}')
    return ctx


# ══════════════════════════════════════════════════════════════════════════
#  底层 OM 封装（同时支持同步/异步两种推理模式）
# ══════════════════════════════════════════════════════════════════════════
class _OmModel:
    """
    单个 .om 模型的加载与推理封装。

    支持：
      infer_sync()   → 同步推理（H2D + execute + D2H）
      submit_async() → 异步提交（H2D + execute_async），不阻塞
      collect_output() → synchronize 后 D2H 取回结果
    """

    def __init__(self, model_path: str, output_dtypes: list = None):
        self.model_path    = model_path
        self.output_dtypes = output_dtypes or []

        self.model_id, ret = acl.mdl.load_from_file(model_path)
        _chk('load_from_file', ret)

        self.model_desc = acl.mdl.create_desc()
        _chk('get_desc', acl.mdl.get_desc(self.model_desc, self.model_id))

        self.input_buffers  = []
        self.output_buffers = []
        self._alloc_buffers()

        name = os.path.basename(model_path)
        n_in  = acl.mdl.get_num_inputs(self.model_desc)
        n_out = acl.mdl.get_num_outputs(self.model_desc)
        print(f'[{name}] inputs={n_in}  outputs={n_out}')

    def _alloc_buffers(self):
        self.input_dataset  = acl.mdl.create_dataset()
        self.output_dataset = acl.mdl.create_dataset()

        n_in = acl.mdl.get_num_inputs(self.model_desc)
        for i in range(n_in):
            sz = acl.mdl.get_input_size_by_index(self.model_desc, i)
            buf, ret = acl.rt.malloc(sz, ACL_MEM_MALLOC_NORMAL)
            _chk(f'malloc input[{i}]', ret)
            db = acl.create_data_buffer(buf, sz)
            _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, db)
            _chk(f'add input[{i}]', ret)
            self.input_buffers.append({'buf': buf, 'size': sz})
            print(f'  input[{i}] size={sz}  '
                  f'(fp32≈{sz/4:.0f}el / fp16≈{sz/2:.0f}el)')

        n_out = acl.mdl.get_num_outputs(self.model_desc)
        for i in range(n_out):
            sz = acl.mdl.get_output_size_by_index(self.model_desc, i)
            buf, ret = acl.rt.malloc(sz, ACL_MEM_MALLOC_NORMAL)
            _chk(f'malloc output[{i}]', ret)
            db = acl.create_data_buffer(buf, sz)
            _, ret = acl.mdl.add_dataset_buffer(self.output_dataset, db)
            _chk(f'add output[{i}]', ret)
            self.output_buffers.append({'buf': buf, 'size': sz})
            print(f'  output[{i}] size={sz}  '
                  f'(fp32≈{sz/4:.0f}el / fp16≈{sz/2:.0f}el)')

    # ── 同步推理 ──────────────────────────────────────────────────────────
    def infer_sync(self, inputs: list) -> list:
        """H2D → execute → D2H，完全同步，返回 numpy 列表。"""
        for i, data in enumerate(inputs):
            raw = data.tobytes()
            _chk(f'H2D[{i}]', acl.rt.memcpy(
                self.input_buffers[i]['buf'], self.input_buffers[i]['size'],
                acl.util.bytes_to_ptr(raw), len(raw), ACL_MEMCPY_H2D))

        _chk('execute', acl.mdl.execute(
            self.model_id, self.input_dataset, self.output_dataset))

        return self._d2h()

    # ── 异步提交 ──────────────────────────────────────────────────────────
    def submit_async(self, inputs: list, stream,
                     dynamic_gear: int = None):
        """
        异步提交：(填写 ascend_mbatch_shape_data) + memcpy_async(H2D) + execute_async。

        Args:
            inputs       : list of np.ndarray，与 ONNX input_names 顺序对应
            stream       : ACL stream
            dynamic_gear : int，当 OM 用 --dynamic_dims 编译时传入当前档位值。
                           内部直接把 [gear, gear, gear] 写入最后一个 buffer
                           （ascend_mbatch_shape_data），绕过 set_input_dynamic_dims API。
        """
        # 若有动态档位，把控制 tensor 追加到 inputs 末尾
        if dynamic_gear is not None:
            # ascend_mbatch_shape_data = 3×int32，对应三个动态输入的第 0 维
            ctrl = np.array([dynamic_gear, dynamic_gear, dynamic_gear],
                            dtype=np.int32)
            inputs = list(inputs) + [ctrl]

        for i, data in enumerate(inputs):
            raw = data.tobytes()
            _chk(f'memcpy_async H2D[{i}]', acl.rt.memcpy_async(
                self.input_buffers[i]['buf'], self.input_buffers[i]['size'],
                acl.util.bytes_to_ptr(raw), len(raw),
                ACL_MEMCPY_H2D, stream))

        _chk('execute_async', acl.mdl.execute_async(
            self.model_id, self.input_dataset, self.output_dataset, stream))

    def collect_output(self) -> list:
        """
        synchronize_stream 之后调用，同步 D2H 并返回 numpy 列表。
        不需要 stream 参数——D2H 使用普通同步 memcpy，数据已在 device。
        """
        return self._d2h()

    def _d2h(self) -> list:
        results = []
        for i, ob in enumerate(self.output_buffers):
            hp, ret = acl.rt.malloc_host(ob['size'])
            _chk(f'malloc_host[{i}]', ret)
            _chk(f'D2H[{i}]', acl.rt.memcpy(
                hp, ob['size'], ob['buf'], ob['size'], ACL_MEMCPY_D2H))
            raw = acl.util.ptr_to_bytes(hp, ob['size'])
            dt  = (self.output_dtypes[i]
                   if i < len(self.output_dtypes) else np.float32)
            results.append(np.frombuffer(raw, dtype=dt).copy())
            acl.rt.free_host(hp)
        return results

    def __del__(self):
        if hasattr(self, 'model_id'):
            acl.mdl.unload(self.model_id)
        if hasattr(self, 'model_desc') and self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)


# ══════════════════════════════════════════════════════════════════════════
#  BEVFusion 并行推理主类
# ══════════════════════════════════════════════════════════════════════════
class BEVFusionParallelNet:
    """
    BEVFusion 三模型并行推理（两条 ACL Stream）。

    推理时序：
      Stream 1 → lidar_branch  (async) ─┐
      Stream 2 → camera_branch (async) ─┤
                                         ├─ synchronize_stream ×2 (Barrier)
                                         └─ fusion_head (sync)

    LUT 缓存：
      首次 forward() 自动构建并缓存；固定标定时后续帧复用（CPU 开销≈0）。
      推荐提前调用 precompute_lut(geom_np) 预热避免首帧额外延迟。

    接口：
      与旧版 BEVFusionFullNPUNet.forward() 完全兼容，evaluator 无需修改。
    """

    OUTPUT_DTYPES_FUSION = [
        np.float32,  # dense_heatmap
        np.int64,    # top_cls
        np.float32,  # query_heatmap_score
        np.float32,  # heatmap_q
        np.float32,  # center
        np.float32,  # height
        np.float32,  # dim
        np.float32,  # rot
        np.float32,  # vel
        np.int64,    # top_idx
    ]

    OUTPUT_NAMES = [
        'dense_heatmap', 'top_cls', 'query_heatmap_score',
        'heatmap_q', 'center', 'height', 'dim', 'rot', 'vel', 'top_idx',
    ]

    def __init__(self,
                 lidar_om:  str,
                 camera_om: str,
                 fusion_om: str,
                 gears: list = None,
                 # BEV 网格参数（与训练配置保持一致）
                 xbound: tuple = (-54.0, 54.0, 0.3),
                 ybound: tuple = (-54.0, 54.0, 0.3),
                 zbound: tuple = (-10.0, 10.0, 20.0),
                 max_pts:     int = 16,
                 D:           int = 118,
                 feature_size: tuple = (32, 88),
                 image_size:   tuple = (256, 704),
                 num_cameras:  int = 6):
        """
        Args:
            lidar_om   : lidar_branch.om 路径
            camera_om  : camera_branch.om 路径
            fusion_om  : fusion_head.om 路径
            gears      : 动态体素档位列表（与 ATC --dynamic_dims 对应）
            xbound/ybound/zbound : BEV 网格范围 (min, max, step)
            max_pts    : 每个 voxel LUT 槽位上限（与 export 保持一致）
            D          : depth bins
            feature_size: (fH, fW) 相机特征图尺寸
            image_size  : (H, W)  原始图像尺寸
            num_cameras : 相机数量
        """
        self.gears    = gears or [6000, 8000, 10000]
        self.max_pts  = int(max_pts)
        self.D        = int(D)
        self.fH, self.fW = feature_size
        self.H,  self.W  = image_size
        self.N = int(num_cameras)

        # BEV 网格常量
        self._bx  = np.array([xbound[0] + xbound[2] / 2,
                               ybound[0] + ybound[2] / 2,
                               zbound[0] + zbound[2] / 2], dtype=np.float32)
        self._dx  = np.array([xbound[2], ybound[2], zbound[2]], dtype=np.float32)
        self._nx0 = int(round((xbound[1] - xbound[0]) / xbound[2]))
        self._nx1 = int(round((ybound[1] - ybound[0]) / ybound[2]))
        self._nx2 = int(round((zbound[1] - zbound[0]) / zbound[2]))
        self._out_cells = 1 * self._nx2 * self._nx0 * self._nx1

        # LUT 缓存
        self._lut_cache = None  # tuple(pool_lookup_np, pool_mask_np) | None

        # 加载三个 OM
        print(f'[ParallelNet] Loading lidar_branch  : {lidar_om}')
        self._lidar_om  = _OmModel(lidar_om)

        print(f'[ParallelNet] Loading camera_branch : {camera_om}')
        self._camera_om = _OmModel(camera_om)

        print(f'[ParallelNet] Loading fusion_head   : {fusion_om}')
        self._fusion_om = _OmModel(fusion_om,
                                   output_dtypes=self.OUTPUT_DTYPES_FUSION)

        # 创建三条独立 ACL Stream
        self._stream_lidar,  ret = acl.rt.create_stream()
        _chk('create lidar stream',  ret)
        self._stream_camera, ret = acl.rt.create_stream()
        _chk('create camera stream', ret)
        self._stream_fusion, ret = acl.rt.create_stream()
        _chk('create fusion stream', ret)

        # 时延统计
        self.timing = {
            'lut_build':    [],
            'submit':       [],   # 两路 submit_async 耗时（CPU，极短）
            'npu_parallel': [],   # 两路并行等待时间 ≈ max(lidar, camera)
            'fusion':       [],   # D2D async + execute_async + sync + D2H（< 1MB）
            'total':        [],
        }

        print(f'[ParallelNet] Initialized.')
        print(f'  BEV grid: {self._nx0}×{self._nx1}×{self._nx2}'
              f'  out_cells={self._out_cells}  max_pts={self.max_pts}')

        # 启动时立即校验 D2D buffer 尺寸，及早发现 fp16/fp32 不匹配
        # 若报错，需用 --output_type=FP32 重新编译 lidar_branch.om / camera_branch.om
        self._verify_d2d_sizes()

    # ── LUT 构建（向量化 numpy） ───────────────────────────────────────────
    def _build_lut_numpy(self, geom_np: np.ndarray):
        ppb       = self.N * self.D * self.fH * self.fW
        out_cells = self._out_cells
        max_pts   = self.max_pts
        nx0, nx1, nx2 = self._nx0, self._nx1, self._nx2

        coords = geom_np.reshape(-1, 3).astype(np.float32)
        idx    = ((coords - (self._bx - self._dx * 0.5)) / self._dx).astype(np.int64)
        x_id, y_id, z_id = idx[:, 0], idx[:, 1], idx[:, 2]

        valid = ((x_id >= 0) & (x_id < nx0) &
                 (y_id >= 0) & (y_id < nx1) &
                 (z_id >= 0) & (z_id < nx2))

        b_id = np.repeat(np.arange(1, dtype=np.int64), ppb)
        lin  = (b_id * (nx2 * nx0 * nx1)
                + np.clip(z_id, 0, nx2 - 1) * (nx0 * nx1)
                + np.clip(x_id, 0, nx0 - 1) * nx1
                + np.clip(y_id, 0, nx1 - 1))

        lookup = np.zeros((out_cells, max_pts), np.int64)
        mask   = np.zeros((out_cells, max_pts), np.float32)

        vpt   = np.where(valid)[0]
        vvox  = lin[vpt]
        order = np.argsort(vvox, kind='stable')
        spt, svox = vpt[order], vvox[order]

        is_new    = np.concatenate([[True], svox[1:] != svox[:-1]])
        seg_start = np.maximum.accumulate(
            np.where(is_new, np.arange(len(spt), dtype=np.int64), 0))
        slot = np.arange(len(spt), dtype=np.int64) - seg_start

        keep = slot < max_pts
        lookup[svox[keep], slot[keep]] = spt[keep]
        mask  [svox[keep], slot[keep]] = 1.0

        return lookup, mask

    def precompute_lut(self, geom_np: np.ndarray):
        """
        手动预构建并缓存 BEV pool LUT。

        推荐在首帧推理前调用（避免首帧额外延迟），
        也可在标定更新时调用刷新缓存。

        Args:
            geom_np : np.ndarray [B, N, D, fH, fW, 3]  float32
        """
        t0 = time.time()
        self._lut_cache = self._build_lut_numpy(geom_np.astype(np.float32))
        elapsed = time.time() - t0
        self.timing['lut_build'].append(elapsed)
        print(f'[ParallelNet] LUT precomputed in {elapsed*1000:.1f}ms'
              f'  (out_cells={self._out_cells}, max_pts={self.max_pts})')

    # ── gear padding ──────────────────────────────────────────────────────
    def _select_gear(self, v: int) -> int:
        for g in sorted(self.gears):
            if g >= v:
                return g
        return max(self.gears)

    @staticmethod
    def _pad(arr: np.ndarray, n: int) -> np.ndarray:
        p = n - arr.shape[0]
        if p <= 0:
            return arr
        return np.concatenate([arr, np.zeros((p,) + arr.shape[1:], arr.dtype)])

    # ── 主推理入口 ────────────────────────────────────────────────────────
    def forward(self,
                voxels:     np.ndarray,
                num_points: np.ndarray,
                coords:     np.ndarray,
                imgs:       np.ndarray,
                depth:      np.ndarray,
                geom_feats: np.ndarray = None) -> list:
        """
        并行推理（接口与旧版 BEVFusionFullNPUNet.forward 完全兼容）。

        Args:
            voxels     : [V, 32, 5]           float32，原始体素（未 pad）
            num_points : [V]                   float32
            coords     : [V, 4]               float32，(b,z,y,x)
            imgs       : [B, N, 3, H, W]      float32，归一化图像
            depth      : [1, N, 1, H, W]      float32，预计算深度图
            geom_feats : [B, N, D, fH, fW, 3] float32，预计算几何坐标
                         首次调用时必须传入；后续帧可省略（复用缓存 LUT）。

        Returns:
            list of np.ndarray，顺序见 OUTPUT_NAMES
        """
        t_start = time.time()

        # ── 1. LUT：首帧构建并缓存 ────────────────────────────────────────
        if self._lut_cache is None:
            if geom_feats is None:
                raise ValueError(
                    'LUT 未构建。首次调用必须传入 geom_feats，'
                    '或提前调用 precompute_lut(geom_np)。')
            t0 = time.time()
            self._lut_cache = self._build_lut_numpy(geom_feats.astype(np.float32))
            self.timing['lut_build'].append(time.time() - t0)
            print(f'[ParallelNet] LUT built (auto) in '
                  f'{self.timing["lut_build"][-1]*1000:.1f}ms  [cached]')

        pool_lookup, pool_mask = self._lut_cache

        # ── 2. pad voxels 到档位 ──────────────────────────────────────────
        V    = voxels.shape[0]
        gear = self._select_gear(V)
        voxels_p  = np.ascontiguousarray(self._pad(voxels.astype(np.float32),     gear))
        npts_p    = np.ascontiguousarray(self._pad(num_points.astype(np.float32), gear))
        coords_p  = np.ascontiguousarray(self._pad(coords.astype(np.float32),     gear))

        # ── 3. 准备两路输入 ────────────────────────────────────────────────
        lidar_inputs = [voxels_p, npts_p, coords_p]

        camera_inputs = [
            np.ascontiguousarray(imgs.astype(np.float32)),
            np.ascontiguousarray(depth.astype(np.float32)),
            np.ascontiguousarray(pool_lookup),   # int64
            np.ascontiguousarray(pool_mask),     # float32
        ]

        # ── 4. 异步提交两路 ────────────────────────────────────────────────
        # lidar_branch 用 --dynamic_dims 编译，传入当前档位值
        # ascend_mbatch_shape_data 会由 submit_async 自动追加到 inputs 末尾
        t0 = time.time()
        self._lidar_om .submit_async(lidar_inputs,  self._stream_lidar,
                                     dynamic_gear=gear)
        self._camera_om.submit_async(camera_inputs, self._stream_camera)
        self.timing['submit'].append(time.time() - t0)

        # ── 5. Barrier：等待两路均完成 ────────────────────────────────────
        # LiDAR 先完成（~30ms），第一个 sync 迅速返回；
        # Camera（~100ms）决定实际等待时间。
        t0 = time.time()
        _chk('sync lidar  stream', acl.rt.synchronize_stream(self._stream_lidar))
        _chk('sync camera stream', acl.rt.synchronize_stream(self._stream_camera))
        self.timing['npu_parallel'].append(time.time() - t0)

        # ── 6. D2D async + fusion execute_async（串入 _stream_fusion） ────
        # 全部异步入队：D2D 拷贝 → fusion execute，CPU 立即返回，
        # NPU 按队列顺序依次执行，消除同步 memcpy 的隐式 barrier。
        t0 = time.time()
        self._d2d_to_fusion_async(self._stream_fusion)
        _chk('fusion execute_async',
             acl.mdl.execute_async(
                 self._fusion_om.model_id,
                 self._fusion_om.input_dataset,
                 self._fusion_om.output_dataset,
                 self._stream_fusion))
        # 等待 fusion stream 完成（D2D + execute 合并计时）
        _chk('sync fusion stream', acl.rt.synchronize_stream(self._stream_fusion))
        self.timing['fusion'].append(time.time() - t0)

        # ── 7. D2H fusion 输出（检测结果，< 1MB，极快） ─────────────────
        results = self._fusion_om._d2h()

        t_total = time.time() - t_start
        self.timing['total'].append(t_total)

        print(f'[ParallelNet] parallel={self.timing["npu_parallel"][-1]*1000:.1f}ms'
              f'  fusion(D2D+exec)={self.timing["fusion"][-1]*1000:.1f}ms'
              f'  total={t_total*1000:.1f}ms')

        return results

    # ── D2D：BEV 特征图直接从 lidar/camera OM 输出 buffer 写入 fusion 输入 buffer ──
    def _verify_d2d_sizes(self):
        """
        在模型加载后调用，校验 lidar/camera 输出 buffer 尺寸
        与 fusion 输入 buffer 尺寸是否匹配。

        不匹配说明 ATC 未正确执行 --output_type=FP32，
        lidar/camera OM 输出 fp16（半尺寸），fusion OM 期望 fp32（全尺寸）。
        需要用 --output_type=FP32 重新编译 lidar_branch.om 和 camera_branch.om。
        """
        cam_out_sz = self._camera_om.output_buffers[0]['size']
        lid_out_sz = self._lidar_om .output_buffers[0]['size']
        fus_in0_sz = self._fusion_om.input_buffers[0]['size']   # camera_bev
        fus_in1_sz = self._fusion_om.input_buffers[1]['size']   # lidar_bev

        ok = True
        if cam_out_sz != fus_in0_sz:
            print(f'[ParallelNet][ERROR] camera_branch output size {cam_out_sz}'
                  f' != fusion_head input[0] size {fus_in0_sz}')
            if cam_out_sz * 2 == fus_in0_sz:
                print('  → camera_branch 输出 fp16，fusion_head 期望 fp32。'
                      '  请用 --output_type=FP32 重新编译 camera_branch.om')
            ok = False

        if lid_out_sz != fus_in1_sz:
            print(f'[ParallelNet][ERROR] lidar_branch output size {lid_out_sz}'
                  f' != fusion_head input[1] size {fus_in1_sz}')
            if lid_out_sz * 2 == fus_in1_sz:
                print('  → lidar_branch 输出 fp16，fusion_head 期望 fp32。'
                      '  请用 --output_type=FP32 重新编译 lidar_branch.om')
            ok = False

        if ok:
            print('[ParallelNet] D2D size check passed: '
                  f'camera_bev={cam_out_sz}B  lidar_bev={lid_out_sz}B')
        else:
            raise RuntimeError(
                'D2D buffer size mismatch detected.\n'
                '请用 --output_type=FP32 重新编译 lidar_branch.om 和 camera_branch.om。\n'
                '见 export_bevfusion_parallel.py 中打印的 ATC 命令。')

    def _d2d_to_fusion_async(self, stream):
        """
        Device-to-Device 异步内存拷贝：将 lidar_branch / camera_branch 的输出 buffer
        直接拷入 fusion_head 的对应输入 buffer，完全不过 PCIe / host。

        前提：lidar_branch.om 和 camera_branch.om 必须用 --output_type=FP32 编译，
        保证输出 buffer 尺寸与 fusion_head.om 的 fp32 输入 buffer 严格一致。
        否则会出现 fp16/fp32 尺寸不匹配导致的精度错误。

        fusion_head 输入顺序（与 export 脚本 input_names 一致）：
          input[0] = camera_bev
          input[1] = lidar_bev
        """
        # camera_bev: camera_om.output[0] → fusion_om.input[0]
        cam_out = self._camera_om.output_buffers[0]
        fus_in0 = self._fusion_om.input_buffers[0]
        _chk('D2D async camera_bev→fusion_in[0]',
             acl.rt.memcpy_async(
                 fus_in0['buf'], fus_in0['size'],
                 cam_out['buf'], cam_out['size'],
                 ACL_MEMCPY_D2D, stream))

        # lidar_bev: lidar_om.output[0] → fusion_om.input[1]
        lid_out = self._lidar_om.output_buffers[0]
        fus_in1 = self._fusion_om.input_buffers[1]
        _chk('D2D async lidar_bev→fusion_in[1]',
             acl.rt.memcpy_async(
                 fus_in1['buf'], fus_in1['size'],
                 lid_out['buf'], lid_out['size'],
                 ACL_MEMCPY_D2D, stream))

    # ── 时延报表 ─────────────────────────────────────────────────────────
    def print_timing_summary(self):
        print('\n' + '=' * 68)
        print('BEVFusionParallelNet  TIMING SUMMARY')
        print('=' * 68)
        fmt = '{:<16} {:>10.3f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}'
        print(f'{"Stage":<16} {"Total(s)":>10} {"Mean(ms)":>10}'
              f' {"Std(ms)":>10} {"Min(ms)":>10} {"Max(ms)":>10}')
        print('-' * 68)
        for name, arr in self.timing.items():
            if not arr:
                continue
            a = np.array(arr)
            print(fmt.format(name,
                             a.sum(),
                             a.mean() * 1e3,
                             a.std()  * 1e3,
                             a.min()  * 1e3,
                             a.max()  * 1e3))
        print('=' * 68)

    def __del__(self):
        for attr in ('_stream_lidar', '_stream_camera', '_stream_fusion'):
            s = getattr(self, attr, None)
            if s is not None:
                acl.rt.destroy_stream(s)