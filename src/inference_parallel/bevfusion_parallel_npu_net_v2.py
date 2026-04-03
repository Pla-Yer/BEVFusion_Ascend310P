"""
BEVFusion 并行 NPU 推理模块（v2，pipeline 优化版）
================================================================================
在上一版“三 OM 并行”的基础上，继续做两类零精度风险优化：

  1. pool_mask 改为 uint8 输入，模型内再 cast 到特征 dtype。
     - LUT 的 pool_mask 本质上是 0/1 掩码，用 uint8 存储不会影响数值，
       但可将该静态输入的 H2D / device buffer 占用降到原来的 1/4。

  2. 增加 host 侧双缓冲（double buffering）。
     - 为 voxels / num_points / coords / imgs / depth 预分配两个 host slot；
     - CPU 线程可以把“下一帧”的预处理结果先写入空闲 slot；
     - 主线程对“当前帧”做 NPU 推理时，不会与下一帧的 host staging 冲突。

并行拓扑：
    lidar_branch.om   --\
                         >-- fusion_head.om --> detections
    camera_branch.om  --/

关键点：
  1. LiDAR / Camera 两路仍然通过两个 stream 并发。
  2. Camera 分支 pool_lookup / pool_mask 只在首次推理时上传一次，后续复用。
  3. Fusion 模型前两路输入直接绑定 camera/lidar 分支输出的 device buffer，避免 D2H/H2D。
  4. 新增 stage_host_inputs()/forward_staged()，便于 evaluator 做预处理-推理流水化。
"""

import os
import threading
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import acl
import numpy as np

ACL_SUCCESS = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def check_ret(message: str, ret: int):
    if ret != ACL_SUCCESS:
        raise RuntimeError(f'[ACL Error] {message} failed, ret={ret}')


def init_acl(device_id: int = 0):
    """初始化 ACL 运行时并返回 context。"""
    ret = acl.init()
    check_ret('acl.init', ret)
    ret = acl.rt.set_device(device_id)
    check_ret('acl.rt.set_device', ret)
    ctx, ret = acl.rt.create_context(device_id)
    check_ret('acl.rt.create_context', ret)
    print(f'[ACL] Initialized on device {device_id}')
    return ctx


def _numpy_ptr(array: np.ndarray) -> int:
    """尽量零额外拷贝地拿到 numpy buffer 指针。"""
    if hasattr(acl, 'util') and hasattr(acl.util, 'numpy_to_ptr'):
        try:
            return acl.util.numpy_to_ptr(array)
        except Exception:
            pass
    return int(array.ctypes.data)


class _OmModel:
    """单个 .om 模型的加载/执行封装，支持显式 stream 与外部输入 buffer 绑定。"""

    def __init__(
        self,
        model_path: str,
        output_dtypes: Optional[Sequence[np.dtype]] = None,
        gears: Optional[Sequence[int]] = None,
        stream: Optional[int] = None,
        external_input_buffers: Optional[Dict[int, Dict[str, int]]] = None,
        owns_stream: bool = False,
        reuse_host_outputs: bool = True,
    ):
        self.model_path = model_path
        self.output_dtypes = list(output_dtypes or [])
        self.gears = list(gears or [])
        self.external_input_buffers = dict(external_input_buffers or {})
        self.stream = stream
        self.owns_stream = owns_stream
        self.reuse_host_outputs = reuse_host_outputs

        self.model_id, ret = acl.mdl.load_from_file(model_path)
        check_ret(f'acl.mdl.load_from_file({model_path})', ret)

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret('acl.mdl.get_desc', ret)

        self.input_buffers: List[Dict[str, int]] = []
        self.output_buffers: List[Dict[str, int]] = []
        self.host_output_buffers: List[Dict[str, int]] = []
        self._input_data_buffers: List[int] = []
        self._output_data_buffers: List[int] = []
        self.input_dataset = acl.mdl.create_dataset()
        self.output_dataset = acl.mdl.create_dataset()

        self._allocate_buffers()
        if self.reuse_host_outputs:
            self._allocate_host_output_buffers()

    def _allocate_buffers(self):
        n_in = acl.mdl.get_num_inputs(self.model_desc)
        print(f'[{os.path.basename(self.model_path)}] inputs={n_in}')
        for i in range(n_in):
            size = int(acl.mdl.get_input_size_by_index(self.model_desc, i))
            if i in self.external_input_buffers:
                buf_info = self.external_input_buffers[i]
                buf = buf_info['buf']
                if int(buf_info['size']) < size:
                    raise ValueError(
                        f'External input buffer[{i}] is too small: '
                        f'{buf_info["size"]} < {size}')
                owned = False
                print(f'  input[{i}] size={size} (external)')
            else:
                buf, ret = acl.rt.malloc(size, 0)
                check_ret(f'acl.rt.malloc input[{i}]', ret)
                owned = True
                print(f'  input[{i}] size={size}')

            db = acl.create_data_buffer(buf, size)
            _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, db)
            check_ret(f'acl.mdl.add_dataset_buffer input[{i}]', ret)
            self.input_buffers.append({
                'buf': buf,
                'size': size,
                'owned': owned,
                'external': int(i in self.external_input_buffers),
            })
            self._input_data_buffers.append(db)

        n_out = acl.mdl.get_num_outputs(self.model_desc)
        print(f'[{os.path.basename(self.model_path)}] outputs={n_out}')
        for i in range(n_out):
            size = int(acl.mdl.get_output_size_by_index(self.model_desc, i))
            buf, ret = acl.rt.malloc(size, 0)
            check_ret(f'acl.rt.malloc output[{i}]', ret)
            db = acl.create_data_buffer(buf, size)
            _, ret = acl.mdl.add_dataset_buffer(self.output_dataset, db)
            check_ret(f'acl.mdl.add_dataset_buffer output[{i}]', ret)
            self.output_buffers.append({'buf': buf, 'size': size, 'owned': True})
            self._output_data_buffers.append(db)
            print(f'  output[{i}] size={size}')

    def _allocate_host_output_buffers(self):
        for i, ob in enumerate(self.output_buffers):
            host_ptr, ret = acl.rt.malloc_host(ob['size'])
            check_ret(f'acl.rt.malloc_host output[{i}]', ret)
            self.host_output_buffers.append({'ptr': host_ptr, 'size': ob['size'], 'owned': True})

    def get_output_buffer(self, index: int) -> Dict[str, int]:
        return self.output_buffers[index]

    def _set_dynamic_dims(self, dynamic_dims: Optional[dict]):
        if dynamic_dims is None:
            return
        idx, ret = acl.mdl.get_input_index_by_name(
            self.model_desc, 'ascend_mbatch_shape_data')
        check_ret('acl.mdl.get_input_index_by_name', ret)
        ret = acl.mdl.set_input_dynamic_dims(
            self.model_id, self.input_dataset, idx, dynamic_dims)
        check_ret('acl.mdl.set_input_dynamic_dims', ret)

    def copy_input(self, index: int, array: np.ndarray):
        if array is None:
            raise ValueError(f'Input[{index}] is None but copy is required.')
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)

        buf_info = self.input_buffers[index]
        nbytes = int(array.nbytes)
        if nbytes > buf_info['size']:
            raise ValueError(
                f'Input[{index}] bytes overflow: {nbytes} > {buf_info["size"]}')

        ret = acl.rt.memcpy(
            buf_info['buf'],
            buf_info['size'],
            _numpy_ptr(array),
            nbytes,
            ACL_MEMCPY_HOST_TO_DEVICE,
        )
        check_ret(f'acl.rt.memcpy H2D[{index}]', ret)

    def copy_inputs(self, inputs: Sequence[Optional[np.ndarray]], skip_copy_indices: Optional[Iterable[int]] = None):
        skip = set(skip_copy_indices or [])
        t0 = time.time()
        for i, arr in enumerate(inputs):
            if i in skip:
                continue
            self.copy_input(i, arr)
        return time.time() - t0

    def execute_async(
        self,
        inputs: Sequence[Optional[np.ndarray]],
        dynamic_dims: Optional[dict] = None,
        skip_copy_indices: Optional[Iterable[int]] = None,
    ) -> dict:
        if self.stream is None:
            raise RuntimeError(f'{self.model_path} has no stream for async execution.')
        self._set_dynamic_dims(dynamic_dims)
        t_h2d = self.copy_inputs(inputs, skip_copy_indices=skip_copy_indices)
        t0 = time.time()
        ret = acl.mdl.execute_async(
            self.model_id, self.input_dataset, self.output_dataset, self.stream)
        check_ret('acl.mdl.execute_async', ret)
        t_submit = time.time() - t0
        return {'h2d': t_h2d, 'submit': t_submit}

    def synchronize(self):
        if self.stream is None:
            return
        ret = acl.rt.synchronize_stream(self.stream)
        check_ret('acl.rt.synchronize_stream', ret)

    def read_outputs(self) -> Tuple[List[np.ndarray], float]:
        t0 = time.time()
        results: List[np.ndarray] = []
        if self.host_output_buffers:
            host_buffers = self.host_output_buffers
        else:
            host_buffers = []
            for i, ob in enumerate(self.output_buffers):
                host_ptr, ret = acl.rt.malloc_host(ob['size'])
                check_ret(f'acl.rt.malloc_host output[{i}]', ret)
                host_buffers.append({'ptr': host_ptr, 'size': ob['size'], 'owned': True})

        for i, (ob, hb) in enumerate(zip(self.output_buffers, host_buffers)):
            ret = acl.rt.memcpy(
                hb['ptr'], hb['size'], ob['buf'], ob['size'], ACL_MEMCPY_DEVICE_TO_HOST)
            check_ret(f'acl.rt.memcpy D2H[{i}]', ret)
            raw = acl.util.ptr_to_bytes(hb['ptr'], ob['size'])
            dtype = self.output_dtypes[i] if i < len(self.output_dtypes) else np.float32
            results.append(np.frombuffer(raw, dtype=dtype).copy())

        if not self.host_output_buffers:
            for hb in host_buffers:
                try:
                    acl.rt.free_host(hb['ptr'])
                except Exception:
                    pass

        return results, time.time() - t0

    def infer(
        self,
        inputs: Sequence[Optional[np.ndarray]],
        dynamic_dims: Optional[dict] = None,
        skip_copy_indices: Optional[Iterable[int]] = None,
    ) -> Tuple[List[np.ndarray], dict]:
        self._set_dynamic_dims(dynamic_dims)
        t_h2d = self.copy_inputs(inputs, skip_copy_indices=skip_copy_indices)
        t0 = time.time()
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        check_ret('acl.mdl.execute', ret)
        t_infer = time.time() - t0
        outputs, t_d2h = self.read_outputs()
        return outputs, {'h2d': t_h2d, 'infer': t_infer, 'd2h': t_d2h}

    def __del__(self):
        try:
            if hasattr(self, '_input_data_buffers') and getattr(self, 'input_dataset', None) is not None:
                for db in self._input_data_buffers:
                    try:
                        acl.mdl.remove_dataset_buffer(self.input_dataset, db)
                    except Exception:
                        pass
                    try:
                        acl.destroy_data_buffer(db)
                    except Exception:
                        pass
            if hasattr(self, '_output_data_buffers') and getattr(self, 'output_dataset', None) is not None:
                for db in self._output_data_buffers:
                    try:
                        acl.mdl.remove_dataset_buffer(self.output_dataset, db)
                    except Exception:
                        pass
                    try:
                        acl.destroy_data_buffer(db)
                    except Exception:
                        pass
            if getattr(self, 'input_dataset', None) is not None:
                try:
                    acl.mdl.destroy_dataset(self.input_dataset)
                except Exception:
                    pass
            if getattr(self, 'output_dataset', None) is not None:
                try:
                    acl.mdl.destroy_dataset(self.output_dataset)
                except Exception:
                    pass
            if hasattr(self, 'input_buffers'):
                for item in self.input_buffers:
                    if item.get('owned'):
                        try:
                            acl.rt.free(item['buf'])
                        except Exception:
                            pass
            if hasattr(self, 'output_buffers'):
                for item in self.output_buffers:
                    if item.get('owned'):
                        try:
                            acl.rt.free(item['buf'])
                        except Exception:
                            pass
            if hasattr(self, 'host_output_buffers'):
                for item in self.host_output_buffers:
                    if item.get('owned'):
                        try:
                            acl.rt.free_host(item['ptr'])
                        except Exception:
                            pass
            if hasattr(self, 'model_id'):
                try:
                    acl.mdl.unload(self.model_id)
                except Exception:
                    pass
            if hasattr(self, 'model_desc') and self.model_desc:
                try:
                    acl.mdl.destroy_desc(self.model_desc)
                except Exception:
                    pass
            if getattr(self, 'owns_stream', False) and getattr(self, 'stream', None) is not None:
                try:
                    acl.rt.destroy_stream(self.stream)
                except Exception:
                    pass
        except Exception:
            pass


class _StageSlot:
    """一个 host staging slot。"""

    def __init__(self, max_gear: int, num_cameras: int, image_size: Tuple[int, int]):
        H, W = image_size
        self.lock = threading.Lock()
        self.voxels = np.zeros((max_gear, 32, 5), dtype=np.float32)
        self.num_points = np.zeros((max_gear,), dtype=np.float32)
        self.coords = np.zeros((max_gear, 4), dtype=np.float32)
        self.imgs = np.zeros((1, num_cameras, 3, H, W), dtype=np.float32)
        self.depth = np.zeros((1, num_cameras, 1, H, W), dtype=np.float32)
        self.gear = 0
        self.actual_v = 0
        self.ready = False

    def stage(
        self,
        voxels: np.ndarray,
        num_points: np.ndarray,
        coords: np.ndarray,
        imgs: np.ndarray,
        depth: np.ndarray,
        gear: int,
    ):
        actual_v = int(voxels.shape[0])
        if actual_v > gear:
            raise ValueError(f'actual_v({actual_v}) > gear({gear})')

        with self.lock:
            np.copyto(self.voxels[:actual_v], voxels, casting='unsafe')
            np.copyto(self.num_points[:actual_v], num_points, casting='unsafe')
            np.copyto(self.coords[:actual_v], coords, casting='unsafe')
            if actual_v < gear:
                self.voxels[actual_v:gear].fill(0.0)
                self.num_points[actual_v:gear].fill(0.0)
                self.coords[actual_v:gear].fill(0.0)

            np.copyto(self.imgs, imgs, casting='unsafe')
            np.copyto(self.depth, depth, casting='unsafe')
            self.gear = int(gear)
            self.actual_v = actual_v
            self.ready = True


class BEVFusionParallelNPUNet:
    """
    三 OM 版本：LiDAR 分支 / Camera 分支并行，Fusion 串行收尾。

    新增接口：
      - stage_host_inputs(slot_id, ...): 将预处理结果写入 host 双缓冲 slot。
      - forward_staged(slot_id): 直接从 slot 启动一次推理。

    原 forward(...) 仍保留，便于兼容旧 evaluator。
    """

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

    def __init__(
        self,
        lidar_model_path: str,
        camera_model_path: str,
        fusion_model_path: str,
        gears: Optional[Sequence[int]] = None,
        D: int = 118,
        feature_size: Optional[Sequence[int]] = None,
        image_size: Optional[Sequence[int]] = None,
        num_cameras: int = 6,
        xbound: Tuple[float, float, float] = (-54.0, 54.0, 0.3),
        ybound: Tuple[float, float, float] = (-54.0, 54.0, 0.3),
        zbound: Tuple[float, float, float] = (-10.0, 10.0, 20.0),
        max_pts: int = 16,
        launch_order: str = 'lidar_first',
        stage_slots: int = 2,
    ):
        self.gears = list(gears or [6000, 8000, 10000])
        self.D = int(D)
        self.fH, self.fW = [int(x) for x in (feature_size or [32, 88])]
        self.H, self.W = [int(x) for x in (image_size or [256, 704])]
        self.N = int(num_cameras)
        self.max_pts = int(max_pts)
        self.launch_order = launch_order
        self.max_gear = int(max(self.gears))

        self._bx = np.array([
            xbound[0] + xbound[2] / 2,
            ybound[0] + ybound[2] / 2,
            zbound[0] + zbound[2] / 2,
        ], dtype=np.float32)
        self._dx = np.array([xbound[2], ybound[2], zbound[2]], dtype=np.float32)
        self._nx0 = int(round((xbound[1] - xbound[0]) / xbound[2]))
        self._nx1 = int(round((ybound[1] - ybound[0]) / ybound[2]))
        self._nx2 = int(round((zbound[1] - zbound[0]) / zbound[2]))
        self._out_cells = self._nx0 * self._nx1 * self._nx2

        self._lut_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._camera_static_uploaded = False

        self.lidar_stream, ret = acl.rt.create_stream()
        check_ret('acl.rt.create_stream(lidar)', ret)
        self.camera_stream, ret = acl.rt.create_stream()
        check_ret('acl.rt.create_stream(camera)', ret)
        self.fusion_stream, ret = acl.rt.create_stream()
        check_ret('acl.rt.create_stream(fusion)', ret)

        print(f'[BEVFusionParallelNPUNet-v2] Loading LiDAR OM:  {lidar_model_path}')
        self._lidar = _OmModel(
            lidar_model_path,
            output_dtypes=[np.float32],
            gears=self.gears,
            stream=self.lidar_stream,
            owns_stream=False,
            reuse_host_outputs=False,
        )
        print(f'[BEVFusionParallelNPUNet-v2] Loading Camera OM: {camera_model_path}')
        self._camera = _OmModel(
            camera_model_path,
            output_dtypes=[np.float32],
            stream=self.camera_stream,
            owns_stream=False,
            reuse_host_outputs=False,
        )

        external_inputs = {
            0: self._camera.get_output_buffer(0),
            1: self._lidar.get_output_buffer(0),
        }
        print(f'[BEVFusionParallelNPUNet-v2] Loading Fusion OM: {fusion_model_path}')
        self._fusion = _OmModel(
            fusion_model_path,
            output_dtypes=self.OUTPUT_DTYPES,
            stream=self.fusion_stream,
            external_input_buffers=external_inputs,
            owns_stream=False,
            reuse_host_outputs=True,
        )

        self._stage_slots = [
            _StageSlot(self.max_gear, self.N, (self.H, self.W))
            for _ in range(int(stage_slots))
        ]

        self.timing = {
            'lut_build': [],
            'host_stage': [],
            'lidar_h2d': [],
            'camera_h2d': [],
            'branch_stage': [],
            'fusion_h2d': [],
            'fusion_infer': [],
            'fusion_d2h': [],
            'total': [],
        }

        print('[BEVFusionParallelNPUNet-v2] Initialized.')
        print(f'  BEV grid: nx0={self._nx0} nx1={self._nx1} nx2={self._nx2}')
        print(f'  out_cells={self._out_cells}  max_pts={self.max_pts}  launch_order={self.launch_order}')
        print(f'  host_stage_slots={len(self._stage_slots)}  max_gear={self.max_gear}')

    def _build_lut_numpy(self, geom_feats_np: np.ndarray):
        B = 1
        N = self.N
        D = self.D
        fH = self.fH
        fW = self.fW
        nx0, nx1, nx2 = self._nx0, self._nx1, self._nx2
        max_pts = self.max_pts
        ppb = N * D * fH * fW
        out_cells = self._out_cells

        coords = geom_feats_np.reshape(-1, 3).astype(np.float32)
        idx = ((coords - (self._bx - self._dx * 0.5)) / self._dx).astype(np.int64)
        x_id, y_id, z_id = idx[:, 0], idx[:, 1], idx[:, 2]

        valid = (
            (x_id >= 0) & (x_id < nx0) &
            (y_id >= 0) & (y_id < nx1) &
            (z_id >= 0) & (z_id < nx2)
        )

        b_id = np.repeat(np.arange(B, dtype=np.int64), ppb)
        x_c = np.clip(x_id, 0, nx0 - 1)
        y_c = np.clip(y_id, 0, nx1 - 1)
        z_c = np.clip(z_id, 0, nx2 - 1)
        lin = b_id * (nx2 * nx0 * nx1) + z_c * (nx0 * nx1) + x_c * nx1 + y_c

        pool_lookup = np.zeros((out_cells, max_pts), dtype=np.int64)
        pool_mask = np.zeros((out_cells, max_pts), dtype=np.uint8)

        valid_pt_idx = np.where(valid)[0]
        valid_vox = lin[valid_pt_idx]

        order = np.argsort(valid_vox, kind='stable')
        sorted_pt = valid_pt_idx[order]
        sorted_vox = valid_vox[order]

        is_new = np.concatenate([[True], sorted_vox[1:] != sorted_vox[:-1]])
        seg_start_pos = np.where(is_new, np.arange(len(sorted_pt), dtype=np.int64), 0)
        seg_start_pos = np.maximum.accumulate(seg_start_pos)
        slot_arr = np.arange(len(sorted_pt), dtype=np.int64) - seg_start_pos

        keep = slot_arr < max_pts
        final_pt = sorted_pt[keep]
        final_vox = sorted_vox[keep]
        final_slot = slot_arr[keep]

        pool_lookup[final_vox, final_slot] = final_pt
        pool_mask[final_vox, final_slot] = 1
        return pool_lookup, pool_mask

    def precompute_lut(self, geom_feats_np: np.ndarray):
        t0 = time.time()
        pool_lookup, pool_mask = self._build_lut_numpy(geom_feats_np)
        self._lut_cache = (pool_lookup, pool_mask)
        self._camera.copy_input(2, np.ascontiguousarray(pool_lookup))
        self._camera.copy_input(3, np.ascontiguousarray(pool_mask))
        self._camera_static_uploaded = True
        elapsed = time.time() - t0
        self.timing['lut_build'].append(elapsed)
        print(
            '[BEVFusionParallelNPUNet-v2] LUT precomputed '
            f'in {elapsed*1000:.2f} ms  '
            f'(lookup={pool_lookup.dtype}, mask={pool_mask.dtype})'
        )

    def _ensure_lut_ready(self, geom_feats: Optional[np.ndarray]):
        if self._lut_cache is None:
            if geom_feats is None:
                raise ValueError(
                    'LUT 尚未构建。首次调用 forward()/forward_staged() 时必须传入 geom_feats，'
                    '或提前调用 precompute_lut(geom_feats)。')
            self.precompute_lut(geom_feats.astype(np.float32, copy=False))
        elif not self._camera_static_uploaded:
            pool_lookup, pool_mask = self._lut_cache
            self._camera.copy_input(2, np.ascontiguousarray(pool_lookup))
            self._camera.copy_input(3, np.ascontiguousarray(pool_mask))
            self._camera_static_uploaded = True

    def _select_gear(self, actual_v: int) -> int:
        for g in sorted(self.gears):
            if g >= actual_v:
                return g
        return max(self.gears)

    def stage_host_inputs(
        self,
        slot_id: int,
        voxels: np.ndarray,
        num_points: np.ndarray,
        coords: np.ndarray,
        imgs: np.ndarray,
        depth: np.ndarray,
    ) -> dict:
        """仅做 host 侧 staging，不触碰 ACL，可安全用于 CPU 预取线程。"""
        t0 = time.time()
        slot = self._stage_slots[int(slot_id) % len(self._stage_slots)]

        voxels = np.ascontiguousarray(voxels, dtype=np.float32)
        num_points = np.ascontiguousarray(num_points, dtype=np.float32)
        coords = np.ascontiguousarray(coords, dtype=np.float32)
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)
        depth = np.ascontiguousarray(depth, dtype=np.float32)

        actual_v = int(voxels.shape[0])
        gear = self._select_gear(actual_v)
        if gear > self.max_gear:
            raise ValueError(f'selected gear({gear}) > max_gear({self.max_gear})')

        slot.stage(voxels, num_points, coords, imgs, depth, gear)
        elapsed = time.time() - t0
        self.timing['host_stage'].append(elapsed)
        return {
            'slot_id': int(slot_id) % len(self._stage_slots),
            'gear': gear,
            'actual_v': actual_v,
            'stage': elapsed,
        }

    def _launch_lidar(self, voxels_p: np.ndarray, num_points_p: np.ndarray, coords_p: np.ndarray, gear: int):
        dynamic_dims = {
            'name': '',
            'dimCount': 6,
            'dims': [gear, 32, 5, gear, gear, 4],
        }
        return self._lidar.execute_async(
            [voxels_p, num_points_p, coords_p],
            dynamic_dims=dynamic_dims,
        )

    def _launch_camera(self, imgs: np.ndarray, depth: np.ndarray):
        return self._camera.execute_async(
            [imgs, depth, None, None],
            skip_copy_indices={2, 3},
        )

    def forward_staged(
        self,
        slot_id: int,
        geom_feats: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        t_total_start = time.time()
        self._ensure_lut_ready(geom_feats)

        slot = self._stage_slots[int(slot_id) % len(self._stage_slots)]
        with slot.lock:
            if not slot.ready:
                raise RuntimeError(f'stage slot {slot_id} is not ready')
            gear = int(slot.gear)
            if gear <= 0:
                raise RuntimeError(f'invalid gear in stage slot {slot_id}: {gear}')

            t_branch_start = time.time()
            if self.launch_order == 'camera_first':
                camera_detail = self._launch_camera(slot.imgs, slot.depth)
                lidar_detail = self._launch_lidar(
                    slot.voxels[:gear], slot.num_points[:gear], slot.coords[:gear], gear)
            else:
                lidar_detail = self._launch_lidar(
                    slot.voxels[:gear], slot.num_points[:gear], slot.coords[:gear], gear)
                camera_detail = self._launch_camera(slot.imgs, slot.depth)

            self._lidar.synchronize()
            self._camera.synchronize()
            t_branch = time.time() - t_branch_start

            fusion_outputs, fusion_detail = self._fusion.infer(
                [None, None],
                skip_copy_indices={0, 1},
            )

        t_total = time.time() - t_total_start
        self.timing['lidar_h2d'].append(lidar_detail['h2d'])
        self.timing['camera_h2d'].append(camera_detail['h2d'])
        self.timing['branch_stage'].append(t_branch)
        self.timing['fusion_h2d'].append(fusion_detail['h2d'])
        self.timing['fusion_infer'].append(fusion_detail['infer'])
        self.timing['fusion_d2h'].append(fusion_detail['d2h'])
        self.timing['total'].append(t_total)

        print(
            '[Parallel NPU v2] '
            f'Slot={int(slot_id) % len(self._stage_slots)}  '
            f'LiDAR_H2D={lidar_detail["h2d"]*1000:.1f}ms  '
            f'Camera_H2D={camera_detail["h2d"]*1000:.1f}ms  '
            f'Branches={t_branch*1000:.1f}ms  '
            f'Fusion={fusion_detail["infer"]*1000:.1f}ms  '
            f'D2H={fusion_detail["d2h"]*1000:.1f}ms  '
            f'Total={t_total*1000:.1f}ms'
        )
        return fusion_outputs

    def forward(
        self,
        voxels: np.ndarray,
        num_points: np.ndarray,
        coords: np.ndarray,
        imgs: np.ndarray,
        depth: np.ndarray,
        geom_feats: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        self.stage_host_inputs(0, voxels, num_points, coords, imgs, depth)
        return self.forward_staged(0, geom_feats=geom_feats)

    def print_timing_summary(self):
        print('\n' + '=' * 68)
        print('BEVFusionParallelNPUNet-v2 TIMING SUMMARY')
        print('=' * 68)
        fmt = '{:<16} {:>10.3f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}'
        print(f'{"Stage":<16} {"Total(s)":>10} {"Mean(ms)":>10} {"Std(ms)":>10} {"Min(ms)":>10} {"Max(ms)":>10}')
        print('-' * 68)
        for name, times in self.timing.items():
            if not times:
                continue
            arr = np.array(times, dtype=np.float64)
            print(fmt.format(
                name,
                float(arr.sum()),
                float(arr.mean() * 1000.0),
                float(arr.std() * 1000.0),
                float(arr.min() * 1000.0),
                float(arr.max() * 1000.0),
            ))
        print('=' * 68)


__all__ = ['init_acl', 'BEVFusionParallelNPUNet']
