"""
BEVFusion OM Model Inference Base Class
This module provides the base class for running BEVFusion models on Ascend NPU.

Requirements:
    - acl (Ascend Computing Language)
    - numpy

Usage:
    from bevfusion_net import Net, init_acl

    ctx = init_acl(0)
    net = Net("./bevfusion_full.om")
    outputs = net.forward([voxels, num_points, coords])
"""

import numpy as np
import acl
import sys

# Error code constant
ACL_SUCCESS = 0


def check_ret(message, ret):
    """Check return value, print error and exit if failed."""
    if ret != ACL_SUCCESS:
        print(f"[Error] {message} failed, return code: {ret}")
        sys.exit(1)


class Net:
    """Ascend OM Model inference class for BEVFusion with dynamic batch support."""

    def __init__(self, model_path, gears=[4000, 5000, 6000]):
        """
        Initialize the OM model.

        Args:
            model_path: Path to the .om model file
            gears: List of supported dynamic batch sizes (voxel counts)
        """
        self.gears = gears
        # 1. Load model
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
        print(f"Input count: {input_count}")

        for i in range(input_count):
            size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            print(f"Input {i} size: {size}")
            buf, ret = acl.rt.malloc(size, 0)
            check_ret(f"acl.rt.malloc input {i}", ret)
            db = acl.create_data_buffer(buf, size)
            _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, db)
            check_ret(f"acl.mdl.add_dataset_buffer input {i}", ret)
            self.input_buffers.append({"buffer": buf, "size": size})

        # Create output dataset
        self.output_dataset = acl.mdl.create_dataset()
        output_count = acl.mdl.get_num_outputs(self.model_desc)
        print(f"Output count: {output_count}")

        for i in range(output_count):
            size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            print(f"Output {i} size: {size}")
            buf, ret = acl.rt.malloc(size, 0)
            check_ret(f"acl.rt.malloc output {i}", ret)
            db = acl.create_data_buffer(buf, size)
            _, ret = acl.mdl.add_dataset_buffer(self.output_dataset, db)
            check_ret(f"acl.mdl.add_dataset_buffer output {i}", ret)
            self.output_buffers.append({"buffer": buf, "size": size})

    def forward(self, inputs, actual_voxel_num):
        """
        Run inference on the model.

        Args:
            inputs: List of numpy arrays [voxels, num_points, coords]
            actual_voxel_num: Actual number of voxels in the input

        Returns:
            List of numpy arrays (model outputs)
        """
        # 1. Auto-match gear (dynamic batch size)
        target_gear = next((g for g in self.gears if g >= actual_voxel_num), self.gears[-1])
        print(f"Auto-matched gear: {target_gear}")

        # 2. padding
        def pad_to_gear(data, target_gear):
            pad_size = target_gear - data.shape[0]
            if pad_size <= 0:
                return data
            pad_shape = (pad_size,) + data.shape[1:]
            pad = np.zeros(pad_shape, dtype=data.dtype)
            return np.concatenate([data, pad], axis=0)
        
        voxels = pad_to_gear(inputs[0], target_gear)
        num_points = pad_to_gear(inputs[1], target_gear)
        coords = pad_to_gear(inputs[2], target_gear)
        # 打印padding后的形状以验证
        print(f"After padding to gear {target_gear} - voxels: {voxels.shape}, num_points: {num_points.shape}, coords: {coords.shape}")
        inputs = [voxels, num_points, coords]
        # 2. Set dynamic dimensions
        index, ret = acl.mdl.get_input_index_by_name(self.model_desc, "ascend_mbatch_shape_data")
        check_ret("get_dynamic_index", ret)

        # Note: dims must match your ATC conversion settings exactly
        # BEVFusion输入: voxels[num_voxels,32,5], num_points[num_voxels], coords[num_voxels,4]
        current_dims = {'name': '', 'dimCount': 6, 'dims': [target_gear, 32, 5, target_gear, target_gear, 4]}
        ret = acl.mdl.set_input_dynamic_dims(self.model_id, self.input_dataset, index, current_dims)
        check_ret("set_dynamic_dims", ret)

        # 3. H2D data copy
        for i, data in enumerate(inputs):
            bytes_data = data.tobytes()
            print(f"Input {i}: {data.shape}")
            print(f"  Input {i} size: {len(bytes_data)}")
            ret = acl.rt.memcpy(self.input_buffers[i]["buffer"],
                                self.input_buffers[i]["size"],
                                acl.util.bytes_to_ptr(bytes_data),
                                len(bytes_data), 1)
            check_ret(f"memcpy H2D {i}", ret)

        # 4. Execute inference
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        check_ret("execute", ret)

        # 5. D2H result retrieval
        results = []
        for i in range(len(self.output_buffers)):
            size = self.output_buffers[i]["size"]
            host_ptr, ret = acl.rt.malloc_host(size)
            check_ret(f"malloc_host", ret)
            ret = acl.rt.memcpy(host_ptr, size, self.output_buffers[i]["buffer"], size, 2)
            check_ret(f"memcpy D2H", ret)
            data_bytes = acl.util.ptr_to_bytes(host_ptr, size)

            # 根据输出索引确定数据类型
            # 输出: dense_heatmap, top_cls, query_heatmap_score, heatmap_q, center, height, dim, rot, vel, top_idx, top, top_score, keep
            if i == 1  or i == 9 or i == 10 or i == 12:  # top_cls, heatmap_q, top_idx, top, keep 是 int32
                results.append(np.frombuffer(data_bytes, dtype=np.int32).copy())
            else:  # 其他都是 float32
                results.append(np.frombuffer(data_bytes, dtype=np.float32).copy())

            acl.rt.free_host(host_ptr)

        return results

    def __del__(self):
        """Release resources in destructor."""
        if hasattr(self, 'model_id'):
            acl.mdl.unload(self.model_id)
        if hasattr(self, 'model_desc') and self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)


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
