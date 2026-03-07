import onnx
# | 编号 | 类型      |
# | -- | ------- |
# | 1  | float32 |
# | 6  | int32   |
# | 7  | int64   |
# | 11 | float64 |
model = onnx.load("models/onnx_final/bevfusion_final_dynamic.onnx")
for input in model.graph.input:
    print(input.name, input.type.tensor_type.elem_type)

# results:
# voxels 1
# num_points 7
# coords 1

