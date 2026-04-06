## ERR-002: voxel_encoder ONNX/OM导出失败

### Date
2026-03-02

### Environment
- OS: Linux
- Component: voxel_encoder ONNX/OM导出
- Tool: ONNX export, ATC converter

### Symptom
在导出voxel_encoder模型为ONNX/OM格式时失败,错误信息:
```
Concat算子输入形状不匹配: shapes:[[10000, 32, 32], [10000, 1024, 32]] axis:2
```

该错误导致无法将模型部署到Ascend 310P NPU上。

### Debug Process
1. 分析错误信息,发现Concat算子在axis=2维度上形状不匹配
2. 检查模型结构,确认Concat算子输入来源
3. 搜索相关问题和解决方案
4. 尝试不同的导出参数配置
5. 测试不同opset版本的影响
6. 调整体素数量参数

### Root Cause
1. **Opset版本过高**: opset 18版本在某些算子转换时不够稳定,可能导致形状推断错误
2. **体素数量过大**: 10000个体素占用过多GPU内存,影响导出过程
3. **算子兼容性**: 部分自定义或非标准算子在ONNX转换时存在兼容性问题

### Fix
1. **降低opset版本**: 从18降到11(更稳定)
   ```python
   torch.onnx.export(model, args, output_path, opset_version=11)
   ```

2. **减少体素数量**: 从10000降到8000(节省GPU内存)
   ```python
   max_voxels = 8000  # 原值为10000
   ```

3. **添加导出参数**: 使用标准ONNX算子
   ```python
   torch.onnx.export(
       model,
       args,
       output_path,
       opset_version=11,
       operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
       do_constant_folding=True
   )
   ```

### Verification
执行ONNX导出命令后:
- 导出成功,生成.onnx文件
- 使用onnx.checker验证模型有效性
- 使用ATC工具成功转换为.om格式
- 在Ascend 310P上成功加载和推理

### Lessons & Notes
- opset版本不是越高越好,稳定性和兼容性更重要
- 对于边缘部署,需要平衡模型大小和内存占用
- ONNX导出时建议使用标准算子,避免自定义算子
- 遇到形状不匹配错误时,优先检查opset版本和算子兼容性
- 减少不必要的参数(如体素数量)可以降低内存占用,提高导出成功率
