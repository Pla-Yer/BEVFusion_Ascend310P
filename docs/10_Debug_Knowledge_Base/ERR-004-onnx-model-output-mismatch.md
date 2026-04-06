## ERR-004: ONNX模型输出与PyTorch模型不匹配

### Date
2026-03-05

### Environment
- OS: Linux
- Component: ONNX模型导出与推理
- Tool: ONNX Runtime, PyTorch
- Model: BEVFusion

### Symptom
搭建bevfusion_evaluator与bevfusion_net后,实现OM模型推理评估全流程,但结果显示mAP为0。通过对比ONNX与PyTorch模型输出,发现9个输出都存在显著差异:

```
输出名称                   shape                    max|Δ|    mean|Δ|       RMSE  allclose
dense_heatmap          (1, 10, 180, 180)    1.8892e+00 3.1280e-02 7.0119e-02         ❌
top_cls                (1, 200)             9.0000e+00 6.5000e-01 2.0075e+00         ❌
query_heatmap_score    (1, 10, 200)         3.6742e-02 1.4170e-03 5.0199e-03         ❌
heatmap_q              (1, 10, 200)         4.9092e+00 1.6529e-01 4.6310e-01         ❌
center                 (1, 2, 200)          1.7913e+02 1.5294e+01 5.0362e+01         ❌
height                 (1, 1, 200)          1.6983e+00 1.4287e-01 3.6247e-01         ❌
dim                    (1, 3, 200)          2.2428e+00 8.5765e-02 2.6302e-01         ❌
rot                    (1, 2, 200)          1.2246e+00 7.1994e-02 2.0888e-01         ❌
vel                    (1, 2, 200)          4.4105e-01 1.2608e-02 4.8122e-02         ❌
```

### Debug Process
1. 验证体素化部分结果正确,排除体素化问题
2. 创建测试脚本`debug/verify_onnx_vs_pth.py`,统一使用CPU体素化算子结果作为输入
3. 对比ONNX与PyTorch模型输出,发现所有输出都不匹配
4. 初步怀疑NMS导致大量tie,使heatmap被置为0
5. 增加小偏置使topk输入有严格全序,但问题未解决
6. 发现模型输出的heatmap在decoder之前就存在误差
7. 创建`debug/diagnose_layer.py`,对模型每个层单独导出与验证
8. 定位到MiddleEncoder和Backbone出现问题

### Root Cause
通过逐层验证发现:
```
Stage1: VoxelEncoder               : ✅ PASS
Stage2: MiddleEncoder              : ❌ FAIL
Stage3: Backbone                   : ❌ FAIL
Stage4: Neck                       : ✅ PASS
Stage5: SharedConv                 : ✅ PASS
Stage6: HeatmapHead                : ✅ PASS
```

问题定位到MiddleEncoder,具体原因见[ERR-005](./ERR-005-pointpillars-scatter-onnx-incompatibility.md)。

### Fix
见[ERR-005](./ERR-005-pointpillars-scatter-onnx-incompatibility.md)的修复方案。

### Verification
修复MiddleEncoder后,重新验证各层输出,Stage2变为✅ PASS。

### Lessons & Notes
- **逐层验证是关键**: 当模型输出不匹配时,应逐层验证定位问题模块
- **ONNX trace问题**: for循环和in-place操作会导致ONNX trace错误
- **调试策略**:
  - 统一输入源,排除输入差异
  - 逐层验证,定位问题模块
  - 使用专门的诊断脚本
- **工具使用**: 创建专门的对比和诊断脚本,提高调试效率
