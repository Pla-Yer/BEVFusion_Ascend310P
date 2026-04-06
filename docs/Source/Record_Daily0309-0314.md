3.09-3.13：
今天来进行优化，首先将每个模块进行导出测试，然后进行性能测试和精度测试。

| Component       | Latency (ms) | % of Total | Status |
| --------------- | ------------ | ---------- | ------ |
| stageE (head)   | 40.752490    | **16.56%** | ✅      |
| stageD_neck     | 2.630        | **1.07%**  | ✅      |
| stageC_backbone | 5.755        | **2.34%**  | ✅      |
| stageB_scatter  | 18.04        | **7.33%**  | ✅      |
| stageA_encoder  | 178.9        | **72.69%** | ⚠️     |
| **NPU Total**   | **246.077**  | **100%**   | ✅      |

可以看见模型在 stageA_encoder 模块存在性能问题，需要优化。
stageA的工作内容为：增加两个特征：cluster center，pillar center，然后进行特征聚合，再输入pfn进行编码；
通过msprof工具对stageA进行分析，运行10次，查看结果如下：

| opType          | accCore | count | totalTime  | avgTime | maxTime  | minTime  |
| --------------- | ------- | ----- | ---------- | ------- | -------- | -------- |
| ScatterNdUpdate | AI_CORE | 33    | 1665701.46 | 50475.8 | 52459.08 | 48658.56 |
| Transpose       | AI_CORE | 66    | 130345.8   | 1974.94 | 3864.48  | 1141.18  |
| Cast            | AI_CORE | 99    | 39629.31   | 400.3   | 1703.56  | 2.08     |
| BNInferenceD    | AI_CORE | 33    | 34057.71   | 1032.05 | 2264.51  | 280.22   |
| Relu            | AI_CORE | 33    | 22419.94   | 679.39  | 1204.33  | 275.61   |
| BatchMatMulV2   | AI_CORE | 33    | 22232.85   | 673.72  | 1260.03  | 33.57    |
| ConcatD         | AI_CORE | 33    | 20092.79   | 608.87  | 1176.54  | 56.95    |
| GatherV2        | AI_CORE | 66    | 9914.22    | 150.22  | 295.79   | 6.07     |
| ReduceMaxD      | AI_CORE | 22    | 5587.02    | 253.96  | 336.91   | 168.39   |
| TileD           | AI_CORE | 22    | 4743.22    | 215.6   | 367.12   | 65.76    |
| TransData       | AI_CORE | 22    | 2195.08    | 99.78   | 177.11   | 23       |
| StridedSliceD   | AI_CORE | 22    | 1090.37    | 49.56   | 51.33    | 48.02    |
| ReduceSumD      | AI_CORE | 11    | 996.85     | 90.62   | 90.89    | 90.5     |
| Mul             | AI_CORE | 11    | 730.44     | 66.4    | 68.08    | 64.74    |
| TensorMove      | AI_CORE | 33    | 317.56     | 9.62    | 18.26    | 5.31     |

发现ScatterNdUpdate算子的运行时间过长，占比80%以上，通过分析代码发现,在添加特征pillar center时，有如下计算：

```python
# 这几行是罪魁祸首
f_center[:, :, 0] = features[:, :, 0] - (...)  # → ScatterND
f_center[:, :, 1] = features[:, :, 1] - (...)  # → ScatterND  
f_center[:, :, 2] = features[:, :, 2] - (...)  # → ScatterND
```

这里的 “=”会被映射为ScatterNd算子，会对6000*32的位置进行3次scatter；

优化策略：将**索引赋值操作**操作变为**拼接**操作，即将其替换为：

```python
f_center_x = features[:, :, 0] - (...
f_center_y = features[:, :, 1] - (...
f_center_z = features[:, :, 2] - (...)
f_center = torch.stack([f_center_x, f_center_y, f_center_z], dim=-1)
```

将ScatterNd算子替换为了`Sub` + `Stack`（`Unsqueeze/Concat`）

最后的分析如下：

|                         |                |       |           |         |         |         |
| ----------------------- | -------------- | ----- | --------- | ------- | ------- | ------- |
| opType                  | accCore        | count | totalTime | avgTime | maxTime | minTime |
| Transpose               | AI_CORE        | 66    | 130412.7  | 1975.95 | 3883.96 | 1140.21 |
| Cast                    | AI_CORE        | 99    | 39541.39  | 399.41  | 1702.81 | 2       |
| BNInferenceD            | AI_CORE        | 33    | 34090.27  | 1033.04 | 2273.16 | 279.49  |
| Relu                    | AI_CORE        | 33    | 22407.33  | 679.01  | 1209.22 | 275.81  |
| BatchMatMulV2           | AI_CORE        | 33    | 22216.47  | 673.23  | 1261.26 | 33.73   |
| ConcatD                 | AI_CORE        | 44    | 20147.3   | 457.89  | 1175.19 | 8.02    |
| GatherV2                | AI_CORE        | 66    | 9929.19   | 150.44  | 294.33  | 6.2     |
| ReduceMaxD              | AI_CORE        | 22    | 5578.81   | 253.58  | 337.12  | 170.68  |
| TileD                   | AI_CORE        | 22    | 4796.82   | 218.04  | 371.81  | 71.04   |
| TransData               | AI_CORE        | 22    | 2188.36   | 99.47   | 177.5   | 22.89   |
| StridedSliceD           | AI_CORE        | 22    | 1068.86   | 48.58   | 49.25   | 48.23   |
| ReduceSumD              | AI_CORE        | 11    | 981.79    | 89.25   | 89.38   | 89.22   |
| Mul                     | AI_CORE        | 11    | 737.62    | 67.06   | 68.93   | 65.44   |
| AutomaticBufferFusionOp | AI_CORE        | 44    | 355.32    | 8.08    | 19.82   | 3.93    |
| AutomaticBufferFusionOp | AI_VECTOR_CORE | 11    | 294.75    | 26.8    | 29.01   | 22.68   |

耗时从180ms降低到了41ms。此时的模型各阶段推理耗时：

| Component       | Latency (ms) | % of Total | Status |
| --------------- | ------------ | ---------- | ------ |
| stageE (head)   | 40.752490    | **37.67%** | ✅      |
| stageD_neck     | 2.630        | **2.43%**  | ✅      |
| stageC_backbone | 5.755        | **5.32%**  | ✅      |
| stageB_scatter  | 18.04        | **16.68%** | ✅      |
| stageA_encoder  | 41           | **37.90%** | ✅      |
| **NPU Total**   | **108.177**  | **100%**   | ✅      |

同时，精度没有下降。

考虑对scatter进行优化，尝试了MatMul 替代 Scatter；index_add → ScatterAdd，两种方法都没有实现更优。

---

为了将精度进一步与torch模型对齐，将体素化个数从上线6000转到上限10000：

- 模型推理耗时：108->132ms;

- 全流程pipline：0.21->0.25s;

- 精度map：0.189->0.219

同时，对模型进行量化，在atc时使用命令：[--precision_mode_v2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/devaids/atctool/atlasatcparam_16_0069.html)=force_fp16将模型精度从force_32->force_16:

- 模型推理耗时：132->102ms;

- 全流程pipline：0.25->0.22s;

- 精度map：0.219->0.218；

使用[aoe]([AOE简介-AOE工具（Ascend EP）-AOE调优工具-开发工具-CANN社区版8.3.RC1.alpha003开发文档-昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/devaids/aoe/aoeep_16_001.html))工具，发现难以提高性能：
![](/home/ttt/.config/marktext/images/2026-03-11-16-07-06-image.png)

阶段总结，但前bevfusion的lidar分支全流程计时统计如下：

Sweep聚合                   2.739     0.0338     0.0177     0.0012     0.1179
体素化                       6.905     0.0853     0.0152     0.0304     0.1480
模型推理                      8.297     0.1024     0.0036     0.0837     0.1137
结果解码                      0.038     0.0005     0.0002     0.0003     0.0015
NMS处理                     0.085     0.0001     0.0003     0.0000     0.0033
坐标转换                      0.419     0.0052     0.0027     0.0028     0.0272
单样本总计                    18.559     0.2291     0.0293     0.1209     0.3807

mAP: 0.2186

---

接下来将图像分支也融合，优先运行完全部流程：

pipline如下：

point->sweep->voxelize->camera->(lidar brach->camere brach->depth gen->geo gen->bevpoll->fusion head-)>decoder-> nms->transform

结果如下：
Task                      Total(s)   Mean(ms)   Std(ms)    Min(ms)    Max(ms)   

sweep_loading             3.76       46.42      20.49      1.22       107.76    
voxelization              5.74       70.85      22.68      7.55       100.15    
image_loading             21.72      268.14     30.52      214.92     359.19    
fusion_inference          115.61     1427.23    173.96     1203.61    1905.11   
decode                    0.03       0.40       0.05       0.30       0.62      
nms                       0.25       3.09       0.84       1.19       5.56      
transform                 0.41       5.01       1.02       2.73       7.69      
total_per_sample          147.53     1821.42    187.15     1533.01    2358.69   
============================================================

Task                      Total(s)   Mean(ms)   Std(ms)    Min(ms)    Max(ms)    Count

lidar_branch              15.921     196.55     14.21      158.69     210.03     81   
camera_branch             3.696      45.63      1.45       43.98      54.70      81   
depth_generation          1.345      16.61      8.90       3.99       66.92      81   
geometry_calculation      5.492      67.80      32.48      24.34      160.88     81   
bev_pool                  10.933     134.97     17.75      112.38     206.82     81   
fusion_head               17.020     210.12     6.22       183.52     225.70     81   
total_inference           102.180    1261.49    47.89      1187.47    1530.95    81   
cpu_parts                 0.772      771.54     0.00       771.54     771.54     1    
============================================================

map：    0.217.

可以看见模型耗时较长，通过逐阶段分析发现，大量时间花在拷入/出数据上，于是打算整合为一个om模型进行，预期pipline为：
cpu（depth gen+geo gen）->om(lidar->camera->bevpool->fusion head）->cpu（decoder），模型推理耗时达到了1500ms，根据分析，将bevpool过程单独设置为一个om模型，推理耗时也达到了1300ms，这是因为，bevpool的主要计算是在将特征撒在bev网格中，这种计算是npu不擅长的。

下一周工作：学习AscendC算子设计，将bevpool设计为一个npu算子。