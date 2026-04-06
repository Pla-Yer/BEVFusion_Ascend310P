今天学习AscendC算子开发：实现了add与sigmoid算子。
当前为了优化，目标是将bevpool算子化插入到om模型中，那么现在有几种方法：

1. 将全过程打包为一个om模型，将bevpool算子插入到om模型中，然后调用om模型进行推理。
2. 将bevpool算子单独打包为一个om模型，然后通过多模型串行进行推理。
3. 将bevpool算子打包入相机处理分支中，然后多模型串行进行推理。
   现在的问题是：
4. 不知道怎么将bevpool算子打包进入om模型中，只知道怎么将一个算子打包成一个om模型。
5. 不知道怎么将om模型进行多模型串行推理。

分析：
将算子插入到om模型中是比较完整的做法，但是我并不知道怎么将算子插入到om模型中。首先应该在onnx中注册算子，导出为onnx，然后在atc时，是否需要添加参数?还是说AscendC算子编译运行后，ATC会自动将AscendC算子插入到om模型中呢?
我现在知道怎么将一个算子打包成一个om模型，但是不知道怎么将算子掺入到om模型中，我也没有尝试过om模型多模型串行推理。

最后决定还是优先按照当前路径进行优化，通过算法层面实现优化。

通过对模型进行profiling，发现模型大量时间花在ScatterElements上，通过分析，判断是BEVPool环节的随机写过程带来的巨大延时具体参照：[深入理解硬件计算性能——以BEVPOOL为例.md](/home/ttt/PY080313/BEVFusion_Ascend310P/深入理解硬件计算性能——以BEVPOOL为例.md)。通过将scatter->gather+reduceSum，大大减少写冲突与串行操作。

效果如下：

原过程：

========================================================================
Task                         Total(s)   Mean(ms)    Std(ms)    Min(ms)    Max(ms)

------------------------------------------------------------------------

sweep_loading                    3.44      42.41      18.59       1.29     106.33
voxelization                     5.24      64.68      25.79       2.69     104.07
image_loading                   19.90     245.62      45.54      85.93     358.42
precompute_depth                 9.76     120.54      75.34       7.34     445.56
precompute_geometry              4.11      50.78      29.07      25.26     178.06
npu_inference                  145.86    1800.77     481.92    1468.50    2900.82
decode                           0.05       0.66       0.10       0.44       0.92
nms                              0.22       2.73       0.78       1.01       4.07
transform                        0.20       2.51       0.52       1.23       3.40
total_per_sample               189.05    2334.00     485.54    1896.13    3757.58
========================================================================

现过程：

=======================================================================
Task Total(s) Mean(ms) Std(ms) Min(ms) Max(ms)

------------------------------------------------------------------------

sweep_loading 5.20 64.19 21.41 3.83 115.56
voxelization 5.89 72.74 22.54 7.34 106.32
image_loading 24.02 296.56 29.38 229.96 380.26
precompute_depth 13.50 164.65 111.14 91.95 554.60
precompute_geometry 0.51 6.23 43.56 0.00 374.50
npu_inference 14.76 182.22 4.27 158.52 191.85
decode 0.05 0.67 0.16 0.43 1.56
nms 0.22 2.77 0.85 0.91 5.13
transform 0.21 2.54 0.70 1.36 4.73
total_per_sample 64.16 792.10 142.24 633.02 1387.86
========================================================================

模型推理从1800ms降低到182ms，但整个流程平均耗时792ms，这仍然太耗时，通过上面的打印可以发现，耗时主要集中在：image_loading,precompute_depth,分析实现中的瓶颈如下：

| 问题                        | 根因                                   |
| ------------------------- | ------------------------------------ |
| image_loading 296ms       | 6 张图串行 disk I/O                      |
| precompute_depth 164ms    | torch + Python per-camera 循环，每帧重算逆矩阵 |
| sweep/voxelization ~135ms | 可以并行预读下一帧                            |
| pose/calib nusc.get()     | 每帧都 lookup，但 sensor calib 完全固定       |

三步优化：①并行图像加载；②纯 numpy 向量化深度投影；③缓存不变的传感器标定矩阵。

1. 图片串行读取：

```
# 旧：串行，总时间 = sum(单张时间) ≈ 6 × 50ms
for cam_key in self.cam_keys:
    img = load_and_preprocess_image(img_path, ...)

# 新：并行，总时间 = max(单张时间) ≈ 50ms
with ThreadPoolExecutor(max_workers=N) as exe:
    imgs_list = list(exe.map(_load, img_paths))
```

2. `torch.inverse` 每帧算单位矩阵的逆（结果仍为单位矩阵，纯浪费）；② per-camera Python 循环做 scatter。

```
# 旧：torch + 6 次 Python 循环
for c in range(N):           # Python loop，每次 ~20ms
    valid_pix = pix[c, on_img[c]].long()
    depth[b, c, 0, valid_pix[:,0], valid_pix[:,1]] = valid_dist

# 新：纯 numpy，一次性完成所有相机
pts_cam = lidar2image_np[:, :3, :] @ pts_hom.T   # [N,3,M] 单次 matmul
cam_idx, pt_idx = np.where(valid)                  # 一次 where
depth_out[linear] = d[order]                       # 一次 fancy-index scatter
```

最后结果：

Task                         Total(s)   Mean(ms)    Std(ms)    Min(ms)    Max(ms)

------------------------------------------------------------------------

sweep_loading                    2.93      36.14      17.68       1.32     108.46
voxelization                     5.43      67.01      22.02       8.41      98.05
image_loading                    2.64      32.64      11.74      14.62      64.89
precompute_depth                 2.04      24.85       9.67       1.30      62.99
precompute_geometry              0.19       2.36      14.91       0.00      98.55
npu_inference                   14.99     185.03       5.30     165.40     197.14
decode                           0.06       0.69       0.12       0.46       1.09
nms                              0.23       2.80       0.83       1.09       5.71
transform                        0.21       2.61       0.66       1.51       4.77
total_per_sample                28.76     355.01      33.82     250.99     462.15
========================================================================

==============================================================
BEVFusionFullNPUNet  TIMING SUMMARY
==============================================================
Stage            Total(s)   Mean(ms)    Std(ms)    Min(ms)    Max(ms)

--------------------------------------------------------------

lut_build           0.116     116.15       0.00     116.15     116.15
h2d                 2.438      30.10       4.27      27.03      58.77
infer              12.241     151.12       2.94     133.06     153.53
d2h                 0.079       0.97       0.15       0.81       2.09
total              15.055     185.86       6.00     164.08     214.29
==============================================================

此时模型精度几乎不变：mAP=0.2137.

为了进一步提高pipline性能，我认为通过高度并行化是极好的工作方向得益于BEVFusion模型的设计，雷达分支与图像分支在fusion前都可以独立进行，只有图像分支需要点云生成depth_map来进行LSS，但这对于我们现在的cpu+NPU设计来说仍然可以较好地进行解耦，那么优化包括以下几个方向：

1. 预处理并行化：通过观察可以发现，在预处理中，点云加载与图像加载可以并行；体素化与depth_map计算可以并行化。

2. 模型推理并行化：雷达特征提取分支与摄像头特征提取分支完全可以并行。

事实上，onnx与om在编译时会在一定程度上进行并行处理（多流并行），例如，通过atc命令可以看见模型的info：

```
atc --mode=6 --om="models/om/bevfusion_full_npu_dynamic.om"
ATC start working now, please wait for a moment.
============ Display Model Info start ============
Original Atc command line: /usr/local/Ascend/ascend-toolkit/8.3.RC1.alpha003/x86_64-linux/bin/atc.bin --model=models/onnx_full_npu/bevfusion_full_npu.onnx --framework=5 --output=models/om/bevfusion_full_npu_dynamic --input_format=ND --input_shape=voxels:-1,32,5;num_points:-1;coords:-1,4;imgs:1,6,3,256,704;depth:1,6,1,256,704;pool_lookup:129600,16;pool_mask:129600,16 --dynamic_dims=6000,6000,6000;8000,8000,8000;10000,10000,10000 --soc_version=Ascend310P1 --log=warning
system   info: atc_version[8.3.T14.0.B101], soc_version[Ascend310P1], framework_type[Onnx].
resource info: memory_size[767592960 B], weight_size[201064448 B], stream_num[10], event_num[7].
om       info: modeldef_size[5544269 B], weight_data_size[201064448 B], tbe_kernels_size[1864911 B], cust_aicpu_kernel_store_size[0 B], task_info_size[364448 B], so_store_size[0 B].
============ Display Model Info end   ============
```

上面的命令中有：

```
stream_num[10], event_num[7].
```

`stream_num=10`：支持 10 路并行计算流，

`event_num=7`：计算同步事件数

这表示在atc时，已经将模型在一定程度上进行了并行，但这与理想的“camera与lidar并行”并不冲突，通过在业务侧显式控制两条分支的开始，结束，中间结果转移到head推理，更能实现我们的目的。

为了实现模型并行，有下面3个方法：

| **方案** | **思路**                           | **优点**                                    | **局限**                                        | **结论**    |
| ------ | -------------------------------- | ----------------------------------------- | --------------------------------------------- | --------- |
| A      | 继续保持单 OM                         | 业务层改动最小；导出最简单                             | 无法显式控制两分支并行；静态输入和中间结果难做精细生命周期管理               | 不选为主方案    |
| B      | 单 OM + 多线程/多 stream              | 看上去改动小                                    | 同一个 model_id 无法在多 stream 并发推理，工程上不可作为“真正并行”方案 | 不可作为可靠主方案 |
| C      | 三 OM：LiDAR / Camera 并行，Fusion 收尾 | 分支边界自然；能显式并行；能复用中间 device buffer；数学结构基本不变 | 导出链路更复杂；需要管理 3 个 model_id / stream            | 主方案       |

于是，新的模型推理优化设计如下：
一、模型边界：

1. LiDAR Branch OM：输入 voxels / num_points / coords，输出 lidar_bev。
2. Camera Branch OM：输入 imgs / depth / pool_lookup / pool_mask，输出 camera_bev。
3. Fusion Head OM：输入 camera_bev / lidar_bev，输出检测头结果（dense_heatmap、center、dim、
   rot、vel 等）

这样切分有三个直接好处。第一，LiDAR 与 Camera 分支可并发；第二，Fusion 前的接口是两个已经聚合后的 BEV 特征图，比传中间多尺度特征更稳；第三，数学结构基本没变，只是把原本 full OM 中已经存在的自然边界显式化。

二、runtime并行：

运行时显式创建了 3 个 stream：lidar_stream、camera_stream、fusion_stream。LiDAR 分支和 Camera 分支分别在自己的 stream 中用 execute_async 下发；两路都完成后，Fusion Head 再在第三个 stream 中执行。

通过实验发现lidar分支会花更多时间，将lidar分支先下发，减少wall time。

三、H2D过程优化

1. 将计算的查找表LUT 固定在device上，减少copy时间。

2. 将lidar branch与camera branch的输出直接接到head的输入上，避免无味的H2D,D2H或者D2D。

通过上面分析的优化手段（预处理并行与模型推理并行），得到下面的结果：
========================================================================

TIMING STATISTICS SUMMARY
========================================================================

Task                         Total(s)   Mean(ms)    Std(ms)    Min(ms)    Max(ms)

------------------------------------------------------------------------

sweep_loading                    5.13      63.27      16.25      12.97     105.50
voxelization                     1.75      21.60      11.10       4.81      84.10
image_loading                    4.23      52.25      13.77      24.64     130.45
precompute_depth                 4.12      50.87      14.79       6.66      89.85
precompute_geometry              0.05       0.63       5.63       0.00      50.99
npu_inference                   10.95     135.18       3.40     118.38     144.68
decode                           0.05       0.66       0.10       0.43       0.94
nms                              0.22       2.69       0.82       1.02       4.84
transform                        0.21       2.54       0.70       1.25       4.18
total_per_sample                21.88     270.11      30.92     175.23     362.86
========================================================================

====================================================================
BEVFusionParallelNPUNet TIMING SUMMARY
====================================================================
Stage              Total(s)   Mean(ms)    Std(ms)    Min(ms)    Max(ms)

--------------------------------------------------------------------

lut_build             0.132     131.93       0.00     131.93     131.93
lidar_h2d             0.325       4.01       0.99       2.57      10.24
camera_h2d            0.808       9.97       0.74       9.11      13.23
branch_stage          7.114      87.83       3.37      71.06      97.42
fusion_h2d            0.000       0.00       0.00       0.00       0.01
fusion_infer          3.695      45.61       0.22      45.20      46.28
fusion_d2h            0.078       0.96       0.07       0.78       1.25
total                10.945     135.12       3.39     118.34     144.64
====================================================================

实现了模型推理耗时：185ms->135ms，全流程pipline耗时：350ms->270ms。

进一步优化方案设计：
收到AscendC算子开发中，二级缓冲的启发，在进行推理时，可以将下一帧的数据copy到buffer上，具体实现方式是：
在 `net` 里预分配了两个 host slot，分别存：  
`voxels / num_points / coords / imgs / depth`。  
这样“当前帧”推理时，“下一帧”可以先把预处理结果写进另一个 slot，避免互相踩内存，也减少反复分配/拼接。

同时，还可以将这一思想用到预处理与推理的过程中，将这两个过程进行流水化，具体方式为：

`evaluator` 现在会把“下一帧”的：  
点云加载、图像加载、体素化、depth 预计算、host staging  放到后台预取线程里，和“当前帧”的 NPU 推理 + 后处理重叠起来。  
结果上面的优化，运行结果如下：

========================================================================
TIMING STATISTICS SUMMARY
========================================================================
Task                         Total(s)   Mean(ms)    Std(ms)    Min(ms)    Max(ms)

------------------------------------------------------------------------

sweep_loading                    4.33      53.40      21.75       4.89     147.99
voxelization                     1.47      18.14      21.15       2.61      99.66
image_loading                    1.96      24.21      17.36      11.40     141.30
precompute_depth                 6.87      84.86      25.28       6.06     139.15
precompute_geometry              0.05       0.57       5.14       0.00      46.51
npu_inference                   10.83     133.71       4.06     117.02     147.62
decode                           0.11       1.40       0.32       0.64       2.84
nms                              0.51       6.32       2.25       1.91      12.41
transform                        0.53       6.50       2.50       2.07      14.71
total_per_sample                13.45     166.09      38.94     128.37     450.14
host_staging                     0.32       3.90       6.14       1.19      46.37
prefetch_wait                    1.20      14.78      24.00       0.01     135.36
========================================================================

Pipeline summary:
  Total wall time      : 13.473s
  Mean wall/sample     : 166.33ms
  Effective throughput : 6.01 samples/s

====================================================================
BEVFusionParallelNPUNet-v2 TIMING SUMMARY
====================================================================
Stage              Total(s)   Mean(ms)    Std(ms)    Min(ms)    Max(ms)

--------------------------------------------------------------------

lut_build             0.235     234.67       0.00     234.67     234.67
host_stage            0.316       3.90       6.14       1.19      46.37
lidar_h2d             0.275       3.39       0.28       2.43       4.93
camera_h2d            0.765       9.44       0.83       8.41      15.74
branch_stage          7.056      87.11       3.98      70.22     101.49
fusion_h2d            0.000       0.00       0.00       0.00       0.01
fusion_infer          3.687      45.52       0.53      45.02      47.76
fusion_d2h            0.080       0.99       0.29       0.82       3.41
total                10.825     133.65       4.06     116.99     147.58
====================================================================

全流程pipline耗时：270ms->166ms,fps达到6。

下一步工作计划：

量化，初步计划使用PTQ（Post_Train Quantization）。