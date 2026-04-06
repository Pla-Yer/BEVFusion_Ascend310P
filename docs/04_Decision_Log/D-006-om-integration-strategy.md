## D-006: OM模型集成策略选择

### Date
2026-03-20

### Background

当前为了优化，目标是将bevpool算子化插入到om模型中。面临以下问题：
1. 不知道怎么将bevpool算子打包进入om模型中，只知道怎么将一个算子打包成一个om模型
2. 不知道怎么将om模型进行多模型串行推理

需要决定采用哪种方式将算子集成到推理流程中。

### Options Considered

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A. 全过程单OM | 将全过程打包为一个om模型，将bevpool算子插入到om模型中 | 业务层改动最小；导出最简单 | 无法显式控制两分支并行；静态输入和中间结果难做精细生命周期管理 |
| B. 单OM + 多线程/多stream | 将bevpool算子单独打包为一个om模型，通过多模型串行进行推理 | 看上去改动小 | 同一个model_id无法在多stream并发推理，工程上不可作为"真正并行"方案 |
| C. 三OM串行 | 将bevpool算子打包入相机处理分支中，通过多模型串行进行推理 | 分支边界自然；能复用中间device buffer | 无法实现真正并行；串行执行效率低 |
| D. 三OM并行 | LiDAR/Camera并行，Fusion收尾 | 分支边界自然；能显式并行；能复用中间device buffer；数学结构基本不变 | 导出链路更复杂；需要管理3个model_id/stream |

### Choice
**Option D: 三OM并行架构**

### Reason
1. **分支边界自然**：LiDAR与Camera分支在fusion前可以独立进行，数学结构基本不变
2. **显式并行控制**：能显式控制两条分支的开始、结束、中间结果转移
3. **性能优势**：LiDAR与Camera分支可真正并发执行，显著降低wall time
4. **资源复用**：能复用中间device buffer，避免无谓的H2D、D2H或D2D操作

### Risk
- 导出链路更复杂，需要管理3个model_id和stream
- 需要确保三个模型之间的数据传递正确性
- 需要处理分支同步问题

### Mitigation
- 设计清晰的模型边界和接口定义
- 实现完善的错误处理和同步机制
- 通过profiling验证并行效果

### Outcome
**已成功实施，效果显著：**
- NPU推理时间：185ms → 135ms（提升26.9%）
- 总流程时间：355ms → 270ms（提升23.9%）
- 实现了LiDAR与Camera分支真正并行执行

**技术实现：**
- LiDAR Branch OM：输入voxels/num_points/coords，输出lidar_bev
- Camera Branch OM：输入imgs/depth/pool_lookup/pool_mask，输出camera_bev
- Fusion Head OM：输入camera_bev/lidar_bev，输出检测头结果
- Runtime创建3个stream实现并行执行
