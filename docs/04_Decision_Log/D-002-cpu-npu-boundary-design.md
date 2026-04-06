## D-002: CPU/NPU Boundary Design

### Date
2026-02-28

### Background
需要合理划分 CPU 和 NPU 的计算边界,平衡性能和开发复杂度。

### Options Considered

| Option | CPU | NPU | Pros | Cons |
|--------|-----|-----|------|------|
| A. 最小 NPU | Voxel, Pillar, Fusion, Head | 仅 Backbone | NPU 开发少 | 性能差 |
| B. 最大 NPU | 仅 Voxel, NMS | Pillar, Backbone, Fusion, Head | 性能最优 | 开发复杂 |
| C. 平衡方案 | Voxel, NMS | Pillar, Backbone, Fusion, Head | 平衡 | 需要优化数据传输 |

### Choice
**Option C: 平衡方案**

### Reason
1. Voxelization 涉及稀疏操作,NPU 不擅长,且需要定制动态输入算子
2. Backbone、Fusion、Head 是密集计算,NPU 效率高
3. NMS 计算量小,CPU 实现简单
4. 最小化 NPU 自定义算子开发

### Risk
- CPU/NPU 数据传输可能成为瓶颈
- CPU 实现性能可能不足

### Mitigation
- 使用共享内存减少数据拷贝
- 优化 CPU 实现(多线程、向量化)
- 监控数据传输时间

### Outcome
待验证
