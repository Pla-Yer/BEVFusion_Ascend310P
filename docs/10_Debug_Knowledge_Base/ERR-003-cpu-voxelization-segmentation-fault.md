## ERR-003: CPU体素化算子段错误

### Date
2026-03-03

### Environment
- OS: Linux
- Component: CPU体素化算子
- Source File: src/bevfusion/ops/voxel/src/voxelization_cpu.cpp
- Error Type: 段错误 (核心已转储)

### Symptom
使用编译的CPU体素化算子时,程序崩溃并出现段错误:
```
段错误 (核心已转储)
```

该错误通常是由于索引超出内存范围导致的,导致体素化功能无法正常使用。

### Debug Process
1. 分析段错误的常见原因,初步判断为内存访问越界
2. 定位到源代码文件: `src/bevfusion/ops/voxel/src/voxelization_cpu.cpp`
3. 逐行检查内存访问相关代码
4. 分析`hard_voxelize_kernel`和`dynamic_voxelize_kernel`两个核心函数
5. 发现维度顺序与访问顺序不匹配的问题
6. 验证修复方案

### Root Cause

#### 问题1: hard_voxelize_kernel 中 coor_to_voxelidx 维度顺序与访问顺序不匹配

**错误代码**:
```cpp
// 创建时:维度顺序是 (Z, Y, X)
at::Tensor coor_to_voxelidx =
    -at::ones({grid_size[2], grid_size[1], grid_size[0]}, ...);

// 访问时:索引顺序是 (X, Y, Z)  ← 与创建顺序相反!
voxelidx = coor_to_voxelidx[coor[i][0]][coor[i][1]][coor[i][2]];
//                                  X↑           Y↑           Z↑
```

**为什么会崩溃**:

| 维度 | tensor实际大小 | 原代码用什么索引访问 |
|------|---------------|---------------------|
| dim0 | grid_size[2] (Z方向格数,通常最小,如1~4) | coor[i][0] (X方向索引,可能很大) |
| dim1 | grid_size[1] (Y方向格数) | coor[i][1] (Y,偶然对了) |
| dim2 | grid_size[0] (X方向格数,通常最大) | coor[i][2] (Z,通常很小) |

X方向的索引值(可以很大)被用来索引Z方向大小的维度,立即越界 → 段错误。

同样地,`coor_to_voxelidx[...] = voxelidx`的写入也存在相同问题,会破坏内存。

#### 问题2: dynamic_voxelize_kernel 中 coor 数组索引与 coors_range/voxel_size 维度语义不一致

`points[i][j]`按j=0,1,2对应X,Y,Z,而coors_range和grid_size的布局需要确认与之一致。

### Fix

**修复后的代码**:
```cpp
// 访问时改为与创建维度一致的 (Z, Y, X) 顺序
voxelidx = coor_to_voxelidx[coor[i][2]][coor[i][1]][coor[i][0]];
//                                  Z↑           Y↑           X↑
```

**关键修改点**:
1. 将`coor_to_voxelidx`的访问索引顺序从`[coor[i][0]][coor[i][1]][coor[i][2]]`改为`[coor[i][2]][coor[i][1]][coor[i][0]]`
2. 确保访问顺序与创建时的维度顺序(Z, Y, X)一致
3. 同步修复所有相关的读写操作

### Verification
修复后:
- 重新编译CPU体素化算子
- 运行测试用例,无段错误
- 验证体素化结果正确性
- 在完整pipeline中测试通过

### Lessons & Notes
- **维度顺序一致性至关重要**: 创建tensor时的维度顺序必须与访问时的索引顺序完全一致
- **坐标系约定要明确**: X, Y, Z的顺序在不同上下文中可能不同,需要统一约定
- **段错误调试方法**:
  - 首先检查数组索引是否越界
  - 验证维度顺序和访问顺序是否匹配
  - 使用调试工具(gdb)定位崩溃位置
- **代码审查重点**: 涉及多维数组访问的代码,必须仔细检查维度顺序
- **测试建议**: 使用边界值测试,更容易发现维度不匹配问题
