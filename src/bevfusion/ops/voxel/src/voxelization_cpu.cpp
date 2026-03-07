#include <ATen/TensorUtils.h>
#include <torch/extension.h>

namespace {

template <typename T, typename T_int>
void dynamic_voxelize_kernel(const torch::TensorAccessor<T, 2> points,
                             torch::TensorAccessor<T_int, 2> coors,
                             const std::vector<float> voxel_size,
                             const std::vector<float> coors_range,
                             const std::vector<int> grid_size,
                             const int num_points, const int num_features,
                             const int NDim) {
  bool failed = false;
  int* coor = new int[NDim]();
  int c;

  for (int i = 0; i < num_points; ++i) {
    failed = false;
    for (int j = 0; j < NDim; ++j) {
      c = floor((points[i][j] - coors_range[j]) / voxel_size[j]);
      // 过滤超出范围的点
      if (c < 0 || c >= grid_size[j]) {
        failed = true;
        break;
      }
      coor[j] = c;
    }

    for (int k = 0; k < NDim; ++k) {
      if (failed)
        coors[i][k] = -1;
      else
        coors[i][k] = coor[k];
    }
  }

  delete[] coor;
  return;
}

template <typename T, typename T_int>
void hard_voxelize_kernel(const torch::TensorAccessor<T, 2> points,
                          torch::TensorAccessor<T, 3> voxels,
                          torch::TensorAccessor<T_int, 2> coors,
                          torch::TensorAccessor<T_int, 1> num_points_per_voxel,
                          torch::TensorAccessor<T_int, 3> coor_to_voxelidx,
                          int& voxel_num,
                          const std::vector<float> voxel_size,
                          const std::vector<float> coors_range,
                          const std::vector<int> grid_size,
                          const int max_points, const int max_voxels,
                          const int num_points, const int num_features,
                          const int NDim) {
  // 先用 dynamic voxelization 获取每个点的体素坐标
  at::Tensor temp_coors = at::zeros(
      {num_points, NDim},
      at::TensorOptions().dtype(at::kInt).device(at::kCPU));

  dynamic_voxelize_kernel<T, int>(
      points, temp_coors.accessor<int, 2>(),
      voxel_size, coors_range, grid_size,
      num_points, num_features, NDim);

  auto coor = temp_coors.accessor<int, 2>();

  for (int i = 0; i < num_points; ++i) {
    // 跳过越界点（dynamic kernel 标记为 -1）
    if (coor[i][0] == -1) continue;

    // ==============================================================
    // BUG FIX：原代码 coor_to_voxelidx 创建时维度顺序为
    //   {grid_size[2], grid_size[1], grid_size[0]}  → (Z, Y, X)
    // 但访问时用的是 [coor[i][0]][coor[i][1]][coor[i][2]] → (X, Y, Z)
    // 维度大小与索引值完全错位，导致越界 → 段错误！
    //
    // 修复：访问顺序改为与创建时一致的 (Z=coor[2], Y=coor[1], X=coor[0])
    // ==============================================================
    int voxelidx =
        coor_to_voxelidx[coor[i][2]][coor[i][1]][coor[i][0]];

    if (voxelidx == -1) {
      voxelidx = voxel_num;
      if (max_voxels != -1 && voxel_num >= max_voxels) continue;
      voxel_num += 1;

      coor_to_voxelidx[coor[i][2]][coor[i][1]][coor[i][0]] = voxelidx;

      for (int k = 0; k < NDim; ++k) {
        coors[voxelidx][k] = coor[i][k];
      }
    }

    // 将点填入体素
    int num = num_points_per_voxel[voxelidx];
    if (max_points == -1 || num < max_points) {
      for (int k = 0; k < num_features; ++k) {
        voxels[voxelidx][num][k] = points[i][k];
      }
      num_points_per_voxel[voxelidx] += 1;
    }
  }

  return;
}

}  // namespace

namespace voxelization {

int hard_voxelize_cpu(const at::Tensor& points,
                      at::Tensor& voxels,
                      at::Tensor& coors,
                      at::Tensor& num_points_per_voxel,
                      const std::vector<float> voxel_size,
                      const std::vector<float> coors_range,
                      const int max_points,
                      const int max_voxels,
                      const int NDim = 3) {
  AT_ASSERTM(points.device().is_cpu(), "points must be a CPU tensor");

  std::vector<int> grid_size(NDim);
  const int num_points = points.size(0);
  const int num_features = points.size(1);

  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
  }

  // ==============================================================
  // coor_to_voxelidx 形状：{grid_size[2], grid_size[1], grid_size[0]}
  //   → 对应 (Z轴格数, Y轴格数, X轴格数)
  // 访问时必须用 [z_idx][y_idx][x_idx]，见 kernel 中的修复
  // ==============================================================
  at::Tensor coor_to_voxelidx = -at::ones(
      {grid_size[2], grid_size[1], grid_size[0]}, coors.options());

  int voxel_num = 0;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      points.scalar_type(), "hard_voxelize_forward", [&] {
        hard_voxelize_kernel<scalar_t, int>(
            points.accessor<scalar_t, 2>(),
            voxels.accessor<scalar_t, 3>(),
            coors.accessor<int, 2>(),
            num_points_per_voxel.accessor<int, 1>(),
            coor_to_voxelidx.accessor<int, 3>(),
            voxel_num, voxel_size, coors_range, grid_size,
            max_points, max_voxels, num_points, num_features, NDim);
      });

  return voxel_num;
}

void dynamic_voxelize_cpu(const at::Tensor& points,
                          at::Tensor& coors,
                          const std::vector<float> voxel_size,
                          const std::vector<float> coors_range,
                          const int NDim = 3) {
  AT_ASSERTM(points.device().is_cpu(), "points must be a CPU tensor");

  std::vector<int> grid_size(NDim);
  const int num_points = points.size(0);
  const int num_features = points.size(1);

  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      points.scalar_type(), "hard_voxelize_forward", [&] {
        dynamic_voxelize_kernel<scalar_t, int>(
            points.accessor<scalar_t, 2>(),
            coors.accessor<int, 2>(),
            voxel_size, coors_range, grid_size,
            num_points, num_features, NDim);
      });

  return;
}

}  // namespace voxelization