"""
对比分析ONNX模型和OM模型的输入输出数据
使用新的验证策略：
  1. dense_heatmap全图对比（与候选顺序无关）
  2. 对高置信目标做匹配对比（基于center距离）
  3. 不要求topk顺序完全一致
"""

import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


class ONNXOMComparator:
    """ONNX和OM数据对比分析器"""

    def __init__(self, onnx_data_dir, om_data_dir, output_dir='./analysis_results'):
        """
        初始化对比器

        Args:
            onnx_data_dir: ONNX模型数据目录
            om_data_dir: OM模型数据目录
            output_dir: 分析结果输出目录
        """
        self.onnx_data_dir = onnx_data_dir
        self.om_data_dir = om_data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self, sample_token):
        """加载样本数据"""
        onnx_sample_dir = os.path.join(self.onnx_data_dir, sample_token)
        om_sample_dir = os.path.join(self.om_data_dir, sample_token)

        # 加载输入数据
        onnx_voxels = np.load(os.path.join(onnx_sample_dir, 'voxels.npy'))
        onnx_coords = np.load(os.path.join(onnx_sample_dir, 'coords.npy'))
        onnx_num_points = np.load(os.path.join(onnx_sample_dir, 'num_points.npy'))

        om_voxels = np.load(os.path.join(om_sample_dir, 'voxels.npy'))
        om_coords = np.load(os.path.join(om_sample_dir, 'coords.npy'))
        om_num_points = np.load(os.path.join(om_sample_dir, 'num_points.npy'))


        # 加载输出数据
        output_names = [
            'dense_heatmap', 'top_cls', 'query_heatmap_score',
            'heatmap_q', 'center', 'height', 'dim', 'rot', 'vel',
            'top_idx', 'top', 'top_score', 'keep', 'hm', 'lm'

        ]

        onnx_outputs = {}
        om_outputs = {}

        for name in output_names:
            onnx_outputs[name] = np.load(os.path.join(onnx_sample_dir, f'{name}.npy'))
            om_outputs[name] = np.load(os.path.join(om_sample_dir, f'{name}.npy'))

        return {
            'onnx': {
                'voxels': onnx_voxels,
                'coords': onnx_coords,
                'num_points': onnx_num_points,
                'outputs': onnx_outputs
            },
            'om': {
                'voxels': om_voxels,
                'coords': om_coords,
                'num_points': om_num_points,
                'outputs': om_outputs
            }
        }

    def compare_heatmap(self, onnx_hm, om_hm, atol=1e-2):
        """
        对比dense_heatmap（全图，与候选顺序无关）
        
        Args:
            onnx_hm: ONNX的dense_heatmap
            om_hm: OM的dense_heatmap
            atol: 绝对误差阈值
        """
        onnx = onnx_hm.astype(np.float32)
        om = om_hm.astype(np.float32)
        diff = np.abs(onnx - om)
        
        max_diff = diff.max()
        mean_diff = diff.mean()
        rmse = np.sqrt((diff**2).mean())
        ok = np.allclose(onnx, om, atol=atol, rtol=atol)
        
        print(f"  dense_heatmap  shape={onnx.shape}")
        print(f"    max|Δ|={max_diff:.4e}  mean|Δ|={mean_diff:.4e}  "
              f"RMSE={rmse:.4e}  "
              f"allclose(atol={atol}): {'✅' if ok else '❌'}")
        
        return {
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'rmse': float(rmse),
            'is_match': ok
        }

    def get_high_conf_preds(self, outputs, score_thresh=0.1):
        """
        从模型输出中提取高置信度预测
        score = sigmoid(heatmap_q) 取各类最大值
        
        Returns:
            dict: {score, cls, center, height, dim, rot, vel}
        """
        top_cls = outputs['top_cls'][0]          # [K]
        heatmap_q = outputs['heatmap_q'][0]      # [num_cls, K]
        center = outputs['center'][0]            # [2, K]
        height = outputs['height'][0]            # [1, K]
        dim = outputs['dim'][0]                  # [3, K]
        rot = outputs['rot'][0]                  # [2, K]
        vel = outputs['vel'][0]                  # [2, K]

        scores = 1 / (1 + np.exp(-heatmap_q))  # sigmoid
        # 每个候选取其对应类别的分数
        K = top_cls.shape[0]
        per_pred_score = scores[top_cls, np.arange(K)]  # [K]

        mask = per_pred_score > score_thresh
        result = {
            "score": per_pred_score[mask],
            "cls": top_cls[mask],
            "center": center[:, mask].T,   # [N, 2]
            "height": height[0, mask],     # [N]
            "dim": dim[:, mask].T,         # [N, 3]
            "rot": rot[:, mask].T,         # [N, 2]
            "vel": vel[:, mask].T,         # [N, 2]
        }
        return result

    def compare_high_conf(self, onnx_outputs, om_outputs, score_thresh=0.1, center_tol=2.0):
        """
        对高置信目标做匹配对比：
        - 对每个ONNX高置信预测，在OM输出中找最近邻（按center距离）
        - 报告命中率和bbox误差
        """
        onnx_pred = self.get_high_conf_preds(onnx_outputs, score_thresh)
        om_pred = self.get_high_conf_preds(om_outputs, score_thresh)

        n_onnx = len(onnx_pred["score"])
        n_om = len(om_pred["score"])

        print(f"[高置信目标对比] (score_thresh={score_thresh})")
        print(f"    ONNX 高置信目标数: {n_onnx}")
        print(f"    OM   高置信目标数: {n_om}")

        if n_onnx == 0:
            print("    ⚠️  ONNX 无高置信目标，跳过对比")
            return {'is_match': True, 'hit_rate': 1.0}

        if n_om == 0:
            print("    ❌ OM 无高置信目标")
            return {'is_match': False, 'hit_rate': 0.0}

        # 按center做最近邻匹配
        onnx_centers = onnx_pred["center"]    # [N_onnx, 2]
        om_centers = om_pred["center"]        # [N_om, 2]

        # 距离矩阵 [N_onnx, N_om]
        diff_mat = onnx_centers[:, None, :] - om_centers[None, :, :]  # [N_onnx, N_om, 2]
        dist_mat = np.linalg.norm(diff_mat, axis=-1)                   # [N_onnx, N_om]

        matched_dists = []
        matched_cls_ok = []
        matched_dim_err = []
        n_matched = 0

        for i in range(n_onnx):
            j = dist_mat[i].argmin()
            d = dist_mat[i, j]
            if d < center_tol:
                n_matched += 1
                matched_dists.append(d)
                matched_cls_ok.append(onnx_pred["cls"][i] == om_pred["cls"][j])
                dim_err = np.abs(onnx_pred["dim"][i] - om_pred["dim"][j]).max()
                matched_dim_err.append(dim_err)

        hit_rate = n_matched / n_onnx if n_onnx > 0 else 0
        all_ok = hit_rate >= 0.9

        print(f"    匹配率 (center_tol={center_tol}): {n_matched}/{n_onnx} = {hit_rate*100:.1f}%"
              f"  {'✅' if all_ok else '❌'}")
        if matched_dists:
            print(f"    center 误差: mean={np.mean(matched_dists):.4f}  "
                  f"max={np.max(matched_dists):.4f}")
            print(f"    类别一致率: {sum(matched_cls_ok)}/{len(matched_cls_ok)}")
            print(f"    dim 误差:   mean={np.mean(matched_dim_err):.4f}  "
                  f"max={np.max(matched_dim_err):.4f}")
        
        return {
            'is_match': all_ok,
            'hit_rate': hit_rate,
            'n_matched': n_matched,
            'n_onnx': n_onnx,
            'n_om': n_om,
            'center_error_mean': float(np.mean(matched_dists)) if matched_dists else 0.0,
            'center_error_max': float(np.max(matched_dists)) if matched_dists else 0.0,
            'cls_match_rate': sum(matched_cls_ok) / len(matched_cls_ok) if matched_cls_ok else 0.0,
            'dim_error_mean': float(np.mean(matched_dim_err)) if matched_dim_err else 0.0,
            'dim_error_max': float(np.max(matched_dim_err)) if matched_dim_err else 0.0
        }

    def analyze_sample(self, sample_token, score_thresh=0.1, center_tol=2.0, atol_heatmap=1e-2):
        """分析单个样本"""
        print(f"\n{'='*80}")
        print(f"分析样本: {sample_token}")
        print(f"{'='*80}")

        # 加载数据
        data = self.load_data(sample_token)

        results = {
            'sample_token': sample_token,
            'heatmap_comparison': {},
            'high_conf_comparison': {},
            'output_comparison': {},
            'diagnosis': {}
        }

        # 1. 对比dense_heatmap（全图）
        print("\n" + "="*60)
        print("1. dense_heatmap 全图对比")
        print("="*60)
        
        hm_result = self.compare_heatmap(
            data['onnx']['outputs']['dense_heatmap'],
            data['om']['outputs']['dense_heatmap'],
            atol=atol_heatmap
        )
        results['heatmap_comparison'] = hm_result

        # 2. 高置信目标匹配对比
        print("\n" + "="*60)
        print("2. 高置信目标匹配对比")
        print("="*60)
        
        conf_result = self.compare_high_conf(
            data['onnx']['outputs'],
            data['om']['outputs'],
            score_thresh=score_thresh,
            center_tol=center_tol
        )
        results['high_conf_comparison'] = conf_result

        # 3. 原始输出误差（参考）
        print("\n" + "="*60)
        print("3. 原始输出误差参考（不作为 pass/fail 标准）")
        print("="*60)
        
        output_names = [
            'dense_heatmap', 'top_cls', 'query_heatmap_score',
            'heatmap_q', 'center', 'height', 'dim', 'rot', 'vel',
            'top_idx', 'top', 'top_score', 'keep', 'hm', 'lm'
        ]
        
        print(f"  {'名称':<22} {'max|Δ|':>10} {'mean|Δ|':>10}")
        print(f"  {'─'*44}")
        
        for name in output_names:
            onnx = data['onnx']['outputs'][name].astype(np.float32)
            om = data['om']['outputs'][name].astype(np.float32)
            # 对onnx的keep进行形状转化
            if name == 'keep':
                onnx = onnx.reshape(om.shape)
            diff = np.abs(onnx - om)
            print(f"  {name:<22} {diff.max():>10.4e} {diff.mean():>10.4e}")
            
            results['output_comparison'][name] = {
                'max_diff': float(diff.max()),
                'mean_diff': float(diff.mean()),
                'sum_diff': float(diff.sum()),
            }

        # 4. 诊断分析
        print("\n" + "="*60)
        print("4. 诊断分析")
        print("="*60)
        
        diagnosis = self.diagnose(results)
        results['diagnosis'] = diagnosis

        # 5. 保存结果
        result_file = os.path.join(self.output_dir, f'{sample_token}_onnx_om_analysis.json')
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n分析结果已保存到: {result_file}")

        # 6. 生成可视化
        self.visualize_results(data, sample_token)

        return results

    def diagnose(self, results):
        """诊断问题"""
        diagnosis = {
            'heatmap_issue': False,
            'high_conf_issue': False,
            'issues': [],
            'recommendations': []
        }

        # 检查heatmap
        if not results['heatmap_comparison']['is_match']:
            diagnosis['heatmap_issue'] = True
            diagnosis['issues'].append("dense_heatmap误差过大")
            diagnosis['recommendations'].append("检查OM模型编译精度设置")
            diagnosis['recommendations'].append("检查ATC转换参数是否正确")

        # 检查高置信目标匹配
        if not results['high_conf_comparison']['is_match']:
            diagnosis['high_conf_issue'] = True
            diagnosis['issues'].append(f"高置信目标匹配率过低: {results['high_conf_comparison']['hit_rate']*100:.1f}%")
            diagnosis['recommendations'].append("检查算子精度模式配置")
            diagnosis['recommendations'].append("检查浮点精度是否导致topk结果不同")

        # 打印诊断结果
        print("\n诊断结果:")
        if diagnosis['issues']:
            for issue in diagnosis['issues']:
                print(f"  - {issue}")
            print("\n建议:")
            for rec in diagnosis['recommendations']:
                print(f"  - {rec}")
        else:
            print("  ✅ OM模型转换正确，输出与ONNX模型匹配")

        return diagnosis

    def visualize_results(self, data, sample_token):
        """可视化对比结果"""
        print("\n生成可视化图表...")

        # 创建图表目录
        viz_dir = os.path.join(self.output_dir, 'visualizations', sample_token)
        os.makedirs(viz_dir, exist_ok=True)

        # 1. 可视化heatmap对比
        onnx_heatmap = data['onnx']['outputs']['dense_heatmap']
        om_heatmap = data['om']['outputs']['dense_heatmap']

        if len(onnx_heatmap.shape) == 4:
            # 取第一个batch，第一个类别
            onnx_hm = onnx_heatmap[0, 0]
            om_hm = om_heatmap[0, 0]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            im0 = axes[0].imshow(onnx_hm, cmap='hot')
            axes[0].set_title('ONNX Heatmap')
            plt.colorbar(im0, ax=axes[0])

            im1 = axes[1].imshow(om_hm, cmap='hot')
            axes[1].set_title('OM Heatmap')
            plt.colorbar(im1, ax=axes[1])

            diff_hm = np.abs(onnx_hm - om_hm)
            im2 = axes[2].imshow(diff_hm, cmap='hot')
            axes[2].set_title('Difference')
            plt.colorbar(im2, ax=axes[2])

            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'heatmap_comparison.png'), dpi=150)
            plt.close()

        # 2. 可视化输出差异分布
        output_names = ['dense_heatmap', 'heatmap_q', 'center', 'height', 'dim', 'rot']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, name in enumerate(output_names):
            if name in data['onnx']['outputs'] and name in data['om']['outputs']:
                onnx_data = data['onnx']['outputs'][name].flatten()
                om_data = data['om']['outputs'][name].flatten()

                # 只取前10000个点进行可视化
                onnx_data = onnx_data[:10000]
                om_data = om_data[:10000]

                axes[idx].scatter(onnx_data, om_data, alpha=0.5, s=1)
                axes[idx].plot([onnx_data.min(), onnx_data.max()],
                              [onnx_data.min(), onnx_data.max()],
                              'r--', label='y=x')
                axes[idx].set_xlabel('ONNX')
                axes[idx].set_ylabel('OM')
                axes[idx].set_title(name)
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'output_scatter.png'), dpi=150)
        plt.close()

        print(f"可视化结果已保存到: {viz_dir}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='对比分析ONNX和OM模型数据')
    parser.add_argument('--onnx_data_dir', type=str,
                       default='./onnx_data',
                       help='ONNX模型数据目录')
    parser.add_argument('--om_data_dir', type=str,
                       default='./om_data',
                       help='OM模型数据目录')
    parser.add_argument('--sample_token', type=str,
                       default='sample_0',
                       help='样本标识')
    parser.add_argument('--output_dir', type=str,
                       default='./analysis_results',
                       help='分析结果输出目录')
    parser.add_argument('--score_thresh', type=float,
                       default=0.1,
                       help='高置信目标分数阈值')
    parser.add_argument('--center_tol', type=float,
                       default=2.0,
                       help='center距离容忍度')
    parser.add_argument('--atol_heatmap', type=float,
                       default=1e-2,
                       help='heatmap绝对误差阈值')

    args = parser.parse_args()

    # 创建对比器
    comparator = ONNXOMComparator(
        args.onnx_data_dir,
        args.om_data_dir,
        args.output_dir
    )

    # 分析样本
    comparator.analyze_sample(
        args.sample_token,
        score_thresh=args.score_thresh,
        center_tol=args.center_tol,
        atol_heatmap=args.atol_heatmap
    )

    print("\n分析完成!")


if __name__ == '__main__':
    main()
