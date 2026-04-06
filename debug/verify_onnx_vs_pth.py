"""
验证 ONNX 与 PyTorch 输出一致性（最终版）

验证策略：
  不要求 200 个候选的顺序完全一致（topk tie-breaking 在不同框架间不确定）。
  而是：
    1. dense_heatmap / dense_heatmap sigmoid 全图对比（与候选顺序无关）
    2. 对高置信目标（score > threshold）取交集，对比 bbox 预测
    3. 报告高置信目标的命中率和预测误差

用法:
    python verify_onnx_vs_pth.py \
        --onnx         models/onnx_final/bevfusion_final_static.onnx \
        --dynamic-onnx models/onnx_final/bevfusion_final_dynamic.onnx
"""

import argparse
import os
from os import path as osp
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, osp.dirname(__file__))
try:
    import fix_pytorch_weights_only  # noqa
except ModuleNotFoundError:
    pass

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("pip install onnxruntime-gpu")

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmdet3d.utils import register_all_modules
from mmdet3d.apis import init_model


# ════════════════════════════════════════════════════════════════════════════
#  Deploy 模块（与导出脚本一致）
# ════════════════════════════════════════════════════════════════════════════

class OnnxPointPillarsScatter(nn.Module):
    def __init__(self, in_channels, ny, nx):
        super().__init__()
        self.C, self.ny, self.nx = in_channels, ny, nx

    def forward(self, voxel_features, coors, batch_size):
        C, ny, nx = self.C, self.ny, self.nx
        HW      = ny * nx
        indices = coors[:,0].long() * HW + coors[:,2].long() * nx + coors[:,3].long()
        canvas  = torch.zeros(C, batch_size * HW,
                              dtype=voxel_features.dtype,
                              device=voxel_features.device)
        canvas  = canvas.scatter(1, indices.unsqueeze(0).expand(C, -1),
                                 voxel_features.t())
        return canvas.view(C, batch_size, ny, nx).permute(1, 0, 2, 3).contiguous()


class BEVFusionDeployFinal(nn.Module):
    TIEBREAK_EPS = 1e-6

    def __init__(self, model, K=200):
        super().__init__()
        self.pts_voxel_encoder  = model.pts_voxel_encoder
        self.pts_backbone       = model.pts_backbone
        self.pts_neck           = model.pts_neck
        self.head               = model.bbox_head
        self.K                  = K
        orig = model.pts_middle_encoder
        self.pts_middle_encoder = OnnxPointPillarsScatter(
            orig.in_channels, orig.ny, orig.nx)

    def forward(self, voxels, num_points, coords):
        vf         = self.pts_voxel_encoder(voxels, num_points, coords)
        batch_size = int(coords[-1, 0].item()) + 1
        bev        = self.pts_middle_encoder(vf, coords, batch_size)
        neck       = self.pts_neck(list(self.pts_backbone(bev)))[0]

        head = self.head
        B    = neck.shape[0]
        ff   = head.shared_conv(neck)
        _, C, H, W = ff.shape
        HW      = H * W
        ff_flat = ff.view(B, C, HW)
        bev_pos = head.bev_pos.repeat(B, 1, 1).to(ff.device)

        dm  = head.heatmap_head(ff)
        hm  = torch.sigmoid(dm)
        p   = head.nms_kernel_size // 2
        lm  = F.max_pool2d(hm, kernel_size=head.nms_kernel_size, stride=1, padding=p)
        hmf = (hm * (hm == lm)).view(B, head.num_classes, HW)

        num_el_tensor = hmf.shape[2] * head.num_classes
        idx_t = torch.arange(num_el_tensor, dtype=hmf.dtype, device=hmf.device)
        bias  = (self.TIEBREAK_EPS * (num_el_tensor - idx_t) / num_el_tensor).view(
                     1, head.num_classes, hmf.shape[2])
        _, top  = torch.topk((hmf + bias).view(B, -1), k=self.K, dim=-1)
        top_cls = top // HW
        top_idx = top % HW

        qf = ff_flat.gather(-1, top_idx[:, None, :].expand(-1, C, -1))
        qf = qf + head.class_encoding(
                F.one_hot(top_cls, head.num_classes).permute(0, 2, 1).float())
        qp = bev_pos.gather(1, top_idx[:, :, None].expand(-1, -1, bev_pos.shape[-1]))

        for i in range(head.num_decoder_layers):
            qf  = head.decoder[i](qf, key=ff_flat, query_pos=qp, key_pos=bev_pos)
            res = head.prediction_heads[i](qf)
            res["center"] = res["center"] + qp.permute(0, 2, 1)
            qp  = res["center"].permute(0, 2, 1)

        qhs = hmf.gather(-1, top_idx[:, None, :].expand(-1, head.num_classes, -1))
        return (dm, top_cls.to(torch.int32), qhs,
                res["heatmap"], res["center"], res["height"], res["dim"], res["rot"],
                res.get("vel", torch.zeros(B, 2, self.K, device=voxels.device)))


# ════════════════════════════════════════════════════════════════════════════
#  输入生成
# ════════════════════════════════════════════════════════════════════════════

def build_inputs(num_voxels, M=32, Cin=5, seed=42, device="cuda", ny=360, nx=360):
    torch.manual_seed(seed)
    np.random.seed(seed)
    voxels     = torch.randn(num_voxels, M, Cin, device=device)
    num_points = torch.randint(1, M+1, (num_voxels,), device=device)
    flat_idx   = torch.randperm(ny * nx, device=device)[:num_voxels]
    coords     = torch.zeros(num_voxels, 4, device=device)
    coords[:, 2] = (flat_idx // nx).float()
    coords[:, 3] = (flat_idx %  nx).float()
    return voxels, num_points, coords[coords[:, 0].argsort()]


# ════════════════════════════════════════════════════════════════════════════
#  推理
# ════════════════════════════════════════════════════════════════════════════

OUTPUT_NAMES = ["dense_heatmap", "top_cls", "query_heatmap_score",
                "heatmap_q", "center", "height", "dim", "rot", "vel"]


def run_pt(deploy, voxels, num_points, coords):
    with torch.no_grad():
        outs = deploy(voxels, num_points, coords)
    return [o.cpu().numpy() for o in outs]


def run_ort(path, voxels, num_points, coords):
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if ort.get_device() == "GPU" else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(path, providers=providers)
    return sess.run(
        None,
        {
            "voxels": voxels.cpu().numpy().astype(np.float32),
            "num_points": num_points.cpu().numpy().astype(np.int64),  # 如果需要
            "coords": coords.cpu().numpy().astype(np.float32),          # 如果需要
        }
    )


# ════════════════════════════════════════════════════════════════════════════
#  对比函数
# ════════════════════════════════════════════════════════════════════════════

def compare_heatmap(pt_outs, onnx_outs, atol=1e-2):
    """对比 dense_heatmap（全图，与候选顺序无关）"""
    pt   = pt_outs[0].astype(np.float32)
    onnx = onnx_outs[0].astype(np.float32)
    diff = np.abs(pt - onnx)
    ok   = np.allclose(pt, onnx, atol=atol, rtol=atol)
    print(f"  dense_heatmap  shape={pt.shape}")
    print(f"    max|Δ|={diff.max():.4e}  mean|Δ|={diff.mean():.4e}  "
          f"RMSE={np.sqrt((diff**2).mean()):.4e}  "
          f"allclose(atol={atol}): {'✅' if ok else '❌'}")
    return ok


def get_high_conf_preds(outs, score_thresh=0.1):
    """
    从模型输出中提取高置信度预测。
    score = sigmoid(heatmap_q) 取各类最大值。
    返回 dict: idx -> {score, cls, center, height, dim, rot, vel}
    """
    # outs 顺序: dense_heatmap, top_cls, query_heatmap_score,
    #            heatmap_q, center, height, dim, rot, vel
    top_cls = outs[1][0]          # [K]
    heatmap_q = outs[3][0]        # [num_cls, K]
    center    = outs[4][0]        # [2, K]
    height    = outs[5][0]        # [1, K]
    dim       = outs[6][0]        # [3, K]
    rot       = outs[7][0]        # [2, K]
    vel       = outs[8][0]        # [2, K]

    scores = 1 / (1 + np.exp(-heatmap_q))  # sigmoid
    # 每个候选取其对应类别的分数
    K = top_cls.shape[0]
    per_pred_score = scores[top_cls, np.arange(K)]  # [K]

    mask = per_pred_score > score_thresh
    result = {
        "score":  per_pred_score[mask],
        "cls":    top_cls[mask],
        "center": center[:, mask].T,   # [N, 2]
        "height": height[0, mask],     # [N]
        "dim":    dim[:, mask].T,      # [N, 3]
        "rot":    rot[:, mask].T,      # [N, 2]
        "vel":    vel[:, mask].T,      # [N, 2]
    }
    return result


def compare_high_conf(pt_outs, onnx_outs, score_thresh=0.1, center_tol=2.0, label="ONNX"):
    """
    对高置信目标做匹配对比：
    - 对每个 PT 高置信预测，在 ONNX 输出中找最近邻（按 center 距离）
    - 报告命中率和 bbox 误差
    """
    pt_pred   = get_high_conf_preds(pt_outs,   score_thresh)
    onnx_pred = get_high_conf_preds(onnx_outs, score_thresh)

    n_pt   = len(pt_pred["score"])
    n_onnx = len(onnx_pred["score"])

    print(f"[{label}] 高置信目标对比 (score_thresh={score_thresh})")
    print(f"    PyTorch 高置信目标数: {n_pt}")
    print(f"    ONNX    高置信目标数: {n_onnx}")

    if n_pt == 0:
        print("    ⚠️  PyTorch 无高置信目标，跳过对比")
        return True

    if n_onnx == 0:
        print("    ❌ ONNX 无高置信目标")
        return False

    # 按 center 做最近邻匹配
    pt_centers   = pt_pred["center"]    # [N_pt, 2]
    onnx_centers = onnx_pred["center"]  # [N_onnx, 2]

    # 距离矩阵 [N_pt, N_onnx]
    diff_mat = pt_centers[:, None, :] - onnx_centers[None, :, :]  # [N_pt, N_onnx, 2]
    dist_mat = np.linalg.norm(diff_mat, axis=-1)                   # [N_pt, N_onnx]

    matched_dists  = []
    matched_cls_ok = []
    matched_dim_err = []
    n_matched = 0

    for i in range(n_pt):
        j = dist_mat[i].argmin()
        d = dist_mat[i, j]
        if d < center_tol:
            n_matched += 1
            matched_dists.append(d)
            matched_cls_ok.append(pt_pred["cls"][i] == onnx_pred["cls"][j])
            dim_err = np.abs(pt_pred["dim"][i] - onnx_pred["dim"][j]).max()
            matched_dim_err.append(dim_err)

    hit_rate = n_matched / n_pt if n_pt > 0 else 0
    all_ok   = hit_rate >= 0.9

    print(f"    匹配率 (center_tol={center_tol}): {n_matched}/{n_pt} = {hit_rate*100:.1f}%"
          f"  {'✅' if all_ok else '❌'}")
    if matched_dists:
        print(f"    center 误差: mean={np.mean(matched_dists):.4f}  "
              f"max={np.max(matched_dists):.4f}")
        print(f"    类别一致率: {sum(matched_cls_ok)}/{len(matched_cls_ok)}")
        print(f"    dim 误差:   mean={np.mean(matched_dim_err):.4f}  "
              f"max={np.max(matched_dim_err):.4f}")
    return all_ok


def compare_all(pt_outs, onnx_outs, atol_heatmap=1e-2,
                score_thresh=0.1, center_tol=2.0, label="ONNX"):
    """综合对比：heatmap 全图 + 高置信目标匹配"""
    sep = "═" * 64
    print(f"\n{sep}")
    print(f"  对比报告: PyTorch  vs  {label}")
    print(sep)

    ok1 = compare_heatmap(pt_outs, onnx_outs, atol=atol_heatmap)
    ok2 = compare_high_conf(pt_outs, onnx_outs,
                            score_thresh=score_thresh,
                            center_tol=center_tol,
                            label=label)

    # 附加：原始逐输出误差（参考）
    print(f"\n  原始输出误差参考（不作为 pass/fail 标准）:")
    print(f"  {'名称':<22} {'max|Δ|':>10} {'mean|Δ|':>10}")
    print(f"  {'─'*44}")
    for name, pt, onnx in zip(OUTPUT_NAMES, pt_outs, onnx_outs):
        pt   = pt.astype(np.float32)
        onnx = onnx.astype(np.float32)
        diff = np.abs(pt - onnx)
        print(f"  {name:<22} {diff.max():>10.4e} {diff.mean():>10.4e}")

    all_ok = ok1 and ok2
    print(f"\n  总体结论: {'✅ PASS' if all_ok else '❌ FAIL'}")
    print(sep)
    return all_ok


# ════════════════════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",
                   default="src/configs/bevfusion_lidar_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d.py")
    p.add_argument("--ckpt",          default="work_dirs/bevfusion/epoch_20.pth")
    p.add_argument("--onnx",          default="models/onnx_final/bevfusion_final_static.onnx")
    p.add_argument("--dynamic-onnx",  default=None)
    p.add_argument("--num-voxels",    type=int,   default=6000)
    p.add_argument("--K",             type=int,   default=200)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--score-thresh",  type=float, default=0.1)
    p.add_argument("--center-tol",    type=float, default=2.0)
    p.add_argument("--atol-heatmap",  type=float, default=1e-2)
    p.add_argument("--cpu",           action="store_true")
    args   = p.parse_args()
    device = "cpu" if args.cpu else "cuda"

    print(f"[1/4] 加载模型: {args.ckpt}")
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    if "img_backbone" in cfg.model and "init_cfg" in cfg.model.get("img_backbone", {}):
        cfg.model["img_backbone"]["init_cfg"] = None
    model  = init_model(cfg, args.ckpt)
    model.eval()
    deploy = BEVFusionDeployFinal(model, K=args.K).eval()
    if device == "cuda":
        deploy = deploy.cuda()

    enc = model.pts_middle_encoder
    # print(f"[2/4] 生成唯一坐标输入 (V={args.num_voxels})")
    # voxels, num_points, coords = build_inputs(
    #     args.num_voxels, seed=args.seed, device=device,
    #     ny=enc.ny, nx=enc.nx)
    # 导入预先生成的输入（与导出时完全相同）
    print("[2/4] 预先生成的输入 ")
    om_sample_dir = "om_data/sample_0"
    om_voxels = np.load(os.path.join(om_sample_dir, 'voxels.npy'))
    om_coords = np.load(os.path.join(om_sample_dir, 'coords.npy'))
    om_num_points = np.load(os.path.join(om_sample_dir, 'num_points.npy'))
    # 转为 tensor
    voxels = torch.from_numpy(om_voxels).to(device)
    coords = torch.from_numpy(om_coords).to(device)
    num_points = torch.from_numpy(om_num_points).to(device)
    # coords = coords[:, [0, 3, 1, 2]].contiguous()  

    print("[3/4] PyTorch 推理...")
    pt_outs = run_pt(deploy, voxels, num_points, coords)
    pt_pred = get_high_conf_preds(pt_outs, args.score_thresh)
    print(f"      高置信目标数 (thresh={args.score_thresh}): {len(pt_pred['score'])}")

    results = {}
    # if args.onnx and os.path.isfile(args.onnx):
    #     print(f"[4/4] 静态 ONNX 推理...")
    #     onnx_outs = run_ort(args.onnx, voxels, num_points, coords)
    #     results["static"] = compare_all(
    #         pt_outs, onnx_outs,
    #         atol_heatmap=args.atol_heatmap,
    #         score_thresh=args.score_thresh,
    #         center_tol=args.center_tol,
    #         label=f"静态ONNX ({osp.basename(args.onnx)})")

    if args.dynamic_onnx and os.path.isfile(args.dynamic_onnx):
        print(f"[4b] 动态 ONNX 推理...")
        onnx_dyn_outs = run_ort(args.dynamic_onnx, voxels, num_points, coords)
        results["dynamic"] = compare_all(
            pt_outs, onnx_dyn_outs,
            atol_heatmap=args.atol_heatmap,
            score_thresh=args.score_thresh,
            center_tol=args.center_tol,
            label=f"动态ONNX ({osp.basename(args.dynamic_onnx)})")

    print("\n" + "="*64)
    print("验证汇总:")
    for k, v in results.items():
        print(f"  {k:<12}: {'✅ PASS' if v else '❌ FAIL'}")
    print("="*64)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()