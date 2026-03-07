# 导出脚本修复说明

## 修复的问题

### 1. Conda路径错误
**原问题**: 脚本中使用 `~/miniconda3/etc/profile.d/conda.sh`，但实际conda安装在 `~/anaconda3`

**修复方案**:
```bash
# 修复前
source ~/miniconda3/etc/profile.d/conda.sh

# 修复后
source ~/anaconda3/etc/profile.d/conda.sh
```

### 2. 项目根目录路径计算错误
**原问题**: 脚本在 `src/export/` 目录下，使用 `dirname "$SCRIPT_DIR"` 只能获取到 `src` 目录，而不是项目根目录

**修复方案**:
```bash
# 修复前
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORK_DIR="$(dirname "$SCRIPT_DIR")"  # 得到 src 目录

# 修复后
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORK_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"  # 得到项目根目录
```

## 修复的文件列表

### OM导出脚本
- `src/export/export_voxel_encoder_om.sh`
- `src/export/export_pts_backbone_om.sh`
- `src/export/export_pts_neck_om.sh`
- `src/export/export_transfusion_head_om.sh`
- `src/export/export_bevfusion_full_om.sh`

### 精度验证脚本
- `src/accuracy_check/run_check.sh`

## 验证结果

### ✓ 修复验证
```bash
# 测试conda环境激活
bash -c 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate ascend-py3.7.10 && echo "Conda环境激活成功"'
# 输出: Conda环境激活成功

# 测试OM导出脚本（路径查找）
bash src/export/export_voxel_encoder_om.sh
# 输出:
# 开始导出 OM 模型...
# 输入模型: /home/ttt/PY080313/BEVFusion_Ascend310P/models/onnx/voxel_encoder.onnx
# 输出目录: /home/ttt/PY080313/BEVFusion_Ascend310P/models/om
```

### ⚠️ 已知问题

**voxel_encoder OM导出失败**: 不是脚本问题，而是ONNX模型本身的问题
- 错误信息: Concat算子的输入形状不匹配
- 原因: PillarFeatureNet使用了动态形状的Concat算子
- 解决方案: 需要修改ONNX导出脚本，使用更合适的导出参数或简化模型结构

## 使用说明

### 导出单个模型
```bash
# 导出体素编码器
bash src/export/export_voxel_encoder_om.sh

# 导出点云骨干网络
bash src/export/export_pts_backbone_om.sh

# 导出点云颈部网络
bash src/export/export_pts_neck_om.sh

# 导出检测头
bash src/export/export_transfusion_head_om.sh

# 导出完整模型
bash src/export/export_bevfusion_full_om.sh
```

### 导出所有模型
```bash
bash src/export/export_all_om.sh
```

### 精度验证
```bash
bash src/accuracy_check/run_check.sh
```

## 路径说明

### 目录结构
```
BEVFusion_Ascend310P/              # 项目根目录 ($WORK_DIR)
├── src/
│   ├── export/                    # 导出脚本目录 ($SCRIPT_DIR)
│   │   ├── export_*.sh           # OM导出脚本
│   │   └── export_*.py           # ONNX导出脚本
│   └── accuracy_check/           # 精度验证目录
│       ├── check_*.py            # 精度验证脚本
│       └── run_check.sh          # 运行脚本
├── models/
│   ├── onnx/                     # ONNX模型目录
│   │   ├── voxel_encoder.onnx
│   │   ├── pts_backbone.onnx
│   │   ├── pts_neck.onnx
│   │   ├── transfusion_head.onnx
│   │   └── bevfusion_full.onnx
│   └── om/                       # OM模型目录
│       ├── voxel_encoder.om
│       ├── pts_backbone.om
│       ├── pts_neck.om
│       ├── transfusion_head.om
│       └── bevfusion_full.om
└── ...
```

### 路径计算逻辑
```bash
# 脚本位置: /home/ttt/PY080313/BEVFusion_Ascend310P/src/export/export_voxel_encoder_om.sh

SCRIPT_DIR="/home/ttt/PY080313/BEVFusion_Ascend310P/src/export"
WORK_DIR="/home/ttt/PY080313/BEVFusion_Ascend310P"

# ONNX模型路径
ONNX_MODEL="$WORK_DIR/models/onnx/voxel_encoder.onnx"
# = /home/ttt/PY080313/BEVFusion_Ascend310P/models/onnx/voxel_encoder.onnx

# OM输出目录
OUTPUT_DIR="$WORK_DIR/models/om"
# = /home/ttt/PY080313/BEVFusion_Ascend310P/models/om
```

## 注意事项

1. **Conda环境**: 确保系统上安装了anaconda3，而不是miniconda3
2. **环境名称**: 确保存在名为 `ascend-py3.7.10` 的conda环境
3. **ONNX模型**: 在导出OM之前，必须先导出对应的ONNX模型
4. **权限**: 所有脚本都已添加执行权限，可以直接运行

## 故障排查

### 问题1: conda.sh找不到
```bash
# 检查conda安装位置
which conda

# 查找conda.sh
ls ~/anaconda3/etc/profile.d/conda.sh
ls ~/miniconda3/etc/profile.d/conda.sh

# 根据实际位置修改脚本中的conda路径
```

### 问题2: ONNX模型找不到
```bash
# 检查ONNX模型是否存在
ls models/onnx/*.onnx

# 如果不存在，先导出ONNX模型
python src/export/export_voxel_encoder_onnx.py
```

### 问题3: OM导出失败
```bash
# 查看详细错误日志
atc --model=models/onnx/voxel_encoder.onnx --framework=5 --log=debug

# 检查ATC工具版本
atc --version
```

## 更新历史

- 2026-03-02: 修复conda路径和项目根目录路径计算问题
