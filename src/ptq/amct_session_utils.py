"""
amct_session_utils.py
================================================================================
统一封装"带 AMCT 自定义 op 的 ONNX Runtime Session 创建"逻辑。

背景：
    AMCT 向 modified_model.onnx 插入了属于 'amct.customop' 域的校准算子
    （IFMR、ActMaxCalib 等）。原生 OnnxRuntime 不认识这些算子，必须先通过
    sess_opts.register_custom_ops_library() 注册 AMCT 提供的 .so 共享库，
    再创建 InferenceSession，否则报错：
        "No opset import for domain 'amct.customop'"

    关键：必须找到真正导出了 `RegisterCustomOps` 符号的 .so，
    其余辅助库（如 libamct_ncx.so）不含该符号，注册时会报
        "undefined symbol: RegisterCustomOps"

查找策略：
    遍历 amct_onnx 包目录下所有 *.so，用 ctypes.CDLL 尝试加载并检查
    `RegisterCustomOps` 符号是否存在，找到第一个有效库即返回。
    也可通过环境变量 AMCT_CUSTOM_OP_LIB 手动指定，跳过自动搜索。
"""

import os
import glob
import ctypes
import onnxruntime as ort


def _has_register_custom_ops(so_path: str) -> bool:
    """
    检查 so_path 是否导出了 ORT 要求的 RegisterCustomOps 符号。
    使用 ctypes 加载，不会污染 ORT 状态。
    """
    try:
        lib = ctypes.CDLL(so_path)
        return hasattr(lib, 'RegisterCustomOps') and lib.RegisterCustomOps is not None
    except OSError:
        return False


def _find_amct_custom_op_lib() -> str:
    """
    自动定位 AMCT 自定义 op 共享库（必须导出 RegisterCustomOps）。

    查找顺序：
      1. 环境变量 AMCT_CUSTOM_OP_LIB（用户手动指定）
      2. amct_onnx 包目录下优先候选名（精确匹配）
      3. amct_onnx 包目录下所有 *.so（glob，按名称排序）
         → 每个都用 ctypes 验证 RegisterCustomOps 是否存在

    返回有效 .so 的绝对路径，找不到时抛出 FileNotFoundError。
    """
    # 1. 环境变量优先
    env_path = None #os.environ.get("AMCT_CUSTOM_OP_LIB", "")
    if env_path:
        if not os.path.isfile(env_path):
            raise FileNotFoundError(
                f"AMCT_CUSTOM_OP_LIB 指定的路径不存在: {env_path}"
            )
        if not _has_register_custom_ops(env_path):
            raise RuntimeError(
                f"AMCT_CUSTOM_OP_LIB 指定的库未导出 RegisterCustomOps: {env_path}\n"
                "请确认路径正确。"
            )
        return os.path.abspath(env_path)

    # 2. 定位 amct_onnx 包目录
    try:
        import amct_onnx
        pkg_dir = os.path.dirname(os.path.abspath(amct_onnx.__file__))
    except ImportError:
        pkg_dir = None

    search_roots = []
    if pkg_dir:
        search_roots.append(pkg_dir)
        search_roots.append(os.path.dirname(pkg_dir))   # site-packages 根

    # 3. 优先候选名（精确匹配 + 验证符号）
    priority_names = [
        "libamct_onnx_ops.so",
        "libamct_onnxruntime_ops.so",
        "libamct_custom_ops.so",
        "libamct_ops.so",
        "libamct_ort_ops.so",
    ]
    for root in search_roots:
        for name in priority_names:
            for dirpath, _, filenames in os.walk(root):
                if name in filenames:
                    candidate = os.path.join(dirpath, name)
                    if _has_register_custom_ops(candidate):
                        return candidate

    # 4. glob 兜底：遍历所有 *.so，验证符号
    checked = []
    for root in search_roots:
        for candidate in sorted(
            glob.glob(os.path.join(root, "**", "*.so"), recursive=True)
        ):
            checked.append(candidate)
            if _has_register_custom_ops(candidate):
                return candidate

    checked_str = "\n    ".join(checked) if checked else "（未找到任何 .so）"
    raise FileNotFoundError(
        "未能在 amct_onnx 包目录下找到导出 RegisterCustomOps 的共享库。\n"
        f"已检查以下文件（均不含 RegisterCustomOps）：\n    {checked_str}\n\n"
        "解决方法：\n"
        "  export AMCT_CUSTOM_OP_LIB=/path/to/libamct_onnxruntime_ops.so\n\n"
        "可用以下命令手动搜索：\n"
        "  find $CONDA_PREFIX -name '*.so' | xargs -I{} sh -c "
        "'nm -D {} 2>/dev/null | grep -q RegisterCustomOps && echo {}'"
    )


def make_amct_session(modified_onnx: str,
                      providers=None) -> ort.InferenceSession:
    """
    创建已注册 AMCT 自定义 op 的 OnnxRuntime InferenceSession。

    参数：
        modified_onnx : AMCT quantize_model 生成的 modified_model.onnx 路径
        providers     : ORT provider 列表，默认 ['CPUExecutionProvider']

    返回：
        ort.InferenceSession
    """
    if providers is None:
        providers = ['CPUExecutionProvider']

    lib_path = _find_amct_custom_op_lib()
    print(f"  [amct_session_utils] 注册自定义 op 库: {lib_path}")

    sess_opts = ort.SessionOptions()
    # 必须关闭图优化，否则 ORT 可能折叠掉 AMCT 校准算子
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # 注册 AMCT 自定义域算子
    sess_opts.register_custom_ops_library(lib_path)

    session = ort.InferenceSession(
        modified_onnx,
        sess_options=sess_opts,
        providers=providers,
    )
    return session
