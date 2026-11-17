"""工程根 = ``PPO_Rice_Env/``。所有业务 import 必须解析到本目录内。

约定
----
- ``sys.path`` 顶端只挂本仓库根（本文件的 ``ROOT``）
- 训练代码用 ``from train.xxx import ...``（包导入），禁止依赖把 ``train/`` 当扁平路径
- 不要 ``from PPO_RL_brain import ...``：会撞到工作区外同名旧文件
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ensure_repo_path() -> Path:
    """把 ``PPO_Rice_Env`` 插到 ``sys.path[0]``，并去掉易冲突的路径。

    - 去掉 ``ROOT/train``（避免 flat ``import PPO_RL_brain``）
    - 若工作区父目录也在 path 上，把它挪到本 ROOT 之后（外层 ``hitl/`` 等不得抢解析）
    """
    root = str(ROOT)
    train = str(ROOT / "train")
    parent = str(ROOT.parent)

    while train in sys.path:
        sys.path.remove(train)

    # 父目录（JMM616-5000）若存在，先摘掉再挂到后面，避免外层 hitl / 旧脚本抢包名
    parent_was = parent in sys.path
    while parent in sys.path:
        sys.path.remove(parent)

    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)

    if parent_was and parent != root:
        sys.path.append(parent)

    return ROOT


def module_file(mod) -> Path:
    """返回已加载模块的源文件路径（用于自检）。"""
    f = getattr(mod, "__file__", None)
    if not f:
        raise RuntimeError(f"模块无 __file__: {mod!r}")
    return Path(f).resolve()


def assert_modules_local(*mods) -> list[str]:
    """断言模块文件都在 ``ROOT`` 下；返回相对路径列表。失败则抛 ``RuntimeError``。"""
    ensure_repo_path()
    root = ROOT.resolve()
    rels: list[str] = []
    for m in mods:
        p = module_file(m)
        try:
            rel = p.relative_to(root)
        except ValueError as e:
            raise RuntimeError(
                f"模块 {getattr(m, '__name__', m)} 不在 PPO_Rice_Env 内:\n"
                f"  解析到: {p}\n"
                f"  期望根: {root}\n"
                f"请确认从本目录运行，且未残留工作区根的同名 .py"
            ) from e
        rels.append(rel.as_posix())
    return rels


def mkdir(path: str | Path) -> str:
    """目录不存在则创建。相对路径相对 ROOT（不受 chdir 影响）。返回绝对路径字符串。"""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    abs_dir = os.path.abspath(str(p))
    os.makedirs(abs_dir, exist_ok=True)
    return abs_dir
