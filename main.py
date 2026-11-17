#!/usr/bin/env python
"""PPO_Rice_Env 正式流水线入口（分模块逐步跑）。

========================================================================
怎么用
========================================================================
1. 工作目录必须是本文件夹::

       cd PPO_Rice_Env
       python main.py

2. 只改下面三块：
   - ``PATHS``   ：上一步产物路径（跑完 train / seed 后回填）
   - ``STEP``    ：当前要跑的步骤编号（见下方流水线表）
   - （可选）超参：一期轮数、BC epochs 等

3. 底层一律 ``api.dispatch`` —— 与 ``run.py`` / Web 看板调用的是同一套模块，
   不要再去改外层 ``JMM616-5000/`` 旧脚本。

========================================================================
流水线总览（正式，非冒烟）
========================================================================

【一期】产物目录 ``output/runs/run_<时间戳>/``

  1.1  show_config   校验 configs + data 表是否齐全
  1.2  train         训练 PPO → ``01_train/ppo_rice_best.pth``（正式默认 1500 轮）
  1.3  evaluate      评估 PPO vs 固定规则 → ``02_eval/``（默认全部验证年）
  1.0  pipeline      一键 = 1.1→1.2→1.3（正式长跑，一次出完整 run）

【二期】产物目录 ``output/hitl_sessions/sess_<时间戳>/``
       （依赖一期已有带 best 的 run）

  2.1  hitl_seed     整拷一期 uploads+01_train+02_eval 进会话
  2.2  hitl_review   冻结 PPO 滚审阅表 → ``03_review/current/``
  2.3  hitl_resync   专家改动作后重推环境 + 刷新后续建议
  2.4  hitl_bc       一致重算 + BC → ``04_bc_export/`` + ``05_bc/ppo_hitl.pth``
  2.5  hitl_compare  规则 / 原PPO / HITL 三方 → ``06_compare`` + ``07_metrics``

【Web 完整方案】与上面同一套 api 模块，面板点按顺序 = 1.2+1.3 → 2.1→…→2.5

  web  hitl_board    浏览器 http://127.0.0.1:5050

========================================================================
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from common.paths import ensure_repo_path  # noqa: E402

ensure_repo_path()

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  A. 路径台账 —— 跑完上一步后，把打印结果填到这里再跑下一步          ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# 一期 train / pipeline 结束后终端会打印 run / run_id，例如：
#   "run_id": "run_20261023_180512"
# 填到 RUN_REL（相对 PPO_Rice_Env）或直接改 MODEL_REL。
#
# 二期 seed 结束后会打印 session，例如：
#   "session": ".../output/hitl_sessions/sess_20261023_181020"
# 填到 SESSION_REL。

RUN_REL = ""
# 例: "output/runs/run_20261023_180512"
# 留空时：evaluate / seed 会尽量用「最近一次带 best 的 run」

MODEL_REL = ""
# 例: "output/runs/run_20261023_180512/01_train/ppo_rice_best.pth"
# 留空且 RUN_REL 已填 → 自动拼 01_train/ppo_rice_best.pth

SESSION_REL = ""
# 例: "output/hitl_sessions/sess_20261023_181020"
# 留空时：二期步骤用最近会话（须先 seed）；seed 本身会新建会话


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  B. 正式超参（不要改成冒烟小数）                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

EPISODES = 1500          # 一期训练轮数（正式 1000～2000；本默认 1500）
TRAJECTORY_NUM = 10      # 每多少局做一次 PPO update
MODEL_NAME = "ppo_rice"  # 权重文件名前缀 → ppo_rice_best.pth / _last.pth
EVAL_MAX_YEARS = None    # None = 全部验证年；仅调试可改 int
BC_EPOCHS = 40           # 二期行为克隆 epochs（与 hitl.config.BC_DEFAULT 对齐）
BOARD_HOST = "127.0.0.1"
BOARD_PORT = 5050


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  C. 当前步骤 —— 只改这一行（按序从 1.1 跑到 2.5，或直接 web）        ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# 一期分步:  "1.1" | "1.2" | "1.3" | "1.0"
# 二期分步:  "2.1" | "2.2" | "2.3" | "2.4" | "2.5"
# Web 全流程: "web"
#
STEP = "web"


# ---------------------------------------------------------------------------
# 步骤 → api 方法（与 api.METHODS / run.py / Web 同源）
# ---------------------------------------------------------------------------
_STEP_HELP = {
    "1.1": "一期·校验配置与数据表",
    "1.2": "一期·训练 PPO → output/runs/…/01_train/",
    "1.3": "一期·评估 PPO vs 规则 → …/02_eval/",
    "1.0": "一期·一键 pipeline（show_config→train→evaluate）",
    "2.1": "二期·seed：整拷一期 run → hitl_sessions/sess_*/",
    "2.2": "二期·review：冻结 PPO 生成审阅表",
    "2.3": "二期·resync：改动作后重推 + 刷新建议",
    "2.4": "二期·bc：一致重算 + 行为克隆微调",
    "2.5": "二期·compare：规则 / 原PPO / HITL 三方对比",
    "web": "Web 看板：与上述同一套模块，面板内按 训评→seed→…→compare",
}


def _resolve_model() -> str | None:
    if MODEL_REL.strip():
        return MODEL_REL.strip()
    if RUN_REL.strip():
        return f"{RUN_REL.strip().rstrip('/')}/01_train/{MODEL_NAME}_best.pth"
    return None


def _resolve_run() -> str | None:
    return RUN_REL.strip() or None


def _resolve_session() -> str | None:
    return SESSION_REL.strip() or None


def build_call(step: str) -> tuple[str, dict[str, Any]]:
    """把 STEP 编成 ``(api 方法名, kwargs)``。"""
    step = step.strip().lower()
    model = _resolve_model()
    run = _resolve_run()
    work = _resolve_session()

    if step == "1.1":
        return "show_config", {}

    if step == "1.2":
        # 新建 output/runs/run_<ts>/01_train/；结束后请把 run_id 填回 RUN_REL
        return "train", {
            "episodes": EPISODES,
            "trajectory_num": TRAJECTORY_NUM,
            "model_name": MODEL_NAME,
            "verbose": True,
        }

    if step == "1.3":
        # 优先写入同一 run 的 02_eval（RUN_REL 已填时）
        kw: dict[str, Any] = {
            "model": model,
            "only": "both",
            "max_years": EVAL_MAX_YEARS,
        }
        if run:
            kw["run"] = run
        if not model:
            raise SystemExit(
                "1.3 evaluate 需要模型：请先跑 1.2，并把 RUN_REL 或 MODEL_REL 填上。"
            )
        return "evaluate", kw

    if step == "1.0":
        return "pipeline", {
            "episodes": EPISODES,
            "trajectory_num": TRAJECTORY_NUM,
            "model_name": MODEL_NAME,
            "max_years": EVAL_MAX_YEARS,
            "verbose": True,
            "note": "main.pipeline",
        }

    if step == "2.1":
        # from_run 省略 → api/hitl 自动选「最近带 best 的一期 run」
        return "hitl_seed", {
            "from_run": run,
            "work": work,  # 通常留空，新建 sess_*
        }

    if step == "2.2":
        return "hitl_review", {
            "work": work,
            "model_path": model,
            "max_years": EVAL_MAX_YEARS,
            "from_run": run,
        }

    if step == "2.3":
        return "hitl_resync", {
            "work": work,
            "model_path": model,
        }

    if step == "2.4":
        return "hitl_bc", {
            "work": work,
            "model_path": model,
            "epochs": BC_EPOCHS,
        }

    if step == "2.5":
        return "hitl_compare", {
            "work": work,
            "max_years": EVAL_MAX_YEARS,
        }

    if step == "web":
        # 看板内按钮顺序对应：一期训评 → seed → review → 改/确认
        # → resync → BC → compare；业务函数与上面 STEP 相同
        return "hitl_board", {
            "host": BOARD_HOST,
            "port": BOARD_PORT,
            "debug": False,
        }

    raise SystemExit(
        "未知 STEP=%r。可选: %s" % (step, ", ".join(_STEP_HELP.keys()))
    )


def _print_next_hint(step: str, out: Any) -> None:
    """跑完后提示如何回填 PATHS。"""
    if not isinstance(out, dict):
        return
    if step in ("1.2", "1.0") and out.get("run_id"):
        print()
        print("── 请回填 main.py 路径台账 ──")
        print('RUN_REL = "output/runs/%s"' % out["run_id"])
        print(
            'MODEL_REL = "output/runs/%s/01_train/%s_best.pth"'
            % (out["run_id"], MODEL_NAME)
        )
        print("然后把 STEP 改为 \"1.3\"（若刚跑完 1.2）或 \"2.1\"（若刚跑完 1.0）")
    if step == "2.1" and out.get("session"):
        sess = Path(str(out["session"]))
        try:
            rel = sess.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            rel = str(sess)
        print()
        print("── 请回填 main.py 路径台账 ──")
        print('SESSION_REL = "%s"' % rel)
        print("然后把 STEP 改为 \"2.2\"")
    if step == "web":
        print()
        print("浏览器打开: http://%s:%s" % (BOARD_HOST, BOARD_PORT))
        print(
            "面板顺序（与 STEP 1.2/1.3→2.1→2.2→2.3→2.4→2.5 同源模块）:\n"
            "  一期训练+评估 → Seed → 生成审阅 → 确认/修改 → Resync → BC → 三方对比"
        )


def main() -> int:
    from api import dispatch, list_methods

    method, kw = build_call(STEP)
    # 去掉值为 None 的参数，避免覆盖函数默认（如 from_run=None）
    kw = {k: v for k, v in kw.items() if v is not None}

    if method not in list_methods():
        print("未知 METHOD=%r；可选: %s" % (method, ", ".join(list_methods())))
        return 2

    print("=" * 64)
    print("ROOT   =", ROOT)
    print("STEP   =", STEP, "—", _STEP_HELP.get(STEP, ""))
    print("METHOD =", method)
    print("KW     =", kw)
    print("PATHS  =", {
        "RUN_REL": RUN_REL or "(空→自动/新建)",
        "MODEL_REL": MODEL_REL or _resolve_model() or "(空)",
        "SESSION_REL": SESSION_REL or "(空→最近会话/新建)",
    })
    print("=" * 64)

    if STEP in ("1.2", "1.0"):
        print("正式训练 episodes=%s（耗时较长，请耐心等待）" % EPISODES)

    out = dispatch(method, **kw)

    if hasattr(out, "to_dict"):
        # DataFrame（review / resync）
        print("ok rows=%s" % (len(out),))
        if "year" in getattr(out, "columns", []):
            print("years=%s" % (sorted(out["year"].astype(str).unique().tolist()),))
        _print_next_hint(STEP, {})
        return 0

    if isinstance(out, dict):
        slim = {k: v for k, v in out.items() if k not in ("config", "arms", "scenario")}
        try:
            print(json.dumps(slim, ensure_ascii=False, indent=2, default=str))
        except TypeError:
            print(slim)
        _print_next_hint(STEP, out)
        return 0 if out.get("ok", True) is not False else 1

    print(out)
    _print_next_hint(STEP, {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
