"""一期实验运行目录：每次训练/评估生成唯一时间戳文件夹。

与二期 ``hitl_sessions/`` 平行——各自可单独跑；统一面板只调 ``api`` / ``hitl`` 对外接口，
并把一期 ``run_id`` 写进二期 ``meta.json`` 以便溯源。

目录::

    output/runs/run_YYYYMMDD_HHMMSS/
      meta.json
      uploads/          # 原始训练集/评估集（面板上传或 scenario 快照）
        forecast.csv / actual.csv
        forecast_eval.csv / actual_eval.csv
        growth_para.csv
      01_train/         # best/last、训练曲线
      02_eval/          # PPO vs 固定规则
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .paths import ROOT, mkdir

RUNS_ROOT = ROOT / "output" / "runs"

STAGE_DIRS = (
    "uploads",
    "01_train",
    "02_eval",
)


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def create_run(
    run_id: str | None = None,
    *,
    note: str = "",
    scenario: dict[str, Any] | None = None,
) -> Path:
    """创建一期运行骨架并写 ``meta.json``。"""
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    rid = run_id or new_run_id()
    root = RUNS_ROOT / rid
    # 同一秒多次创建时追加序号，避免撞目录
    if root.exists() and run_id is None:
        n = 1
        while (RUNS_ROOT / f"{rid}_{n}").exists():
            n += 1
        rid = f"{rid}_{n}"
        root = RUNS_ROOT / rid
    root = Path(mkdir(root))
    for rel in STAGE_DIRS:
        mkdir(root / rel)
    meta = {
        "run_id": rid,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": note,
        "kind": "phase1_train_eval",
        "stages": {
            "uploads": "uploads",
            "train": "01_train",
            "eval": "02_eval",
        },
        "scenario": scenario or {},
    }
    (root / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "uploads" / "README.txt").write_text(
        "原始数据表（输入，不含 KPI）：\n"
        "- forecast.csv / actual.csv           训练集气象\n"
        "- forecast_eval.csv / actual_eval.csv 评估集气象\n"
        "- growth_para.csv                    生育期与水深约束\n"
        "可由面板上传，或 pipeline 时从 scenario 自动快照。\n"
        "\n"
        "训评产物与指标请看同级目录：\n"
        "- 01_train/training_log_*.csv        学习曲线（episode/return/action_count）\n"
        "- 01_train/ppo_rice_best.pth         正式评估/审阅用权重\n"
        "- 02_eval/compare_summary.json       多年均值 KPI 与 Δ%\n"
        "- 02_eval/yearly_metrics_*.csv       分年 KPI\n"
        "- 02_eval/policy_output_*.csv        整季逐日水深/动作\n"
        "详见 docs/场景与输出.md §2。\n",
        encoding="utf-8",
    )
    return root


def latest_run(*, require_best: bool = True) -> Optional[Path]:
    """最近一期 run。默认要求 ``01_train/ppo_rice_best.pth`` 存在（跳过空壳/冒烟目录）。"""
    rows = list_runs(require_best=require_best, limit=1)
    if not rows:
        return None
    return Path(rows[0]["path"])


def list_runs(*, require_best: bool = True, limit: int = 40) -> list[dict]:
    """列出 ``output/runs/run_*``（新→旧）。默认只含有 ``ppo_rice_best.pth`` 的 run。"""
    if not RUNS_ROOT.exists():
        return []
    existing = sorted(
        [d for d in RUNS_ROOT.iterdir() if d.is_dir() and (d / "meta.json").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    out: list[dict] = []
    for d in existing:
        has_best = (d / "01_train" / "ppo_rice_best.pth").is_file()
        if require_best and not has_best:
            continue
        meta: dict = {}
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except Exception:
            pass
        out.append({
            "run_id": d.name,
            "path": str(d),
            "has_best": has_best,
            "has_eval": (d / "02_eval" / "compare_summary.json").is_file()
            or (d / "02_eval" / "policy_compare_daily.csv").is_file(),
            "note": meta.get("note", ""),
            "created_at": meta.get("created_at"),
        })
        if len(out) >= max(1, int(limit)):
            break
    return out


def resolve_run(work: str | Path | None = None, *, create: bool = False) -> Path | None:
    """解析一期运行根。``work is None`` 且 create → 新建；否则返回最近一次或 None。

    ``work`` 可为绝对路径、``output/runs/run_*``，或仅 ``run_*`` 名。
    """
    if work is not None:
        p = Path(work)
        if not p.is_absolute():
            cand = ROOT / p
            if cand.exists():
                p = cand
            elif (RUNS_ROOT / p.name).exists():
                p = RUNS_ROOT / p.name
            else:
                p = cand
        if create:
            mkdir(p)
        elif not p.exists():
            raise FileNotFoundError(f"一期 run 不存在: {work} → {p}")
        return p
    if create:
        return create_run()
    return latest_run()


class RunPaths:
    """一期某次运行的路径快捷方式。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.uploads = self.root / "uploads"
        self.train = self.root / "01_train"
        self.eval = self.root / "02_eval"

    @property
    def best_model(self) -> Path:
        return self.train / "ppo_rice_best.pth"

    @property
    def compare_daily(self) -> Path:
        return self.eval / "policy_compare_daily.csv"

    @property
    def compare_summary(self) -> Path:
        return self.eval / "compare_summary.json"

    def ensure(self) -> "RunPaths":
        for d in (self.uploads, self.train, self.eval):
            mkdir(d)
        return self

    def update_meta(self, **kwargs: Any) -> None:
        meta_path = self.root / "meta.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(kwargs)
        meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_upload(self, src: str | Path, name: str) -> Path:
        """面板上传文件落入 ``uploads/``。"""
        self.uploads.mkdir(parents=True, exist_ok=True)
        dst = self.uploads / name
        shutil.copy2(src, dst)
        return dst

    def snapshot_scenario_tables(self, scenario) -> list[str]:
        """把 scenario 训练/评估表快照进 ``uploads/``。"""
        self.uploads.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for name, attr in (
            ("forecast.csv", "forecast_rel"),
            ("actual.csv", "actual_rel"),
            ("forecast_eval.csv", "forecast_eval_rel"),
            ("actual_eval.csv", "actual_eval_rel"),
            ("growth_para.csv", "growth_para_rel"),
        ):
            rel = getattr(scenario, attr, None)
            if not rel:
                continue
            src = Path(rel)
            if not src.is_absolute():
                src = ROOT / src
            if src.exists() and src.is_file():
                shutil.copy2(src, self.uploads / name)
                copied.append(name)
        return copied


def ensure_pipeline_dirs(*, note: str = "pipeline") -> tuple[Path, Path, Path]:
    """新建一期 run，返回 ``(01_train, 02_eval, run_root)``。"""
    root = create_run(note=note)
    rp = RunPaths(root).ensure()
    return rp.train, rp.eval, root
