"""二期会话目录：一次 HITL 实验 = ``output/hitl_sessions/<sess_id>/``。

与 Web 看板、CLI、``api.hitl_*`` **共用同一棵目录树**。

``seed`` 时把一期 ``runs/.../{uploads,01_train,02_eval}`` 整拷进来，单个 sess 即可离线分析。

``uploads/``：面板上传或情景快照的**原始训练集 / 评估集**（与 ``_workdir`` 运行拷贝区分）。
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from common.paths import ROOT, mkdir

SESSIONS_ROOT = ROOT / "output" / "hitl_sessions"

STAGE_DIRS = (
    "uploads",
    "01_train",
    "02_eval",
    "03_review",
    "03_review/original_trajectories",
    "03_review/current",
    "04_bc_export",
    "05_bc",
    "06_compare",
    "06_compare/rule",
    "06_compare/original_ppo",
    "06_compare/hitl_ppo",
    "07_metrics",
    "_workdir",
)


def new_session_id(prefix: str = "sess") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def create_session(session_id: str | None = None, *, note: str = "") -> Path:
    """新建会话骨架 + ``meta.json``。返回会话根路径。"""
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    sid = session_id or new_session_id()
    root = SESSIONS_ROOT / sid
    if root.exists() and session_id is None:
        n = 1
        while (SESSIONS_ROOT / f"{sid}_{n}").exists():
            n += 1
        sid = f"{sid}_{n}"
        root = SESSIONS_ROOT / sid
    root = Path(mkdir(root))
    for rel in STAGE_DIRS:
        mkdir(root / rel)
    meta = {
        "session_id": sid,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": note,
        "stages": {
            "uploads": "uploads",
            "train": "01_train",
            "eval": "02_eval",
            "review": "03_review",
            "bc_export": "04_bc_export",
            "bc": "05_bc",
            "compare": "06_compare",
            "metrics": "07_metrics",
            "workdir": "_workdir",
        },
        "policy": {
            "review_freezes_ppo": True,
            "bc_after_all_years": True,
            "original_trajectory_immutable": True,
            "embeds_phase1_train_eval": True,
            "shared_with_web": True,
        },
    }
    (root / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "07_metrics" / "README.txt").write_text(
        "实验结果写入：\n"
        "- absorption.json\n"
        "- task_compare.json\n"
        "- effort.json\n",
        encoding="utf-8",
    )
    (root / "uploads" / "README.txt").write_text(
        "原始数据表（面板上传或 scenario 快照），勿与 _workdir 混淆：\n"
        "- forecast.csv / actual.csv          训练集\n"
        "- forecast_eval.csv / actual_eval.csv 评估/审阅集\n"
        "- growth_para.csv\n"
        "一期 run 的 uploads/ 会在 seed 时整拷到此处。\n",
        encoding="utf-8",
    )
    return root


def resolve_session(work: str | Path | None = None, *, create: bool = True) -> Path:
    """解析会话根。``work`` 给定则用之；否则最近会话或（``create=True`` 时）新建。"""
    if work is not None:
        p = Path(work)
        if not p.is_absolute():
            p = ROOT / p
        mkdir(p)
        return p

    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    found = latest_session()
    if found is not None:
        return found
    if not create:
        raise FileNotFoundError(
            f"无 HITL 会话（{SESSIONS_ROOT}）。请先 hitl seed / board，或指定 --work"
        )
    return create_session()


def latest_session() -> Optional[Path]:
    if not SESSIONS_ROOT.exists():
        return None
    existing = sorted(
        [d for d in SESSIONS_ROOT.iterdir() if d.is_dir() and (d / "meta.json").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return existing[0] if existing else None


def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """列出 ``hitl_sessions`` 下各会话（按修改时间新→旧）。"""
    if not SESSIONS_ROOT.exists():
        return []
    rows: list[dict[str, Any]] = []
    dirs = sorted(
        [d for d in SESSIONS_ROOT.iterdir() if d.is_dir() and (d / "meta.json").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for d in dirs[: max(1, int(limit))]:
        meta: dict[str, Any] = {}
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except Exception:
            pass
        sp = SessionPaths(d)
        rows.append({
            "session_id": d.name,
            "path": str(d),
            "note": meta.get("note", ""),
            "run_id": meta.get("run_id"),
            "created_at": meta.get("created_at"),
            "has_review": sp.review_json.exists() or sp.review_csv.exists(),
            "has_hitl": sp.hitl_model.exists(),
            "has_compare": (sp.compare / "three_way.json").exists(),
        })
    return rows


class SessionPaths:
    """会话内阶段路径。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.uploads = self.root / "uploads"
        self.train = self.root / "01_train"
        self.phase1_eval = self.root / "02_eval"
        self.review = self.root / "03_review"
        self.original_traj = self.review / "original_trajectories"
        self.current = self.review / "current"
        self.history = self.review / "history"
        self.bc_export = self.root / "04_bc_export"
        self.bc = self.root / "05_bc"
        self.compare = self.root / "06_compare"
        self.metrics = self.root / "07_metrics"
        self.workdir = self.root / "_workdir"

    @property
    def model(self) -> Path:
        return self.train / "ppo_rice_best.pth"

    @property
    def review_json(self) -> Path:
        return self.current / "review.json"

    @property
    def review_csv(self) -> Path:
        return self.current / "review.csv"

    @property
    def edits_log(self) -> Path:
        return self.review / "edits_log.jsonl"

    @property
    def labels(self) -> Path:
        return self.bc_export / "labels.json"

    @property
    def hitl_model(self) -> Path:
        return self.bc / "ppo_hitl.pth"

    @property
    def compare_daily(self) -> Path:
        return self.phase1_eval / "policy_compare_daily.csv"

    def ensure(self) -> "SessionPaths":
        for d in (
            self.uploads, self.train, self.phase1_eval, self.original_traj, self.current,
            self.bc_export, self.bc, self.compare, self.metrics, self.workdir,
        ):
            mkdir(d)
        # 目录准备不能删除历史审阅证据。
        for arm in ("rule", "original_ppo", "hitl_ppo"):
            mkdir(self.compare / arm)
        return self

    def save_upload(self, src: str | Path, name: str) -> Path:
        """面板上传原始表 → ``uploads/<name>``。"""
        self.uploads.mkdir(parents=True, exist_ok=True)
        dst = self.uploads / name
        shutil.copy2(src, dst)
        return dst

    def snapshot_scenario_tables(self, scenario) -> list[str]:
        """把 scenario 指向的训练/评估表快照进 ``uploads/``（论文溯源）。"""
        self.uploads.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        pairs = (
            ("forecast.csv", "forecast_rel"),
            ("actual.csv", "actual_rel"),
            ("forecast_eval.csv", "forecast_eval_rel"),
            ("actual_eval.csv", "actual_eval_rel"),
            ("growth_para.csv", "growth_para_rel"),
        )
        for name, attr in pairs:
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

    def update_meta(self, **kwargs: Any) -> None:
        meta_path = self.root / "meta.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(kwargs)
        meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_edit(self, record: dict[str, Any]) -> None:
        self.edits_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.edits_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def ingest_phase1_run(self, run_root: Path) -> dict[str, Any]:
        """整拷一期 ``uploads`` / ``01_train`` / ``02_eval`` 进本会话。"""
        from common.run_store import RunPaths

        rp = RunPaths(run_root)
        report: dict[str, Any] = {"from_run": str(run_root), "copied_dirs": [], "missing": []}
        for src, dst, label in (
            (rp.uploads, self.uploads, "uploads"),
            (rp.train, self.train, "01_train"),
            (rp.eval, self.phase1_eval, "02_eval"),
        ):
            if not src.exists():
                report["missing"].append(f"{label}: {src}")
                continue
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            report["copied_dirs"].append(label)
        mkdir(self.original_traj)
        mkdir(self.uploads)
        return report

    def save_original_trajectories(self, df, *, force: bool = False) -> Path:
        """按年锁定原始轨迹（默认不覆盖）。"""
        self.original_traj.mkdir(parents=True, exist_ok=True)
        marker = self.original_traj / "_LOCKED.txt"
        if marker.exists() and not force:
            return self.original_traj
        if "year" not in df.columns:
            df = df.copy()
            df["year"] = df["date"].astype(str).map(lambda d: str(d).split("/")[0].split("-")[0])
        for y, g in df.groupby("year"):
            out = self.original_traj / f"year_{y}.json"
            if out.exists() and not force:
                continue
            g.to_json(out, orient="records", force_ascii=False)
        all_path = self.original_traj / "all_years.json"
        if force or not all_path.exists():
            df.to_json(all_path, orient="records", force_ascii=False)
        marker.write_text(
            "原始轨迹已锁定。审阅请改 03_review/current/。\n",
            encoding="utf-8",
        )
        return self.original_traj

    def snapshot_review_history(self, tag: str | None = None) -> Path | None:
        """已废弃：实时态只保留 current + original，不再写 history。"""
        return None

    def clear_review_history(self) -> None:
        """删除无用的 history/ 目录（若存在）。"""
        if self.history.exists():
            shutil.rmtree(self.history, ignore_errors=True)
