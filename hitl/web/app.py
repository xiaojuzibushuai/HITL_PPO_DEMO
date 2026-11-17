"""双流程看板：智能体训练（runs）+ 专家微调（hitl_sessions）。

启动：``python run.py hitl board`` 或 ``main.py`` STEP=web → http://127.0.0.1:5050
"""
from __future__ import annotations

import io
import json
import re
import sys
import threading
import traceback
import zipfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, url_for

from common.paths import ensure_repo_path
from common.execution import allowed_actions, fixed_rule_action_mm
from common.run_store import RunPaths, create_run, list_runs, resolve_run

ensure_repo_path()

from ..compare_eval import compare_three
from ..compare_viz import (
    build_stage_compare,
    build_stage_year_heatmap,
    build_yearly_pairs,
    compute_absorption_probs,
    load_expert_edit_dates,
    resolve_growth_para,
    season_rows_enriched,
    year_role,
)
from ..config import ACTION_SPACE, BC_DEFAULT, BCConfig, describe_bc_rationale
from ..metrics_report import build_bc_dashboard, compute_absorption
from ..pipeline import bc, resync, review, seed
from ..session import SESSIONS_ROOT, SessionPaths, create_session, list_sessions

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)
app.secret_key = "ppo-rice-hitl-board"
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

_RUN: RunPaths | None = None
_SESSION: SessionPaths | None = None
_TRAIN_LOCK = threading.Lock()
_TRAIN_STATE: dict[str, Any] = {"busy": False, "run_id": None, "error": None}


# ---------------------------------------------------------------------------
# Bind helpers
# ---------------------------------------------------------------------------

def _bind_run(root: Path) -> RunPaths:
    global _RUN
    _RUN = RunPaths(root).ensure()
    return _RUN


def _bind_session(root: Path) -> SessionPaths:
    global _SESSION
    _SESSION = SessionPaths(root).ensure()
    return _SESSION


def _rp() -> RunPaths:
    if _RUN is None:
        raise RuntimeError("无当前训练运行：请新建或选择历史 run")
    return _RUN


def _require_run(data: dict | None = None) -> RunPaths:
    """优先用请求里的 ``run_id`` 绑定；否则用进程内当前 run。

    看板刷新后前端仍有 ``runState.run_id``，但 Flask 进程可能已丢 ``_RUN``，
    因此写 uploads / 校验等接口应带上 run_id。
    """
    data = data or {}
    rid = (data.get("run_id") or data.get("run") or "").strip()
    if rid:
        root = resolve_run(_normalize_run_ref(rid), create=False)
        if root is None:
            raise RuntimeError(f"找不到 run: {rid}")
        return _bind_run(root)
    return _rp()


def _sp() -> SessionPaths:
    if _SESSION is None:
        raise RuntimeError("无当前微调会话：请选 run 进入专家微调，或加载历史会话")
    return _SESSION


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_run_ref(ref: str) -> str:
    ref = ref.strip()
    if not ref:
        return ref
    p = Path(ref)
    if not p.is_absolute() and "/" not in ref.replace("\\", "/"):
        return f"output/runs/{ref}"
    return ref


def _scenario_for_run(rp: RunPaths):
    from common.scenario import get_scenario

    base = get_scenario()
    up = rp.uploads
    kw: dict[str, str] = {}
    mapping = {
        "forecast_rel": "forecast.csv",
        "actual_rel": "actual.csv",
        "forecast_eval_rel": "forecast_eval.csv",
        "actual_eval_rel": "actual_eval.csv",
        "growth_para_rel": "growth_para.csv",
    }
    for attr, name in mapping.items():
        p = up / name
        if p.exists():
            kw[attr] = str(p.resolve())
    return replace(base, **kw) if kw else base


# 看板文件夹摘要：文件名 → 用途（含「看哪些指标/列」）
_FILE_ROLE_EXACT: dict[str, str] = {
    "forecast.csv": "训练预报气象（日）· 输入表，无 KPI 列",
    "actual.csv": "训练实测气象（日）· 输入表，无 KPI 列",
    "forecast_eval.csv": "评估预报气象 · 评估输入",
    "actual_eval.csv": "评估实测气象 · 评估输入",
    "growth_para.csv": "生育期 / hmin·hmax·Hp·Hc·λ 等约束参数",
    "README.txt": "uploads 目录说明",
    "ppo_rice_best.pth": "全局最佳权重 · 评估与人工审阅冻结源（优先用）",
    "ppo_rice_last.pth": "训练终点权重 · 完整训完；正式评估请用 best",
    "ppo_rice_best_config.json": "best 旁路网络/超参配置",
    "ppo_rice_last_config.json": "last 旁路网络/超参配置",
    "compare_summary.json": (
        "PPO vs 规则多年均值 · 看 ppo_*/rule_* 与 delta_pct："
        "rain_utilization / irrigation_total / drainage_total / managed_water / yield_factor"
    ),
    "policy_compare_daily.csv": "逐日动作对照 · 看 date/a_ppo/a_rule/disagree（审阅优先分歧日）",
    "policy_output_PPO.csv": (
        "PPO 整季逐日 · 看 date/h_before/h_after/a_mm/a_exec/m/d/reward_total（单年轨迹）"
    ),
    "policy_output_RULE.csv": "规则整季逐日 · 列同 policy_output_PPO",
    "yearly_metrics_PPO.csv": (
        "PPO 分年 KPI · 看 year + rain_utilization / irrigation_total / drainage_total / "
        "managed_water / yield_factor / irrigation_times / drainage_times"
    ),
    "yearly_metrics_RULE.csv": "规则分年 KPI · 列同 yearly_metrics_PPO",
    "run_manifest.json": "本次评估写入了哪些文件的清单",
    "review.csv": (
        "人工审阅表 · 改 a_expert/confirmed；读 h_*/阈值/a_model/a_exec；"
        "只显示决策时可用的 P_cumu7 预报，不展示实测 P_1_a"
    ),
    "review.json": "人工审阅表 JSON（与 review.csv 同步）",
    "labels.json": "行为克隆监督标签（确认动作导出）",
    "ppo_hitl.pth": "行为克隆后 HITL 权重",
    "hitl_result.json": "BC 训练摘要 · 看 n_labels / n_edited / loss",
    "three_way.json": "三方对比原始汇总（规则 / 原 PPO / HITL）",
    "task_compare.json": (
        "三方任务指标 · 看各臂均值与 Δ%；主表同五核心 KPI"
    ),
    "absorption.json": "专家改动吸收率 · edit_reproduce_rate / confirm_hold_rate_policy",
    "expert_rationale_audit.json": (
        "结构化改因事后核查 · wait_forecast_rain / allow_low_water 分项；"
        "不合成等雨成功率；含 P_cumu7 时窗说明"
    ),
}


def _file_role(name: str) -> str:
    base = name.split("/")[-1].split("\\")[-1]
    if base in _FILE_ROLE_EXACT:
        return _FILE_ROLE_EXACT[base]
    if re.match(r"training_log_.*\.csv$", base):
        return "逐局训练日志 · 看 episode / return / mean_return / action_count（学习曲线）"
    if re.match(r"training_loss_.*\.csv$", base) and "detail" not in base:
        return "逐局训练损失 · 看 episode / policy_loss / value_loss / entropy_loss"
    if "training_loss" in base and "detail" in base:
        return "训练损失明细（每次 PPO update）"
    if re.match(r"training_report_.*\.json$", base):
        return "训练报告 · training_summary / convergence_analysis"
    if re.match(r"training_results_.*\.png$", base):
        return "训练曲线离线导出图（回报/动作）"
    if name.endswith("/") or name.endswith("\\"):
        return "对比臂目录 · 内含该臂 yearly_metrics / policy_output"
    if base.endswith(".pth"):
        return "模型权重文件"
    if base.endswith(".csv"):
        return "表格数据（见 docs/场景与输出.md）"
    if base.endswith(".json"):
        return "结构化元数据 / 指标 JSON"
    if base.endswith(".png"):
        return "图像产物"
    return "会话产物（未登记用途，见 docs/场景与输出.md）"


def _list_dir_files(folder: Path, patterns: tuple[str, ...] = ("*",), limit: int = 40) -> list[str]:
    if not folder.exists():
        return []
    names: list[str] = []
    for pat in patterns:
        for p in sorted(folder.glob(pat), key=lambda x: x.name):
            if p.is_file():
                names.append(p.name)
            if len(names) >= limit:
                return names
    return names


def _annotate_files(names: list[str]) -> list[dict[str, str]]:
    return [{"name": n, "role": _file_role(n)} for n in names]


def _run_folder_summary(rp: RunPaths) -> dict[str, Any]:
    """训练会话目录摘要（含文件用途 uploads_detail/train_detail/eval_detail），供面板回放。

    用途文案见 ``_FILE_ROLE_EXACT``；指标对照见 ``docs/场景与输出.md`` §2。
    """
    train_files = _list_dir_files(rp.train, ("*.pth", "*.csv", "*.json", "*.png"))
    eval_files = _list_dir_files(rp.eval, ("*.csv", "*.json"))
    upload_files = _list_dir_files(rp.uploads, ("*.csv", "*.txt"))
    report = None
    reports = sorted(rp.train.glob("training_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if reports:
        report = _read_json(reports[0])
    compare = _read_json(rp.compare_summary)
    return {
        "path": str(rp.root),
        "uploads": upload_files,
        "train_files": train_files,
        "eval_files": eval_files,
        "uploads_detail": _annotate_files(upload_files),
        "train_detail": _annotate_files(train_files),
        "eval_detail": _annotate_files(eval_files),
        "has_best": rp.best_model.exists(),
        "has_eval": rp.compare_summary.exists() or rp.compare_daily.exists(),
        "training_summary": (report or {}).get("training_summary") if isinstance(report, dict) else None,
        "compare_delta": (compare or {}).get("delta_pct") if isinstance(compare, dict) else None,
        "n_years": (compare or {}).get("n_years_ppo") if isinstance(compare, dict) else None,
        "zip_url": "/api/run/download_zip",
    }


def _session_folder_summary(sp: SessionPaths) -> dict[str, Any]:
    """微调会话目录摘要（含 review/bc/compare/metrics 文件用途说明）。"""
    compare_dirs: list[str] = []
    if sp.compare.exists():
        compare_dirs = [d.name + "/" for d in sorted(sp.compare.iterdir()) if d.is_dir()]
    review_files = _list_dir_files(sp.current, ("*.csv", "*.json"))
    bc_files = _list_dir_files(sp.bc, ("*.pth", "*.json"))
    bc_export = _list_dir_files(sp.root / "04_bc_export", ("*.json",))
    compare_files = _list_dir_files(sp.compare, ("*.json",)) + compare_dirs
    metrics_files = _list_dir_files(sp.metrics, ("*.json", "*.txt"))
    return {
        "path": str(sp.root),
        "uploads": _list_dir_files(sp.root / "uploads", ("*.csv",)),
        "train_files": _list_dir_files(sp.root / "01_train", ("*.pth", "*.json", "*.png", "*.csv")),
        "eval_files": _list_dir_files(sp.root / "02_eval", ("*.csv", "*.json")),
        "review_files": review_files,
        "bc_files": bc_files,
        "bc_export": bc_export,
        "compare_files": compare_files,
        "metrics_files": metrics_files,
        "review_detail": _annotate_files(review_files),
        "bc_detail": _annotate_files(bc_files),
        "bc_export_detail": _annotate_files(bc_export),
        "compare_detail": _annotate_files(compare_files),
        "metrics_detail": _annotate_files(metrics_files),
        "has_review": sp.review_json.exists() or sp.review_csv.exists(),
        "has_hitl": sp.hitl_model.exists(),
        "has_compare": (sp.compare / "three_way.json").exists(),
        "zip_url": "/api/session/download_zip",
    }


def _zip_directory(root: Path, arc_prefix: str | None = None) -> io.BytesIO:
    buf = io.BytesIO()
    prefix = arc_prefix or root.name
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(Path(prefix) / p.relative_to(root)))
    buf.seek(0)
    return buf


def _year_from_date_cell(v: Any) -> str | None:
    s = str(v or "").strip()
    if not s or s.lower() == "nan":
        return None
    m = re.match(r"^(\d{4})", s.replace("-", "/"))
    return m.group(1) if m else None


def _season_rows_from_policy(
    path: Path, year: str, *, include_actual_rain: bool = True
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if df.empty or "date" not in df.columns:
        return []
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        y = _year_from_date_cell(r.get("date"))
        if y != str(year):
            continue
        item: dict[str, Any] = {"date": str(r.get("date", "")), "year": y}
        fields = (
            "h_before",
            "h_after",
            "hmin",
            "hmax",
            "Hp",
            "Hc",
            "a_mm",
            "a_exec",
            "a_cmd",
            "stage",
            "P_cumu7",
            "et0_a",
        )
        if include_actual_rain:
            fields += ("P_1_a", "P_actual_display")
        for c in fields:
            if c not in df.columns:
                continue
            v = r.get(c)
            if pd.isna(v):
                item[c] = None
            else:
                try:
                    item[c] = float(v) if c != "stage" else str(v)
                except Exception:
                    item[c] = v
        # 统一雨量字段名，供前端降雨图使用
        if include_actual_rain and "rain" not in item:
            for rk in ("P_1_a", "P_actual_display"):
                if item.get(rk) is not None:
                    item["rain"] = item[rk]
                    break
        rows.append(item)
    return rows


def _yearly_arm_map(arm_dir: Path, prefer: str) -> dict[str, dict[str, Any]]:
    rows = _yearly_rows(arm_dir / f"yearly_metrics_{prefer}.csv")
    if not rows:
        cands = list(arm_dir.glob("yearly_metrics_*.csv"))
        if cands:
            rows = _yearly_rows(cands[0])
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        y = r.get("year")
        if y is not None:
            out[str(int(y) if not isinstance(y, str) else y)] = r
    return out


_KPI_MEAN_KEYS = (
    "irrigation_total",
    "drainage_total",
    "managed_water",
    "rain_utilization",
    "yield_factor",
    "irrigation_times",
    "drainage_times",
    "irrigation_operation_days",
    "drainage_operation_days",
    "irrigation_contiguous_segments",
    "drainage_contiguous_segments",
    "stress_days",
    "stress_days_all",
    "irrigation_small_run_2d_segments",
    "irrigation_small_run_2d_days",
    "irrigation_small_run_2d_mm",
    "irrigation_small_run_ge3d_segments",
    "irrigation_small_run_ge3d_days",
    "irrigation_small_run_ge3d_mm",
    "irrigation_topup_small_segments",
    "irrigation_topup_small_days",
    "irrigation_topup_small_mm",
    "drainage_small_run_2d_segments",
    "drainage_small_run_2d_days",
    "drainage_small_run_2d_mm",
    "drainage_small_run_ge3d_segments",
    "drainage_small_run_ge3d_days",
    "drainage_small_run_ge3d_mm",
)


def _means_from_yearly_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """从 yearly_metrics 行算多年均值（三方各臂 only=ppo/rule 时无 compare_summary）。"""
    out: dict[str, float] = {}
    for k in _KPI_MEAN_KEYS:
        vals: list[float] = []
        for r in rows:
            v = r.get(k)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except Exception:
                pass
        if vals:
            out[k] = float(sum(vals) / len(vals))
    if rows:
        out["n_years"] = float(len(rows))
    return out


def _arm_means(arm_dir: Path, prefer: str) -> dict[str, float]:
    rows = _yearly_rows(arm_dir / f"yearly_metrics_{prefer}.csv")
    if not rows:
        for p in sorted(arm_dir.glob("yearly_metrics_*.csv")):
            rows = _yearly_rows(p)
            if rows:
                break
    return _means_from_yearly_rows(rows)


def _validate_run(
    rp: RunPaths,
    *,
    need_train: bool = True,
    need_eval: bool = True,
) -> dict[str, Any]:
    from common.scenario import validate_scenario_data

    sc = _scenario_for_run(rp)
    out: dict[str, Any] = {
        "ok": False,
        "source": "uploads" if any(rp.uploads.glob("*.csv")) else "scenario",
        "uploads": sorted(p.name for p in rp.uploads.glob("*.csv")),
        "scenario_name": sc.name,
        "start": sc.start,
        "terminal": sc.terminal,
        "duration_days": sc.duration_days,
        "need_train": need_train,
        "need_eval": need_eval,
        "has_best": rp.best_model.exists(),
        "errors": [],
    }
    try:
        validate_scenario_data(sc, need_train=need_train, need_eval=need_eval)
        out["ok"] = True
    except ValueError as e:
        msg = str(e)
        errs = [x.strip() for x in msg.replace("情景数据校验失败:", "").split("- ") if x.strip()]
        out["errors"] = errs or [msg]
    except Exception as e:
        out["errors"] = [str(e)]
    rp.update_meta(data_validate=out)
    return out


def _run_status(rp: RunPaths | None = None) -> dict[str, Any]:
    if rp is None:
        rp = _RUN
    if rp is None:
        return {
            "run_id": None,
            "run": None,
            "has_uploads": False,
            "uploads": [],
            "data_ok": False,
            "data_validate": None,
            "has_best": False,
            "has_eval": False,
            "train_busy": bool(_TRAIN_STATE.get("busy")),
            "train_error": _TRAIN_STATE.get("error"),
            "note": None,
            "folder": None,
        }
    meta = _read_json(rp.root / "meta.json") or {}
    dv = meta.get("data_validate") if isinstance(meta.get("data_validate"), dict) else None
    uploads = sorted(p.name for p in rp.uploads.glob("*.csv")) if rp.uploads.exists() else []
    return {
        "run_id": rp.root.name,
        "run": str(rp.root),
        "has_uploads": len(uploads) > 0,
        "uploads": uploads,
        "uploads_source": meta.get("uploads_source"),
        "uploads_snapshot": meta.get("uploads_snapshot"),
        "data_ok": bool(dv and dv.get("ok")),
        "data_validate": dv,
        "has_best": rp.best_model.exists(),
        "has_eval": rp.compare_summary.exists() or rp.compare_daily.exists(),
        "train_busy": bool(_TRAIN_STATE.get("busy") and _TRAIN_STATE.get("run_id") == rp.root.name),
        "train_error": _TRAIN_STATE.get("error") if _TRAIN_STATE.get("run_id") == rp.root.name else None,
        "note": meta.get("note"),
        "meta": meta,
        "folder": _run_folder_summary(rp),
    }


def _session_status(sp: SessionPaths | None = None) -> dict[str, Any]:
    if sp is None:
        sp = _SESSION
    if sp is None:
        return {
            "session_id": None,
            "session": None,
            "run_id": None,
            "has_review": False,
            "has_hitl": False,
            "has_compare": False,
            "n_days": 0,
            "years": [],
            "expected_review_years": [],
            "review_years_in_table": [],
            "confirmed_years": [],
            "edited_years": [],
            "missing_review_years": [],
            "bc_ready": False,
            "n_prefer": 0,
            "n_confirmed": 0,
            "n_edit": 0,
            "n_needs_reconfirm": 0,
            "wizard_step": 1,
            "hitl_info": None,
            "absorption": None,
            "bc_dashboard": None,
            "task_compare": None,
            "bc_epochs": BC_DEFAULT.epochs,
            "bc_w_edit_default": float(BC_DEFAULT.w_edit),
            "bc_rationale": describe_bc_rationale(),
            "folder": None,
        }
    has_review = sp.review_json.exists() or sp.review_csv.exists()
    df = pd.DataFrame()
    if has_review:
        p = sp.review_json if sp.review_json.exists() else sp.review_csv
        df = pd.read_json(p) if p.suffix.lower() == ".json" else pd.read_csv(p)
        df = df.reset_index(drop=True)
    years = sorted(df["year"].astype(str).unique().tolist()) if len(df) and "year" in df.columns else []
    n_prefer = int(df["prefer_review"].fillna(0).astype(int).sum()) if "prefer_review" in df.columns else 0
    n_conf = int(df["confirmed"].fillna(False).astype(bool).sum()) if "confirmed" in df.columns else 0
    meta = _read_json(sp.root / "meta.json") or {}
    split = meta.get("year_split") or {}
    finalized_years = [str(y) for y in (meta.get("finalized_years") or [])]
    confirmed_years = list(finalized_years)  # 已定稿年（无逐日勾选）
    from ..exec_layer import expert_action_float
    from ..pipeline import assess_bc_readiness

    # 与 CLI bc() 同一预检（含指纹；轻门槛过后再做 parity）
    gate = assess_bc_readiness(
        df if has_review else pd.DataFrame(),
        meta,
        workdir=sp.workdir if has_review else None,
        require_edits=True,
        require_finalized_years=True,
        check_fingerprints=True,
        check_parity=True,
    )
    n_edit = int(gate.get("n_edit") or 0)
    n_need = int(gate.get("n_needs_reconfirm") or 0)
    edited_years: list[str] = []
    if len(df) and "year" in df.columns and n_edit > 0:
        for y in years:
            sub = df[df["year"].astype(str) == str(y)]
            if "editable" in sub.columns:
                sub = sub[sub["editable"].fillna(True).astype(bool)]
            if len(sub) == 0 or "a_expert" not in sub.columns or "a_model" not in sub.columns:
                continue
            ex_series = sub["a_expert"].map(expert_action_float)
            edit_mask = ex_series.notna() & (
                ex_series.astype(float) - sub["a_model"].astype(float)
            ).abs().gt(1e-6)
            if int(edit_mask.sum()) > 0:
                edited_years.append(str(y))
        edited_years = sorted(set(edited_years))
    expected_review = [str(y) for y in (split.get("review_years") or split.get("review_used") or [])]
    if not expected_review:
        from ..config import resolve_year_split

        expected_review = list(resolve_year_split(years).get("review_years") or [])
    review_years_in_table = list(gate.get("review_years_needed") or [])
    if not review_years_in_table:
        review_years_in_table = [y for y in expected_review if y in set(years)]
        if not review_years_in_table and years and expected_review:
            review_years_in_table = [y for y in years if y in set(expected_review)] or list(years)
    missing_review_years = list(gate.get("missing_review_years") or [])
    has_hitl = sp.hitl_model.exists()
    bc_ready = bool(has_review and gate.get("bc_ready"))
    # 有 HITL 权重且当前标签仍就绪时才暴露吸收率/三方等事后结果
    results_ready = bool(has_hitl and bc_ready)
    bc_blockers = list(gate.get("blockers") or [])
    has_compare = (sp.compare / "three_way.json").exists()
    if not has_review:
        step = 1
    elif not has_hitl:
        step = 2
    elif not has_compare:
        step = 3
    else:
        step = 4
    rationale = describe_bc_rationale()
    if edited_years or confirmed_years:
        rationale += (
            f" 本会话已确认年={','.join(confirmed_years) or '—'}；"
            f"已改动年={','.join(edited_years) or '—'}；改动日={n_edit}。"
        )
        if missing_review_years:
            rationale += f" 尚未定稿的审阅年={','.join(missing_review_years)}（须逐年定稿后再 BC）。"
        if bc_blockers and not bc_ready:
            rationale += " BC 未就绪：" + "；".join(bc_blockers[:4]) + ("…" if len(bc_blockers) > 4 else "") + "。"
    return {
        "session_id": sp.root.name,
        "session": str(sp.root),
        "run_id": meta.get("run_id"),
        "has_review": has_review,
        "has_hitl": has_hitl,
        "has_compare": has_compare,
        "n_days": int(len(df)),
        "years": years,
        "expected_review_years": expected_review,
        "review_years_in_table": review_years_in_table,
        "confirmed_years": confirmed_years,
        "finalized_years": finalized_years,
        "edited_years": edited_years,
        "missing_review_years": missing_review_years,
        "bc_ready": bc_ready,
        "bc_blockers": bc_blockers,
        "fingerprint_bad_years": list(gate.get("fingerprint_bad_years") or []),
        "parity_ok": (None if gate.get("parity") is None else bool(gate["parity"].get("ok"))),
        "n_prefer": n_prefer,
        "n_confirmed": n_conf,
        "n_edit": n_edit,
        "n_needs_reconfirm": n_need,
        "wizard_step": step,
        "hitl_info": _read_json(sp.bc / "hitl_result.json") if results_ready else None,
        "absorption": _read_json(sp.metrics / "absorption.json") if results_ready else None,
        "bc_dashboard": build_bc_dashboard(sp) if results_ready else None,
        "task_compare": _read_json(sp.metrics / "task_compare.json") if results_ready else None,
        "bc_epochs": BC_DEFAULT.epochs,
        "bc_w_edit_default": float(BC_DEFAULT.w_edit),
        "bc_rationale": rationale,
        "folder": _session_folder_summary(sp),
    }


def _review_is_blind(sp: SessionPaths | None = None) -> bool:
    """审阅标签尚未冻结并产出当前 BC 时，隐藏事后结果与原始数据下载。"""
    sp = sp or _SESSION
    if sp is None:
        return False
    has_review = sp.review_json.exists() or sp.review_csv.exists()
    if not has_review:
        return False
    if not sp.hitl_model.exists():
        return True
    return not bool(_session_status(sp).get("bc_ready"))


def _persist_review_enrichment(
    sp: SessionPaths, df: pd.DataFrame | None
) -> pd.DataFrame | None:
    """清除审阅表中的事后实测雨，只回填经验规则对照并写回。"""
    if df is None or df.empty or "date" not in getattr(df, "columns", []):
        return df
    out = df.copy()
    dirty = False
    # 老会话可能存过实测雨；从当前审阅文件剔除，避免看板或下载再次暴露。
    rain_cols = [c for c in ("P_1_a", "P_actual_display", "rain") if c in out.columns]
    if rain_cols:
        out = out.drop(columns=rain_cols)
        dirty = True

    # —— 经验规则 a_rule（旧表无此列 → 界面全是「—」）——
    need_rule = "a_rule" not in out.columns or int(out["a_rule"].map(_numeric_cell).isna().sum()) > max(
        3, int(0.5 * len(out))
    )
    if need_rule:
        rule_map = _rule_action_map(sp)
        if rule_map:
            rules: list[Any] = []
            filled_r = 0
            for _, r in out.iterrows():
                cur = _numeric_cell(r.get("a_rule")) if "a_rule" in out.columns else None
                if cur is None:
                    for k in _date_key_variants(r.get("date")):
                        if k in rule_map:
                            cur = float(rule_map[k])
                            filled_r += 1
                            break
                rules.append(cur if cur is not None else float("nan"))
            if filled_r > 0:
                out["a_rule"] = rules
                dirty = True

    if dirty:
        try:
            _save_review_df(out)
        except Exception:
            pass
    return out


def _sort_review_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """按真实日历排序（勿对无前导零 Y/M/D 字符串排序）。"""
    if df is None or df.empty or "date" not in df.columns:
        return df
    from ..exec_layer import date_ymd

    out = df.copy()
    out["_ord"] = out["date"].map(date_ymd)
    if "year" in out.columns:
        out = out.sort_values(["year", "_ord"], kind="mergesort")
    else:
        out = out.sort_values(["_ord"], kind="mergesort")
    return out.drop(columns=["_ord"]).reset_index(drop=True)


def _load_review_df() -> pd.DataFrame:
    sp = _sp()
    p = sp.review_json if sp.review_json.exists() else sp.review_csv
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_json(p) if p.suffix.lower() == ".json" else pd.read_csv(p)
    df = _sort_review_calendar(df.reset_index(drop=True))
    return _persist_review_enrichment(sp, df)


def _save_review_df(df: pd.DataFrame) -> None:
    sp = _sp()
    sp.current.mkdir(parents=True, exist_ok=True)
    df = _sort_review_calendar(df)
    df.to_json(sp.review_json, orient="records", force_ascii=False)
    drop = [c for c in df.columns if c.startswith("prob_") or c == "state_vec"]
    df.drop(columns=drop, errors="ignore").to_csv(
        sp.review_csv, index=False, encoding="utf-8-sig"
    )


def _apply_overrides(
    df: pd.DataFrame,
    overrides: dict | None = None,
    reasons: dict | None = None,
    confirms: dict | None = None,
    reason_codes: dict | None = None,
) -> int:
    """写入期望动作 / 确认 / 结构化改因 + 自由文本。确认可独立于改动作。"""
    from ..config import REASON_CODE_LABELS
    from ..exec_layer import expert_action_filled

    if "a_expert" not in df.columns:
        df["a_expert"] = pd.NA
    if "confirmed" not in df.columns:
        df["confirmed"] = False
    if "reason" not in df.columns:
        df["reason"] = ""
    if "reason_code" not in df.columns:
        df["reason_code"] = ""
    overrides = overrides or {}
    reasons = reasons or {}
    reason_codes = reason_codes or {}
    confirms = confirms or {}
    n = 0
    touched = set()
    for k, v in overrides.items():
        i = int(k)
        if i < 0 or i >= len(df):
            continue
        if "editable" in df.columns and not bool(df.at[i, "editable"]):
            continue
        if v is None or v == "" or (isinstance(v, str) and not str(v).strip()):
            df.at[i, "a_expert"] = pd.NA
        else:
            from common.execution import validate_requested_action
            df.at[i, "a_expert"] = validate_requested_action(df.loc[i], v)
        touched.add(i)
        n += 1
    for k, flag in confirms.items():
        i = int(k)
        if i < 0 or i >= len(df):
            continue
        if "editable" in df.columns and not bool(df.at[i, "editable"]):
            continue
        if bool(flag):
            pending = bool(df.at[i, "needs_reconfirm"]) if "needs_reconfirm" in df.columns else False
            if pending and not expert_action_filled(df.at[i, "a_expert"]):
                # 确认重推后的当前建议：显式记下 PPO 当前动作，标签与已执行轨迹一致。
                df.at[i, "a_expert"] = float(df.at[i, "a_model"])
            if "needs_reconfirm" in df.columns:
                df.at[i, "needs_reconfirm"] = False
        df.at[i, "confirmed"] = bool(flag)
        if bool(flag) and i not in touched:
            touched.add(i)
            n += 1
    for k, code in reason_codes.items():
        i = int(k)
        if i < 0 or i >= len(df):
            continue
        c = str(code or "").strip()
        if c and c not in REASON_CODE_LABELS:
            continue
        df.at[i, "reason_code"] = c
    for k, text in reasons.items():
        i = int(k)
        if i < 0 or i >= len(df):
            continue
        df.at[i, "reason"] = str(text)
    return n


def _latest_train_log(rp: RunPaths) -> Path | None:
    logs = sorted(rp.train.glob("training_log_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def _latest_train_loss(rp: RunPaths) -> Path | None:
    logs = sorted(rp.train.glob("training_loss_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    # 优先按局聚合的 training_loss_*（排除 detail）
    for p in logs:
        if "detail" not in p.name:
            return p
    return logs[0] if logs else None


def _yearly_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        item: dict[str, Any] = {}
        for c in df.columns:
            v = r.get(c)
            if pd.isna(v):
                item[c] = None
            elif c == "year":
                item[c] = int(v)
            else:
                try:
                    item[c] = float(v)
                except Exception:
                    item[c] = v
        rows.append(item)
    return rows


def _run_summary(rp: RunPaths) -> dict[str, Any]:
    meta = _read_json(rp.root / "meta.json") or {}
    progress = _read_json(rp.train / "progress.json")
    report = None
    reports = sorted(rp.train.glob("training_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if reports:
        report = _read_json(reports[0])
    compare = _read_json(rp.compare_summary)
    log_p = _latest_train_log(rp)
    loss_p = _latest_train_loss(rp)
    n_log = 0
    if log_p and log_p.exists():
        try:
            n_log = len(pd.read_csv(log_p))
        except Exception:
            n_log = 0
    curves = sorted(rp.train.glob("training_results_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    yearly_ppo = _yearly_rows(rp.eval / "yearly_metrics_PPO.csv")
    yearly_rule = _yearly_rows(rp.eval / "yearly_metrics_RULE.csv")
    return {
        "run_id": rp.root.name,
        "meta": meta,
        "progress": progress,
        "report": report,
        "compare_summary": compare,
        "yearly_ppo": yearly_ppo,
        "yearly_rule": yearly_rule,
        "train_log": str(log_p) if log_p else None,
        "train_loss": str(loss_p) if loss_p else None,
        "train_log_rows": n_log,
        "curve_png": curves[0].name if curves else None,
        "has_best": rp.best_model.exists(),
        "has_eval": rp.compare_summary.exists() or rp.compare_daily.exists(),
        "best_model": str(rp.best_model) if rp.best_model.exists() else None,
        "eval_note": "分年指标与 KPI 来自评估集 02_eval（非训练集）",
    }


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    rid = (request.args.get("run") or "").strip()
    if rid:
        try:
            root = resolve_run(_normalize_run_ref(rid), create=False)
            if root:
                _bind_run(root)
        except Exception:
            pass
    sid = (request.args.get("session") or "").strip()
    if sid:
        root = Path(sid) if Path(sid).is_absolute() else SESSIONS_ROOT / sid
        if root.exists() and (root / "meta.json").exists():
            _bind_session(root)
    tab = (request.args.get("tab") or "train").strip().lower()
    if tab not in ("train", "expert"):
        tab = "train"
    return render_template(
        "index.html",
        run=_run_status(),
        session=_session_status(),
        runs=list_runs(require_best=False, limit=40),
        sessions=list_sessions(40),
        tab=tab,
        asset_v=75,
    )


@app.get("/api/boot")
def api_boot():
    from ..config import (
        P_CUMU7_WINDOW,
        REASON_CODE_LABELS,
        REASON_CODE_ORDER,
        resolve_year_split,
    )

    return jsonify({
        "ok": True,
        "action_space": list(ACTION_SPACE),
        "run": _run_status(),
        "session": _session_status(),
        "runs": list_runs(require_best=False, limit=40),
        "sessions": list_sessions(40),
        "train_state": dict(_TRAIN_STATE),
        "year_split": resolve_year_split(),
        "reason_codes": [
            {"code": c, "label": REASON_CODE_LABELS[c]} for c in REASON_CODE_ORDER
        ],
        "p_cumu7_window": P_CUMU7_WINDOW,
        "labels": {
            "bc": "行为克隆(BC)",
            "bc_full": "行为克隆：用专家确认动作监督微调策略（交叉熵+KL），不是再跑 PPO",
        },
    })


# ---------------------------------------------------------------------------
# Flow 1 — runs
# ---------------------------------------------------------------------------

@app.get("/api/runs")
def api_runs():
    return jsonify({
        "ok": True,
        "runs": list_runs(require_best=False, limit=40),
        "current": _RUN.root.name if _RUN else None,
    })


def _snapshot_scenario_into_run(rp) -> dict[str, Any]:
    """把当前 active scenario 的训/评表拷入本 run 的 ``uploads/``（显式调用，非新建默认）。"""
    from common.scenario import get_scenario

    sc = get_scenario()
    files = rp.snapshot_scenario_tables(sc)
    name = getattr(sc, "name", None) or str(sc)
    rp.update_meta(
        uploads_source="scenario_snapshot",
        uploads_snapshot={"scenario": name, "files": files},
        data_validate=None,  # 覆盖后必须重新校验
    )
    return {
        "enabled": True,
        "files": files,
        "scenario_name": name,
        "note": (
            f"已将默认情景「{name}」表拷入 uploads/。"
            "请点「② 数据集校验」；通过后才允许开始训练。"
        ),
    }


@app.post("/api/run/new")
def api_run_new():
    """新建空白训评 run。默认**不**拷贝情景表；须上传或点「使用默认情景」后再校验。

    校验与训练只认磁盘 ``uploads/*.csv``，与浏览器 file input 是否选中无关。
    """
    try:
        data = request.get_json(silent=True) or {}
        note = (data.get("note") or "board_train").strip() or "board_train"
        # 默认 false：避免「未选手传文件却静默用默认表」造成误解
        do_snap = data.get("snapshot_scenario", False)
        if isinstance(do_snap, str):
            do_snap = do_snap.strip().lower() not in ("0", "false", "no")
        root = create_run(note=note)
        rp = _bind_run(root)
        snap: dict[str, Any] = {
            "enabled": False,
            "files": [],
            "scenario_name": None,
            "note": "uploads 为空。请「保存上传文件」或点「使用默认情景表」，再点「② 数据集校验」。",
        }
        if do_snap:
            try:
                snap = _snapshot_scenario_into_run(rp)
            except Exception as ex:
                snap["note"] = f"情景快照失败（uploads 为空，须手动上传）: {ex}"
                rp.update_meta(uploads_source="empty", data_validate=None)
        else:
            rp.update_meta(uploads_source="empty", data_validate=None)
        return jsonify({"ok": True, "run": _run_status(rp), "snapshot": snap})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/api/run/snapshot_scenario")
def api_run_snapshot_scenario():
    """显式：把默认情景表写入指定/当前 run 的 uploads/（须再校验才能训练）。"""
    try:
        data = request.get_json(silent=True) or {}
        rp = _require_run(data)
        snap = _snapshot_scenario_into_run(rp)
        return jsonify({"ok": True, "snapshot": snap, "run": _run_status(rp)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/api/run/switch")
def api_run_switch():
    try:
        data = request.get_json(silent=True) or {}
        rid = (data.get("run_id") or data.get("run") or "").strip()
        if not rid:
            return jsonify({"ok": False, "error": "缺少 run_id"}), 400
        root = resolve_run(_normalize_run_ref(rid), create=False)
        if root is None:
            return jsonify({"ok": False, "error": f"找不到 run: {rid}"}), 404
        rp = _bind_run(root)
        return jsonify({"ok": True, "run": _run_status(rp), "summary": _run_summary(rp)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/api/run/upload")
def api_run_upload():
    """保存用户选中的文件到本 run。未选任何文件 → 失败（不会假装成功）。

    各数据槽位单独检查：有文件则必须是 ``.csv``；可选模型须为 ``.pth``/``.pt``。
    允许只覆盖其中几项；下方 chips 显示的是磁盘已有表，不等于本次上传内容。
    """
    try:
        rid = (request.form.get("run_id") or request.form.get("run") or "").strip()
        rp = _require_run({"run_id": rid} if rid else {})
        rp.uploads.mkdir(parents=True, exist_ok=True)
        rp.train.mkdir(parents=True, exist_ok=True)
        name_map = {
            "forecast": "forecast.csv",
            "actual": "actual.csv",
            "forecast_eval": "forecast_eval.csv",
            "actual_eval": "actual_eval.csv",
            "growth_para": "growth_para.csv",
        }
        field_labels = {
            "forecast": "训练预报",
            "actual": "训练实测",
            "forecast_eval": "评估预报",
            "actual_eval": "评估实测",
            "growth_para": "生育期",
        }
        saved: list[str] = []
        per_field: dict[str, str] = {}
        errors: list[str] = []
        for field, fname in name_map.items():
            f = request.files.get(field)
            label = field_labels.get(field, field)
            if f is None or not f.filename:
                per_field[field] = "未选择"
                continue
            raw_name = str(f.filename)
            if not raw_name.lower().endswith(".csv"):
                errors.append(f"「{label}」须为 .csv 文件，收到: {raw_name}")
                per_field[field] = f"拒绝（非 csv）: {raw_name}"
                continue
            dest = rp.uploads / fname
            f.save(str(dest))
            saved.append(fname)
            per_field[field] = f"已保存 → {fname}"
        # 可选：上传已训模型 → 01_train/ppo_rice_best.pth（路径 B：直推）
        model_file = request.files.get("model")
        model_saved = None
        if model_file is not None and model_file.filename:
            raw_m = str(model_file.filename)
            if not raw_m.lower().endswith((".pth", ".pt")):
                errors.append(f"已训模型须为 .pth / .pt，收到: {raw_m}")
                per_field["model"] = f"拒绝（非权重）: {raw_m}"
            else:
                dest = rp.best_model
                model_file.save(str(dest))
                model_saved = dest.name
                saved.append(f"01_train/{dest.name}")
                per_field["model"] = f"已保存 → {dest.name}"
        else:
            per_field["model"] = "未选择"

        if errors:
            return jsonify({
                "ok": False,
                "error": "；".join(errors),
                "per_field": per_field,
                "saved": saved,
                "run": _run_status(rp),
            }), 400
        if not saved:
            return jsonify({
                "ok": False,
                "error": (
                    "未选择任何文件：请在各槽位选择 .csv（或可选 .pth），"
                    "再点「保存上传文件」。"
                    "若要用情景默认表，请点「使用默认情景表」。"
                    "下方文件标签是磁盘已有内容，不是本次上传结果。"
                ),
                "per_field": per_field,
                "saved": [],
                "run": _run_status(rp),
            }), 400

        note = (request.form.get("note") or "").strip()
        model_name = (request.form.get("model_name") or "").strip()
        csv_saved = [s for s in saved if not s.startswith("01_train/")]
        meta_kw: dict[str, Any] = {}
        if csv_saved:
            # 覆盖上传后须重新校验；标记来源为用户上传
            meta_kw["uploads_source"] = "user_upload"
            meta_kw["data_validate"] = None
            meta_kw["uploads"] = sorted(
                {*(p.name for p in rp.uploads.glob("*.csv")), *csv_saved}
            )
        if model_saved:
            meta_kw["uploaded_best"] = model_saved
            meta_kw["best"] = str(rp.best_model)
        if note:
            meta_kw["note"] = note
        if model_name:
            meta_kw["model_name"] = model_name
        if meta_kw:
            rp.update_meta(**meta_kw)
        return jsonify({
            "ok": True,
            "saved": saved,
            "per_field": per_field,
            "model_saved": model_saved,
            "run": _run_status(rp),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/api/run/validate")
def api_run_validate():
    try:
        data = request.get_json(silent=True) or {}
        rp = _require_run(data)
        mode = (data.get("mode") or "full").strip()  # full | eval_only
        if mode == "eval_only":
            out = _validate_run(rp, need_train=False, need_eval=True)
        else:
            out = _validate_run(rp, need_train=True, need_eval=True)
        return jsonify({"ok": True, "validate": out, "run": _run_status(rp)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


def _progress_writer(rp: RunPaths):
    path = rp.train / "progress.json"

    def _cb(info: dict[str, Any]) -> None:
        payload = {
            **info,
            "status": "training",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "run_id": rp.root.name,
        }
        try:
            _write_json(path, payload)
        except Exception:
            pass

    return _cb


def _patch_progress(path: Path, **fields: Any) -> None:
    """合并写入 progress.json，保留 episode/return 等已有指标。"""
    cur = _read_json(path) or {}
    if not isinstance(cur, dict):
        cur = {}
    cur.update(fields)
    cur["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(path, cur)


def _train_worker(run_root: Path, episodes: int, model_name: str, mode: str, max_years: int | None) -> None:
    global _TRAIN_STATE
    rp = RunPaths(run_root).ensure()
    progress_path = rp.train / "progress.json"
    try:
        from api import evaluate, train as api_train
        from common.scenario import activate_scenario_obj

        activate_scenario_obj(_scenario_for_run(rp))

        do_train = mode != "eval_only"
        if mode == "auto" and rp.best_model.exists():
            do_train = False

        if do_train:
            _write_json(progress_path, {
                "status": "training",
                "episode": 0,
                "total_episodes": episodes,
                "run_id": rp.root.name,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            })
            rp.update_meta(train_status="training")
            train_out = api_train(
                episodes=episodes,
                save_dir=rp.train,
                model_name=model_name or "ppo_rice",
                verbose=True,
                progress_callback=_progress_writer(rp),
                keep_active_scenario=True,
            )
            rp.update_meta(train=train_out.get("model_path"), train_status="trained")
        else:
            train_out = {"model_path": str(rp.best_model), "skipped_train": True}

        if not rp.best_model.exists():
            raise FileNotFoundError("训练后仍无 ppo_rice_best.pth")

        # 评估前再次激活（train 过程可能 chdir；情景对象仍有效）
        activate_scenario_obj(_scenario_for_run(rp))
        _patch_progress(
            progress_path,
            status="evaluating",
            run_id=rp.root.name,
            error=None,
        )
        rp.update_meta(train_status="evaluating")
        eval_out = evaluate(
            rp.best_model,
            run=rp.root,
            max_years=max_years,
            keep_active_scenario=True,
        )
        rp.update_meta(
            eval_dir=str(rp.eval),
            best=str(rp.best_model),
            train_status="done",
            eval_ok=True,
        )
        _patch_progress(
            progress_path,
            status="done",
            run_id=rp.root.name,
            error=None,
            train={"model_path": train_out.get("model_path"), "skipped": train_out.get("skipped_train")},
            eval_keys=list((eval_out or {}).keys())[:20],
        )
        with _TRAIN_LOCK:
            _TRAIN_STATE["busy"] = False
            _TRAIN_STATE["error"] = None
    except Exception as e:
        err = str(e)
        _patch_progress(
            progress_path,
            status="error",
            error=err,
            run_id=rp.root.name,
            trace=traceback.format_exc()[-2000:],
        )
        try:
            rp.update_meta(train_status="error", train_error=err)
        except Exception:
            pass
        with _TRAIN_LOCK:
            _TRAIN_STATE["busy"] = False
            _TRAIN_STATE["error"] = err


@app.post("/api/run/start_train")
def api_run_start_train():
    try:
        data = request.get_json(silent=True) or {}
        rp = _require_run(data)
        episodes = int(data.get("episodes") or 1500)
        model_name = (data.get("model_name") or "ppo_rice").strip() or "ppo_rice"
        mode = (data.get("mode") or "auto").strip()  # auto | train | eval_only
        max_years = data.get("max_years")
        if max_years is not None and str(max_years).strip() != "":
            max_years = int(max_years)
        else:
            max_years = None

        # 门禁：校验通过才允许训练/整季推理；eval_only 可只验评估集，但必须有 best
        if mode == "eval_only":
            if not rp.best_model.exists():
                return jsonify({
                    "ok": False,
                    "error": "仅整季推理需要已有 best（请先训练或上传 .pth）",
                }), 400
            v = _validate_run(rp, need_train=False, need_eval=True)
        else:
            v = _validate_run(rp, need_train=True, need_eval=True)
        if not v.get("ok"):
            return jsonify({"ok": False, "error": "数据集校验未通过，禁止启动", "validate": v}), 400

        with _TRAIN_LOCK:
            if _TRAIN_STATE.get("busy"):
                return jsonify({
                    "ok": False,
                    "error": f"已有训练在进行: {_TRAIN_STATE.get('run_id')}",
                }), 409
            _TRAIN_STATE["busy"] = True
            _TRAIN_STATE["run_id"] = rp.root.name
            _TRAIN_STATE["error"] = None

        t = threading.Thread(
            target=_train_worker,
            args=(rp.root, episodes, model_name, mode, max_years),
            daemon=True,
        )
        t.start()
        return jsonify({"ok": True, "started": True, "run": _run_status(rp)})
    except Exception as e:
        with _TRAIN_LOCK:
            _TRAIN_STATE["busy"] = False
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/api/run/progress")
def api_run_progress():
    try:
        rp = _rp()
        prog = _read_json(rp.train / "progress.json") or {}
        same = _TRAIN_STATE.get("run_id") == rp.root.name
        err = prog.get("error") or (_TRAIN_STATE.get("error") if same else None)
        return jsonify({
            "ok": True,
            "progress": prog,
            "busy": bool(_TRAIN_STATE.get("busy") and same),
            "error": err,
            "run": _run_status(rp),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/run/train_log")
def api_run_train_log():
    try:
        rp = _rp()
        log_p = _latest_train_log(rp)
        loss_p = _latest_train_loss(rp)
        rows: list[dict[str, Any]] = []
        if log_p and log_p.exists():
            df = pd.read_csv(log_p)
            # 均匀下采样，最多约 800 点给曲线
            if len(df) > 800:
                step = max(1, len(df) // 800)
                df = df.iloc[::step]
            for _, r in df.iterrows():
                rows.append({
                    "episode": int(r.get("episode", 0) or 0) if "episode" in df.columns else 0,
                    "return": float(r.get("return", r.get("episode_return", 0)) or 0),
                    "mean_return": float(r.get("mean_return", r.get("mean10", 0)) or 0)
                    if ("mean_return" in df.columns or "mean10" in df.columns)
                    else None,
                    "action_count": float(r.get("action_count", 0) or 0)
                    if "action_count" in df.columns
                    else None,
                })
        loss_rows: list[dict[str, Any]] = []
        if loss_p and loss_p.exists():
            ldf = pd.read_csv(loss_p)
            if len(ldf) > 400:
                step = max(1, len(ldf) // 400)
                ldf = ldf.iloc[::step]
            for _, r in ldf.iterrows():
                loss_rows.append({
                    "episode": int(r.get("episode", 0) or 0),
                    "policy_loss": float(r.get("policy_loss", 0) or 0),
                    "value_loss": float(r.get("value_loss", 0) or 0),
                    "entropy_loss": float(r.get("entropy_loss", 0) or 0),
                    "total_loss": float(r.get("total_loss", 0) or 0)
                    if "total_loss" in ldf.columns
                    else None,
                })
        prog = _read_json(rp.train / "progress.json") or {}
        return jsonify({
            "ok": True,
            "rows": rows,
            "loss_rows": loss_rows,
            "progress": prog,
            "log_file": str(log_p) if log_p else None,
            "loss_file": str(loss_p) if loss_p else None,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "rows": []}), 500


@app.get("/api/run/summary")
def api_run_summary():
    try:
        if _review_is_blind():
            return jsonify({"ok": False, "error": "专家审阅尚未完成，暂不提供训评结果摘要"}), 423
        rp = _rp()
        return jsonify({"ok": True, "summary": _run_summary(rp), "run": _run_status(rp)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/run/download/<name>")
def api_run_download(name: str):
    if _review_is_blind():
        return jsonify({"ok": False, "error": "专家审阅尚未完成，暂不开放训练结果文件下载"}), 423
    try:
        rp = _rp()
    except RuntimeError:
        flash("无当前 run", "error")
        return redirect(url_for("index"))
    allowed: dict[str, Path] = {
        "ppo_rice_best.pth": rp.best_model,
        "compare_summary.json": rp.compare_summary,
        "policy_compare_daily.csv": rp.compare_daily,
    }
    log_p = _latest_train_log(rp)
    if log_p:
        allowed["training_log.csv"] = log_p
    loss_p = _latest_train_loss(rp)
    if loss_p:
        allowed["training_loss.csv"] = loss_p
    for yname in ("yearly_metrics_PPO.csv", "yearly_metrics_RULE.csv"):
        yp = rp.eval / yname
        if yp.exists():
            allowed[yname] = yp
    curves = sorted(rp.train.glob("training_results_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if curves:
        allowed["training_curve.png"] = curves[0]
    path = allowed.get(name)
    if path is None or not path.exists():
        flash(f"文件不存在: {name}", "error")
        return redirect(url_for("index", tab="train"))
    return send_file(path, as_attachment=True)


@app.get("/api/run/download_zip")
def api_run_download_zip():
    try:
        if _review_is_blind():
            return jsonify({"ok": False, "error": "专家审阅尚未完成，暂不开放训练会话打包下载"}), 423
        rp = _rp()
        buf = _zip_directory(rp.root, arc_prefix=rp.root.name)
        return send_file(
            buf,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"{rp.root.name}.zip",
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/run/season")
def api_run_season():
    """评估集单年整季轨迹（PPO / 规则），供训评结果按年查看。"""
    try:
        rp = _rp()
        year = (request.args.get("year") or "").strip()
        years = [
            str(int(r["year"]))
            for r in _yearly_rows(rp.eval / "yearly_metrics_PPO.csv")
            if r.get("year") is not None
        ]
        if not years and (rp.eval / "policy_output_PPO.csv").exists():
            df = pd.read_csv(rp.eval / "policy_output_PPO.csv", usecols=["date"])
            years = sorted({y for y in (_year_from_date_cell(d) for d in df["date"]) if y})
        if not year:
            return jsonify({
                "ok": True,
                "year": None,
                "years": years,
                "ppo": [],
                "rule": [],
                "note": "选择年份查看整季生育期轨迹",
            })
        review_blind = _review_is_blind()
        return jsonify({
            "ok": True,
            "year": year,
            "years": years,
            "ppo": _season_rows_from_policy(
                rp.eval / "policy_output_PPO.csv", year,
                include_actual_rain=not review_blind,
            ),
            "rule": _season_rows_from_policy(
                rp.eval / "policy_output_RULE.csv", year,
                include_actual_rain=not review_blind,
            ),
            "review_blind": review_blind,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


# ---------------------------------------------------------------------------
# Flow 2 — hitl sessions
# ---------------------------------------------------------------------------

@app.get("/api/sessions")
def api_sessions():
    return jsonify({
        "ok": True,
        "sessions": list_sessions(40),
        "current": _SESSION.root.name if _SESSION else None,
    })


@app.post("/api/switch_session")
def api_switch_session():
    try:
        data = request.get_json(silent=True) or {}
        sid = (data.get("session_id") or data.get("session") or "").strip()
        if not sid:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        root = Path(sid) if Path(sid).is_absolute() else SESSIONS_ROOT / sid
        if not root.exists() or not (root / "meta.json").exists():
            return jsonify({"ok": False, "error": f"会话不存在: {sid}"}), 404
        sp = _bind_session(root)
        return jsonify({"ok": True, "session": _session_status(sp)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/api/start_session")
def api_start_session():
    """选 run → 新建 hitl 会话 + seed + review。"""
    try:
        data = request.get_json(silent=True) or {}
        from_run = (data.get("from_run") or data.get("run_id") or "").strip()
        if not from_run and _RUN is not None:
            from_run = str(_RUN.root)
        if not from_run:
            return jsonify({"ok": False, "error": "请选择训练 run"}), 400
        from_run = _normalize_run_ref(from_run)
        max_years = data.get("max_years")
        if max_years is not None and str(max_years).strip() != "":
            max_years = int(max_years)
        else:
            max_years = None

        root = create_session(note=f"expert_from_{Path(from_run).name}")
        _bind_session(root)
        info = seed(root, from_run=from_run)
        if not info.get("ok"):
            return jsonify({
                "ok": False,
                "error": "导入失败: " + "; ".join(info.get("missing") or ["未知"]),
                "seed": info,
            }), 400
        # 仅记审阅模式/待定状态；真正 feedback_source 等定稿后再写（勿提前标 human）
        sp = _sp()
        sp.update_meta(
            review_mode="human_dashboard",
            feedback_status="pending",
            feedback_sources_by_year={},
            paper_protocol="final_human_review_v1",
        )
        df = review(work=root, max_years=max_years, do_seed=False)
        n_prefer = (
            int(df["prefer_review"].fillna(0).astype(int).sum())
            if "prefer_review" in df.columns
            else 0
        )
        return jsonify({
            "ok": True,
            "seed": info,
            "n_days": int(len(df)),
            "n_prefer": n_prefer,
            "session": _session_status(),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


def _numeric_cell(v: Any) -> float | None:
    """解析数值；``?`` / 空 / NaN → None（审阅表旧数据 P_actual_display 常为占位符）。"""
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if pd.isna(v):
            return None
        s = str(v).strip()
        if not s or s in ("?", "nan", "None", "null"):
            return None
        return float(s)
    except Exception:
        return None


def _date_key_variants(d: Any) -> list[str]:
    s = str(d or "").strip()
    if not s:
        return []
    keys = {s, s.replace("-", "/"), s.replace("/", "-")}
    m = re.match(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})", s)
    if m:
        y, mo, da = m.group(1), int(m.group(2)), int(m.group(3))
        keys.add(f"{y}/{mo}/{da}")
        keys.add(f"{y}-{mo:02d}-{da:02d}")
        keys.add(f"{y}/{mo:02d}/{da:02d}")
    return list(keys)


def _rule_action_map(sp: SessionPaths) -> dict[str, float]:
    """按日对齐固定规则动作 a_rule（来自 policy_compare_daily）。"""
    path = sp.phase1_eval / "policy_compare_daily.csv"
    out: dict[str, float] = {}
    if not path.exists():
        return out
    try:
        df = pd.read_csv(path)
    except Exception:
        return out
    if "date" not in df.columns or "a_rule" not in df.columns:
        return out
    for _, r in df.iterrows():
        v = _numeric_cell(r.get("a_rule"))
        if v is None:
            continue
        for k in _date_key_variants(r.get("date")):
            out[k] = float(v)
    return out


@app.get("/api/rows")
def api_rows():
    try:
        year = request.args.get("year")
        prefer_only = request.args.get("prefer") == "1"
        full = _load_review_df()
        if full.empty:
            return jsonify({"ok": True, "rows": [], "years": [], "action_space": list(ACTION_SPACE)})
        full = _sort_review_calendar(full.reset_index(drop=True))
        years = sorted(full["year"].astype(str).unique().tolist())
        view = full
        if year:
            view = view[view["year"].astype(str) == str(year)]
        # prefer 筛选放在按「执行≠规则」重算之后（勿用旧表 prefer_review 列）
        sp = _sp()
        rule_map = _rule_action_map(sp)
        from ..exec_layer import expert_action_float, expert_action_filled, fixed_rule_label

        def _f(row, key, default=0.0):
            v = _numeric_cell(row.get(key))
            return default if v is None else v

        def _optional_f(row, key):
            v = _numeric_cell(row.get(key))
            return None if v is None else float(v)

        def _lookup_map(row, amap: dict[str, float]) -> float | None:
            for k in _date_key_variants(row.get("date")):
                if k in amap:
                    return float(amap[k])
            return None

        def _a_rule(row) -> float | None:
            v = _numeric_cell(row.get("a_rule"))
            if v is not None:
                return v
            return _lookup_map(row, rule_map)

        rows = []
        for idx, r in view.iterrows():
            forecast_rain_7d = _optional_f(r, "P_cumu7")
            a_rule = _a_rule(r)
            # 当前审阅水层下的规定规则建议（与原轨迹 a_rule 区分）
            a_rule_cur = None
            try:
                from decision.infer import fixed_rule_action

                hb, hmn, hmx = _f(r, "h_before"), _f(r, "hmin"), _f(r, "hmax")
                hp, hc = _f(r, "Hp"), _f(r, "Hc")
                if None not in (hb, hmn, hmx, hp, hc):
                    from evaluation.run_eval import _rule_mm
                    a_rule_cur = float(fixed_rule_action_mm(r)) if bool(r.get("used_fixed_rule", False)) else _rule_mm(r, {})[0]
            except Exception:
                a_rule_cur = None
            a_exec = _f(r, "a_exec", _f(r, "a_model"))
            # 分歧 = 执行量 ≠ 规则（与专家改动无关）；无规则日沿用表内 prefer_review
            if a_rule is not None:
                prefer = int(abs(float(a_exec) - float(a_rule)) > 1e-6)
            else:
                prefer = int(r.get("prefer_review", 0) or 0)
            stage = str(r.get("stage", "") or "")
            editable = bool(r.get("editable", True))
            used_fixed = bool(r.get("used_fixed_rule", False))
            flabel = fixed_rule_label(
                None, stage_name=stage, hmin=_numeric_cell(r.get("hmin"))
            )
            if flabel is None and (used_fixed or not editable):
                flabel = "固定规则"
            rows.append({
                "i": int(idx),
                "year": str(r.get("year", "")),
                "date": str(r.get("date", "")),
                "stage": stage,
                "h_before": _f(r, "h_before"),
                "hmin": _f(r, "hmin"),
                "hmax": _f(r, "hmax"),
                "Hp": _f(r, "Hp"),
                "Hc": _f(r, "Hc"),
                "lamda": _f(r, "lamda", 1.0),
                "ks": _f(r, "ks", 1.0),
                "P_cumu7": forecast_rain_7d,
                "forecast_rain_7d": forecast_rain_7d,
                # 不向人工审阅接口返回当日实测雨量。
                "rain": None,
                "P_1_a": None,
                "P_actual_display": None,
                "a_model": _f(r, "a_model"),
                # 显式 float；0.0 合法。缺规则才为 null（前端显示 —）
                "a_rule": (None if a_rule is None else float(a_rule)),
                "a_rule_current_state": a_rule_cur,
                "a_idx": int(_f(r, "a_idx", 0)),
                "a_applied": _f(r, "a_applied", _f(r, "a_model")),
                "a_exec": float(a_exec) if a_exec is not None else None,
                "a_expert": expert_action_float(r.get("a_expert")),
                "a_requested_mm": (expert_action_float(r.get("a_expert")) if expert_action_filled(r.get("a_expert")) else _f(r, "a_model")),
                "allowed_actions": allowed_actions(r),
                "editable": editable,
                "prefer_review": prefer,
                "is_small_run": bool(r.get("is_small_run", False)),
                "confirmed": bool(r.get("confirmed", False)),
                "needs_reconfirm": bool(r.get("needs_reconfirm", False)),
                "used_fixed_rule": used_fixed or (flabel is not None),
                "fixed_label": flabel,
                "used_hc_guard": bool(r.get("used_hc_guard", False)),
                "reward": _f(r, "reward"),
                "reason": str(r.get("reason", "") or ""),
                "reason_code": str(r.get("reason_code", "") or ""),
            })
        if prefer_only:
            # 分歧审阅：固定规则日无需专家看
            rows = [
                row
                for row in rows
                if int(row.get("prefer_review") or 0) == 1 and row.get("editable", True)
            ]
        from ..config import P_CUMU7_WINDOW, REASON_CODE_LABELS, REASON_CODE_ORDER

        return jsonify({
            "ok": True,
            "rows": rows,
            "years": years,
            "action_space": list(ACTION_SPACE),
            "reason_codes": [
                {"code": c, "label": REASON_CODE_LABELS[c]} for c in REASON_CODE_ORDER
            ],
            "rain_meta": {
                "source": "forecast.P_cumu7",
                "actual_rain_hidden": True,
                "p_cumu7_window": P_CUMU7_WINDOW,
            },
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "rows": []}), 500


def _load_original_year_df(sp: SessionPaths, year: str) -> pd.DataFrame:
    """从 original_trajectories 取单年基线；缺单年文件则从 all_years 筛。"""
    y = str(year)
    yp = sp.original_traj / f"year_{y}.json"
    if yp.exists():
        return pd.read_json(yp)
    allp = sp.original_traj / "all_years.json"
    if not allp.exists():
        raise FileNotFoundError("无 original_trajectories，无法重置")
    full = pd.read_json(allp)
    if "year" not in full.columns:
        full = full.copy()
        full["year"] = full["date"].map(lambda d: str(d).split("/")[0].split("-")[0])
    out = full[full["year"].astype(str) == y].copy()
    if out.empty:
        raise ValueError(f"original 中无审阅年 {y}")
    return out


@app.post("/api/review/reset_year")
def api_reset_review_year():
    """把当前审阅年的 current 行恢复为 original，再整表日历排序写回。"""
    try:
        data = request.get_json(silent=True) or {}
        year = str(data.get("year") or "").strip()
        if not year:
            return jsonify({"ok": False, "error": "缺少 year"}), 400
        sp = _sp()
        cur = _load_review_df()
        if cur.empty:
            return jsonify({"ok": False, "error": "当前无审阅表"}), 400
        orig_y = _load_original_year_df(sp, year)
        # 去掉该年，拼回 original 年
        if "year" not in cur.columns:
            return jsonify({"ok": False, "error": "审阅表缺 year 列"}), 400
        keep = cur[cur["year"].astype(str) != year]
        # 对齐列：original 有的列优先，current 多出的列填默认
        merged = pd.concat([keep, orig_y], ignore_index=True, sort=False)
        _save_review_df(merged)
        meta = _read_json(sp.root / "meta.json") or {}
        fin = [str(y) for y in (meta.get("finalized_years") or []) if str(y) != year]
        fps = dict(meta.get("trajectory_fingerprints") or {})
        fps.pop(year, None)
        sources = dict(meta.get("feedback_sources_by_year") or {})
        sources.pop(year, None)
        sp.update_meta(
            finalized_years=fin,
            trajectory_fingerprints=fps,
            feedback_sources_by_year=sources,
        )
        sp.append_edit({
            "date": "",
            "a_expert": None,
            "reason": f"reset_year:{year}",
            "action": "reset_year_to_original",
        })
        return jsonify({
            "ok": True,
            "year": year,
            "n_restored": int(len(orig_y)),
            "n_days": int(len(merged)),
            "session": _session_status(),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/api/review/accept_pending_year")
def api_accept_pending_year():
    """一键认可本审阅年全部「需重看」：接纳当前 PPO 建议（不改变已执行轨迹语义）。

    「未改」要等定稿才批量确认；上游改动后的需重看日必须显式处理，本接口提供批量显式接纳。
    """
    from ..exec_layer import expert_action_filled

    try:
        data = request.get_json(silent=True) or {}
        year = str(data.get("year") or "").strip()
        if not year:
            return jsonify({"ok": False, "error": "缺少 year"}), 400
        sp = _sp()
        df = _load_review_df()
        if df.empty or "year" not in df.columns:
            return jsonify({"ok": False, "error": "当前无审阅表"}), 400
        if "confirmed" not in df.columns:
            df["confirmed"] = False
        if "needs_reconfirm" not in df.columns:
            df["needs_reconfirm"] = False
        if "a_expert" not in df.columns:
            df["a_expert"] = pd.NA
        take = df["year"].astype(str) == year
        if not bool(take.any()):
            return jsonify({"ok": False, "error": f"审阅表无 {year} 年"}), 400
        editable = df.get("editable", pd.Series(True, index=df.index)).fillna(False).astype(bool)
        pending = df["needs_reconfirm"].fillna(False).astype(bool)
        idxs = df.index[take & editable & pending].tolist()
        for i in idxs:
            if not expert_action_filled(df.at[i, "a_expert"]):
                df.at[i, "a_expert"] = float(df.at[i, "a_model"])
            df.at[i, "needs_reconfirm"] = False
            df.at[i, "confirmed"] = True
        _save_review_df(df)
        # 批量认可后该年旧定稿失效，须再点定稿
        meta = _read_json(sp.root / "meta.json") or {}
        fin = set(str(y) for y in (meta.get("finalized_years") or []))
        fps = dict(meta.get("trajectory_fingerprints") or {})
        fin.discard(year)
        fps.pop(year, None)
        sp.update_meta(finalized_years=sorted(fin), trajectory_fingerprints=fps)
        sp.append_edit({
            "date": "",
            "a_expert": None,
            "reason": f"accept_pending_year:{year}:n={len(idxs)}",
            "action": "accept_pending_year",
            "n_accepted": len(idxs),
        })
        return jsonify({
            "ok": True,
            "year": year,
            "n_accepted": len(idxs),
            "session": _session_status(),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/api/review/finalize_year")
def api_finalize_review_year():
    """接纳本审阅年当前轨迹：未改日=认可；写入 confirmed 供 BC。"""
    from ..exec_layer import expert_action_float

    try:
        data = request.get_json(silent=True) or {}
        year = str(data.get("year") or "").strip()
        if not year:
            return jsonify({"ok": False, "error": "缺少 year"}), 400
        sp = _sp()
        df = _load_review_df()
        if df.empty or "year" not in df.columns:
            return jsonify({"ok": False, "error": "当前无审阅表"}), 400
        sub = df[df["year"].astype(str) == year]
        if sub.empty:
            return jsonify({"ok": False, "error": f"审阅表无 {year} 年"}), 400
        if "confirmed" not in df.columns:
            df["confirmed"] = False
        if "needs_reconfirm" not in df.columns:
            df["needs_reconfirm"] = False
        n_edited = 0
        pending = sub.get(
            "needs_reconfirm", pd.Series(False, index=sub.index)
        ).fillna(False).astype(bool)
        editable = sub.get("editable", pd.Series(True, index=sub.index)).fillna(False).astype(bool)
        pending_days = sub.loc[pending & editable, "date"].astype(str).tolist()
        if pending_days:
            return jsonify({
                "ok": False,
                "error": (
                    f"{year} 年仍有 {len(pending_days)} 天待专家复核；"
                    "请点「一键认可本审阅年需重看」或逐日认可/改档后再定稿。"
                ),
                "n_pending": len(pending_days),
                "pending_dates": pending_days[:12],
            }), 409
        for i in sub.index:
            if "editable" in df.columns and not bool(df.at[i, "editable"]):
                continue
            ex_mm = expert_action_float(df.at[i, "a_expert"]) if "a_expert" in df.columns else None
            if (
                ex_mm is not None
                and "a_model" in df.columns
                and abs(ex_mm - float(df.at[i, "a_model"])) > 1e-6
            ):
                n_edited += 1
        from ..pipeline import finalize_review
        df = finalize_review(sp.root, df, years=[year], source="human_whole_season_acceptance")
        finalized_rows = df[df["year"].astype(str) == year]
        finalized_editable = finalized_rows.get(
            "editable", pd.Series(True, index=finalized_rows.index)
        ).fillna(False).astype(bool)
        n_conf = int(
            (finalized_editable & finalized_rows["confirmed"].fillna(False).astype(bool)).sum()
        )
        meta = _read_json(sp.root / "meta.json") or {}
        fps = dict(meta.get("trajectory_fingerprints") or {})
        sp.append_edit({
            "date": "",
            "a_expert": None,
            "reason": f"finalize_year:{year}",
            "action": "finalize_year",
            "trajectory_fingerprint": fps.get(year),
        })
        return jsonify({
            "ok": True,
            "year": year,
            "n_confirmed": n_conf,
            "n_edited": n_edited,
            "trajectory_fingerprint": fps.get(year),
            "session": _session_status(),
        })
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 409
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/api/resync")
def api_resync():
    from ..exec_layer import date_ymd, expert_action_filled, expert_action_float

    try:
        data = request.get_json(silent=True) or {}
        overrides = data.get("overrides") or {}
        reasons = data.get("reasons") or {}
        reason_codes = data.get("reason_codes") or {}
        confirms = data.get("confirms") or {}
        fresh: set[int] = set()
        # 只有「改了专家动作」才作为作废起点；仅确认不拖累更早/更晚日
        override_ids: set[int] = set()
        review_touch_ids: set[int] = set()
        confirm_events: list[int] = []
        deferred_override_ids: set[int] = set()
        deferred_action_dates: list[str] = []
        if overrides or reasons or reason_codes or confirms:
            df = _load_review_df()
            for k, v in (overrides or {}).items():
                i = int(k)
                if i < 0 or i >= len(df):
                    continue
                prev = df.at[i, "a_expert"] if "a_expert" in df.columns else None
                if v is None or v == "" or (isinstance(v, str) and not str(v).strip()):
                    changed = expert_action_filled(prev)
                else:
                    try:
                        new_mm = float(v)
                        prev_mm = expert_action_float(prev)
                        changed = prev_mm is None or abs(new_mm - prev_mm) > 1e-6
                    except (TypeError, ValueError):
                        changed = True
                if changed:
                    override_ids.add(i)
                    review_touch_ids.add(i)
            earliest_by_year: dict[str, tuple[int, int, int]] = {}
            for i in override_ids:
                y = str(df.at[i, "year"])
                ord_i = date_ymd(df.at[i, "date"])
                if y not in earliest_by_year or ord_i < earliest_by_year[y]:
                    earliest_by_year[y] = ord_i
            deferred_override_ids = {
                i for i in override_ids
                if date_ymd(df.at[i, "date"]) > earliest_by_year[str(df.at[i, "year"])]
            }
            deferred_action_dates = [str(df.at[i, "date"]) for i in sorted(deferred_override_ids)]
            effective_confirms: dict[str, Any] = {}
            for k, flag in (confirms or {}).items():
                i = int(k)
                if i < 0 or i >= len(df):
                    continue
                if "editable" in df.columns and not bool(df.at[i, "editable"]):
                    continue
                # 若同批次更早日期已改，当前行的旧轨迹确认随状态一起失效。
                if bool(flag) and str(df.at[i, "year"]) in earliest_by_year and (
                    date_ymd(df.at[i, "date"]) > earliest_by_year[str(df.at[i, "year"])]
                ):
                    continue
                effective_confirms[k] = flag
                old_confirmed = bool(df.at[i, "confirmed"]) if "confirmed" in df.columns else False
                old_pending = bool(df.at[i, "needs_reconfirm"]) if "needs_reconfirm" in df.columns else False
                old_expert = df.at[i, "a_expert"] if "a_expert" in df.columns else None
                if bool(flag) and (
                    not old_confirmed or old_pending or not expert_action_filled(old_expert)
                ):
                    review_touch_ids.add(i)
                    confirm_events.append(i)
                elif not bool(flag) and old_confirmed:
                    review_touch_ids.add(i)
            _apply_overrides(
                df,
                overrides=overrides,
                reasons=reasons,
                reason_codes=reason_codes,
                confirms=effective_confirms,
            )
            # 动作或确认标签变化后，该年定稿指纹失效。
            if review_touch_ids and "year" in df.columns:
                meta = _read_json(_sp().root / "meta.json") or {}
                fin = set(str(y) for y in (meta.get("finalized_years") or []))
                fps = dict(meta.get("trajectory_fingerprints") or {})
                for i in review_touch_ids:
                    if 0 <= i < len(df):
                        y = str(df.at[i, "year"])
                        fin.discard(y)
                        fps.pop(y, None)
                _sp().update_meta(finalized_years=sorted(fin), trajectory_fingerprints=fps)
            sp = _sp()
            for i in confirm_events:
                sp.append_edit({
                    "date": str(df.at[i, "date"]),
                    "a_expert": expert_action_float(df.at[i, "a_expert"]),
                    "reason": "human_confirmed_current_action",
                    "action": "confirm_day",
                })
            if not override_ids:
                _save_review_df(df)
            sp = _sp()
            for k in overrides:
                i = int(k)
                fresh.add(i)
                if 0 <= i < len(df):
                    sp.append_edit({
                        "date": str(df.at[i, "date"]),
                        "a_expert": expert_action_float(df.at[i, "a_expert"]),
                        "reason": (
                            "同批次更早动作改变状态；该选择已暂存为历史并需重审。 "
                            if i in deferred_override_ids else ""
                        ) + str(reasons.get(k, df.at[i, "reason"] if "reason" in df.columns else "")),
                        "action": "defer_expert_pending_resync" if i in deferred_override_ids else "modify_expert",
                    })
            fresh.update(confirm_events)
        if not override_ids:
            df = _load_review_df()
        else:
            df = resync(
                review=df,
                work=_sp().root,
                fresh_indices=fresh or None,
                override_indices=override_ids,
            )
            # 改动作后旧定稿指纹失效：该年须重新 finalize
            sp = _sp()
            meta = _read_json(sp.root / "meta.json") or {}
            touched = set()
            for i in override_ids:
                if 0 <= i < len(df) and "year" in df.columns:
                    touched.add(str(df.at[i, "year"]))
            if touched:
                fps = dict(meta.get("trajectory_fingerprints") or {})
                fin = [str(y) for y in (meta.get("finalized_years") or []) if str(y) not in touched]
                for y in touched:
                    fps.pop(str(y), None)
                sp.update_meta(finalized_years=fin, trajectory_fingerprints=fps)
        return jsonify({
            "ok": True,
            "skipped_resync": not bool(override_ids),
            "n_days": int(len(df)),
            "n_confirmed": int(df["confirmed"].sum()) if "confirmed" in df.columns else 0,
            "n_needs_reconfirm": int(df["needs_reconfirm"].sum())
            if "needs_reconfirm" in df.columns
            else 0,
            "n_deferred_actions": len(deferred_action_dates),
            "deferred_action_dates": deferred_action_dates,
            "session": _session_status(),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/api/bc")
def api_bc():
    try:
        st = _session_status()
        if not st.get("bc_ready"):
            miss = st.get("missing_review_years") or []
            if int(st.get("n_edit") or 0) <= 0:
                err = "改动日为 0：请先在审阅年改专家动作（自动重算）并「本审阅年定稿」后再 BC"
            elif miss:
                err = f"审阅年未全部确认（缺 {', '.join(miss)}），不能提交 BC"
            elif int(st.get("n_needs_reconfirm") or 0) > 0:
                err = f"仍有 {st['n_needs_reconfirm']} 天待复核，请审阅并重算后再 BC"
            else:
                err = "审阅未就绪，不能提交行为克隆"
            return jsonify({"ok": False, "error": err, "session": st}), 400
        data = request.get_json(silent=True) or {}
        cfg = BCConfig(**BC_DEFAULT.to_dict())
        if data.get("epochs") is not None:
            cfg.epochs = int(data["epochs"])
        if data.get("w_edit") is not None:
            cfg.w_edit = float(data["w_edit"])
        out = bc(
            work=_sp().root,
            bc_cfg=cfg,
            require_no_pending=bool(data.get("require_no_pending", True)),
        )
        return jsonify({"ok": True, "result": out, "session": _session_status()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/api/compare")
def api_compare():
    try:
        if _review_is_blind():
            return jsonify({"ok": False, "error": "专家审阅尚未完成，暂不开放三方结果"}), 423
        data = request.get_json(silent=True) or {}
        max_years = data.get("max_years")
        if max_years is not None and str(max_years).strip() != "":
            max_years = int(max_years)
        else:
            max_years = None
        out = compare_three(_sp().root, max_years=max_years)
        rationale_audit = None
        try:
            from ..expert_rationale_analysis import analyze_expert_rationales

            rationale_audit = analyze_expert_rationales(_sp().root)
        except Exception as e:
            rationale_audit = {"ok": False, "error": str(e)}
        return jsonify({
            "ok": True,
            "result": {
                "session": out.get("session"),
                "metrics": out.get("metrics"),
                "compare": out.get("compare"),
                "expert_rationale_audit": rationale_audit,
            },
            "session": _session_status(),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/api/metrics")
def api_metrics():
    try:
        if _review_is_blind():
            return jsonify({"ok": False, "error": "专家审阅尚未完成，暂不开放吸收率结果"}), 423
        abs_ = compute_absorption(_sp())
        rationale_audit = None
        try:
            from ..expert_rationale_analysis import analyze_expert_rationales

            rationale_audit = analyze_expert_rationales(_sp().root)
        except Exception as e:
            rationale_audit = {"ok": False, "error": str(e)}
        return jsonify({
            "ok": True,
            "absorption": abs_,
            "expert_rationale_audit": rationale_audit,
            "session": _session_status(),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/api/compare/yearly")
def api_compare_yearly():
    """三方对比：多年均值默认；可选单年 KPI（来自 06_compare 各臂 yearly）。"""
    try:
        sp = _sp()
        if _review_is_blind(sp):
            return jsonify({"ok": False, "error": "专家审阅尚未完成，暂不开放三方结果"}), 423
        year = (request.args.get("year") or "").strip()
        rule_dir = sp.compare / "rule"
        ppo_dir = sp.compare / "original_ppo"
        hitl_dir = sp.compare / "hitl_ppo"
        rule_map = _yearly_arm_map(rule_dir, "RULE")
        ppo_map = _yearly_arm_map(ppo_dir, "PPO")
        hitl_map = _yearly_arm_map(hitl_dir, "PPO")
        years = sorted(set(rule_map) | set(ppo_map) | set(hitl_map), key=lambda y: int(y))
        task = _read_json(sp.metrics / "task_compare.json")
        # 各臂 only=rule/ppo 时目录内常无 compare_summary；均值直接从 yearly 算
        means = {
            "rule": _arm_means(rule_dir, "RULE"),
            "original_ppo": _arm_means(ppo_dir, "PPO"),
            "hitl_ppo": _arm_means(hitl_dir, "PPO"),
        }
        pairs = build_yearly_pairs(sp)
        year_roles = {y: year_role(y) for y in years}
        if not year:
            return jsonify({
                "ok": True,
                "mode": "mean",
                "years": years,
                "means": means,
                "task_compare": task,
                "year_row": None,
                "pairs": pairs,
                "year_roles": year_roles,
            })
        return jsonify({
            "ok": True,
            "mode": "year",
            "years": years,
            "year": year,
            "means": means,
            "year_row": {
                "rule": rule_map.get(year),
                "original_ppo": ppo_map.get(year),
                "hitl_ppo": hitl_map.get(year),
            },
            "task_compare": task,
            "pairs": pairs,
            "year_roles": year_roles,
            "year_role": year_role(year),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/api/compare/season")
def api_compare_season():
    """三方整季过程：实测雨 / 水层+阈值 / 执行灌排量。默认适合 2016 验证年。"""
    try:
        sp = _sp()
        if _review_is_blind(sp):
            return jsonify({"ok": False, "error": "专家审阅尚未完成，暂不开放三方结果"}), 423
        year = (request.args.get("year") or "2016").strip()
        gp = resolve_growth_para(sp)
        arms_paths = {
            "rule": sp.compare / "rule" / "policy_output_RULE.csv",
            "original_ppo": sp.compare / "original_ppo" / "policy_output_PPO.csv",
            "hitl_ppo": sp.compare / "hitl_ppo" / "policy_output_PPO.csv",
        }
        if not arms_paths["rule"].exists():
            alt = sp.compare / "rule" / "policy_output_PPO.csv"
            if alt.exists():
                arms_paths["rule"] = alt
        arms = {
            k: season_rows_enriched(p, year, growth_para=gp, include_actual_rain=True)
            for k, p in arms_paths.items()
        }
        pairs = build_yearly_pairs(sp)
        year_row = None
        for p in pairs.get("pairs") or []:
            if str(p.get("year")) == str(year):
                year_row = p
                break
        role = year_role(year)
        edit_days = load_expert_edit_dates(sp, year) if role.get("is_review") else []
        return jsonify({
            "ok": True,
            "year": year,
            "year_role": role,
            "growth_para": str(gp) if gp else None,
            "arms": arms,
            "year_row": year_row,
            "expert_edit_days": edit_days,
            "n_expert_edits": len(edit_days),
            "note": "上：实测雨；中：田面水层与阈值；下：执行量 m 为正、−d 为负；审阅年标专家改动日",
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/api/compare/heatmap")
def api_compare_heatmap():
    """生育期 × 全部评估年：HITL 相对原 PPO 的变化热图。"""
    try:
        sp = _sp()
        if _review_is_blind(sp):
            return jsonify({"ok": False, "error": "专家审阅尚未完成，暂不开放三方结果"}), 423
        metric = (request.args.get("metric") or "irrigation_total").strip()
        out = build_stage_year_heatmap(sp, metric=metric)
        return jsonify(out)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/api/compare/stage")
def api_compare_stage():
    """按生育期汇总 PPO–HITL（单年或留出年均值）。"""
    try:
        sp = _sp()
        if _review_is_blind(sp):
            return jsonify({"ok": False, "error": "专家审阅尚未完成，暂不开放三方结果"}), 423
        scope = (request.args.get("scope") or "year").strip()
        year = (request.args.get("year") or "2016").strip()
        if scope == "holdout":
            from ..config import HITL_HOLDOUT_YEARS

            out = build_stage_compare(sp, years=list(HITL_HOLDOUT_YEARS))
            out["scope"] = "holdout"
            out["scope_label"] = "留出年 " + ",".join(HITL_HOLDOUT_YEARS) + " 均值"
        else:
            out = build_stage_compare(sp, year=year)
            out["scope"] = "year"
            out["scope_label"] = f"{year}（{year_role(year)['label']}）"
            out["year_role"] = year_role(year)
        return jsonify(out)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/api/compare/absorption_probs")
def api_compare_absorption_probs():
    """专家改动日 / 确认日：原 PPO 与 HITL 对目标动作的概率配对。"""
    try:
        sp = _sp()
        if _review_is_blind(sp):
            return jsonify({"ok": False, "error": "专家审阅尚未完成，暂不开放吸收诊断"}), 423
        force = (request.args.get("force") or "").strip() in ("1", "true", "yes")
        out = compute_absorption_probs(sp, force=force)
        return jsonify(out)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/api/session/download_zip")
def api_session_download_zip():
    try:
        sp = _sp()
        if _review_is_blind(sp):
            return jsonify({"ok": False, "error": "专家审阅尚未完成，暂不开放会话打包下载"}), 423
        buf = _zip_directory(sp.root, arc_prefix=sp.root.name)
        return send_file(
            buf,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"{sp.root.name}.zip",
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/download/<name>")
def download(name: str):
    try:
        sp = _sp()
    except RuntimeError:
        flash("无当前会话", "error")
        return redirect(url_for("index", tab="expert"))
    if _review_is_blind(sp):
        return jsonify({"ok": False, "error": "专家审阅尚未完成，暂不开放会话文件下载"}), 423
    allowed = {
        "review.csv": sp.review_csv,
        "review.json": sp.review_json,
        "labels.json": sp.labels,
        "ppo_hitl.pth": sp.hitl_model,
        "hitl_result.json": sp.bc / "hitl_result.json",
        "three_way.json": sp.compare / "three_way.json",
        "task_compare.json": sp.metrics / "task_compare.json",
        "absorption.json": sp.metrics / "absorption.json",
    }
    path = allowed.get(name)
    if path is None or not path.exists():
        flash(f"文件不存在: {name}", "error")
        return redirect(url_for("index", tab="expert"))
    return send_file(path, as_attachment=True)


def main(host: str = "127.0.0.1", port: int = 5050, debug: bool = False) -> None:
    global _RUN, _SESSION
    _RUN = None
    _SESSION = None
    print(f"HITL 看板 → http://{host}:{port}")
    print("  流程：训练会话→上传/校验→训评→人工审阅(改动作↔一致性重算)→行为克隆(BC)→三方对比")
    print("  产物：output/runs/  |  output/hitl_sessions/")
    print(f"可用 run: {[r['run_id'] for r in list_runs(require_best=False, limit=8)]}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    main()
