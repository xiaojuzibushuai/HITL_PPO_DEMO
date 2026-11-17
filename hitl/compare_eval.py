"""自主评价：规则 / 原 PPO / HITL（行为克隆后）→ ``06_compare`` + ``07_metrics``。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from common.scenario import set_active_scenario
from evaluation.metrics import compare_summary

from .config import resolve_year_split
from .metrics_report import compute_absorption, timed_stage
from .session import SessionPaths, resolve_session


def _arm_yearly_df(arm_dir: Path, prefer: str) -> pd.DataFrame | None:
    """prefer: PPO|RULE — 取对应 yearly_metrics。"""
    p = arm_dir / f"yearly_metrics_{prefer}.csv"
    if p.exists():
        return pd.read_csv(p)
    cands = list(arm_dir.glob("yearly_metrics_*.csv"))
    return pd.read_csv(cands[0]) if cands else None


def _holdout_task_slice(rule_dir: Path, orig_dir: Path, hitl_dir: Path, holdout: list[str]) -> dict[str, Any]:
    """从全评估年明细中裁剪留出验证年，再算 PPO/HITL 相对规则的 Δ%。"""
    if not holdout:
        return {"ok": False, "error": "无留出验证年"}
    years = {str(y) for y in holdout}
    out: dict[str, Any] = {"ok": True, "holdout_years": list(holdout), "delta_vs_rule": {}}

    def _filt(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None or df.empty or "year" not in df.columns:
            return None
        sub = df[df["year"].astype(str).isin(years)]
        return sub if len(sub) else None

    rule_y = _filt(_arm_yearly_df(rule_dir, "RULE"))
    ppo_y = _filt(_arm_yearly_df(orig_dir, "PPO"))
    hitl_y = _filt(_arm_yearly_df(hitl_dir, "PPO"))
    if rule_y is not None and ppo_y is not None:
        out["original_ppo_vs_rule"] = compare_summary(ppo_y, rule_y)
    if rule_y is not None and hitl_y is not None:
        out["hitl_ppo_vs_rule"] = compare_summary(hitl_y, rule_y)
    return out


def compare_three(
    work: str | Path | None = None,
    *,
    scenario: str | None = None,
    max_years: Optional[int] = None,
    original_model: str | Path | None = None,
    hitl_model: str | Path | None = None,
) -> dict[str, Any]:
    """对规则、原 PPO、HITL（行为克隆后）各跑一遍评估。

    默认 ``max_years=None`` → **评估集全部年份**（与一期二方 ``02_eval`` 对齐）。
    另写入 ``holdout_compare``：仅留出验证年子集的任务 Δ%（专家默认未改这些年）。

    写入 ``06_compare/{rule,original_ppo,hitl_ppo}/``，汇总 ``07_metrics/task_compare.json``。
    """
    from api import evaluate

    root = resolve_session(work, create=False)
    sp = SessionPaths(root).ensure()
    rule_dir = sp.compare / "rule"
    orig_dir = sp.compare / "original_ppo"
    hitl_dir = sp.compare / "hitl_ppo"

    orig = Path(original_model) if original_model else sp.model
    if not orig.exists():
        raise FileNotFoundError(f"找不到原 PPO（会话 01_train）: {orig}")

    hitl = Path(hitl_model) if hitl_model else sp.hitl_model
    if not hitl.exists():
        raise FileNotFoundError(f"找不到 HITL 模型（先行为克隆 bc）: {hitl}")

    set_active_scenario(scenario)
    meta = {}
    try:
        meta = json.loads((sp.root / "meta.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    split = (meta.get("year_split") if isinstance(meta.get("year_split"), dict) else None) or resolve_year_split()

    out: dict[str, Any] = {
        "session": str(root),
        "original_model": str(orig),
        "hitl_model": str(hitl),
        "arms": {},
        "year_policy": {
            "two_way_note": "一期二方对比始终用评估集全部年份",
            "three_way_scope": "全部评估年" if max_years is None else f"前 {max_years} 年",
            "holdout_years": split.get("holdout_years"),
            "review_years": split.get("review_years"),
        },
    }
    with timed_stage(sp, "compare_three", max_years=max_years):
        out["arms"]["rule"] = evaluate(
            None, out_dir=rule_dir, only="rule", max_years=max_years, scenario=scenario
        )
        out["arms"]["original_ppo"] = evaluate(
            orig, out_dir=orig_dir, only="ppo", max_years=max_years, scenario=scenario
        )
        out["arms"]["hitl_ppo"] = evaluate(
            hitl, out_dir=hitl_dir, only="ppo", max_years=max_years, scenario=scenario
        )

    metrics: dict[str, Any] = {
        "arms": {},
        "scope": out["year_policy"],
        "note": (
            "主表=全评估年三方（可与一期二方对齐）；"
            "holdout_compare=仅留出验证年，衡量行为克隆后在未见年的泛化"
        ),
    }
    def _arm_means_local(arm_dir: Path, prefer: str) -> dict[str, float]:
        """各臂 only=ppo/rule 时无双边 compare_summary，从 yearly 算均值供看板。"""
        keys = (
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
        p = arm_dir / f"yearly_metrics_{prefer}.csv"
        path = p if p.exists() else (next(iter(arm_dir.glob("yearly_metrics_*.csv")), None))
        if path is None or not Path(path).exists():
            return {}
        try:
            df = pd.read_csv(path)
        except Exception:
            return {}
        out: dict[str, float] = {}
        for k in keys:
            if k not in df.columns:
                continue
            s = pd.to_numeric(df[k], errors="coerce").dropna()
            if len(s):
                out[k] = float(s.mean())
        if len(df):
            out["n_years"] = float(len(df))
        return out

    for name, d, prefer in (
        ("rule", rule_dir, "RULE"),
        ("original_ppo", orig_dir, "PPO"),
        ("hitl_ppo", hitl_dir, "PPO"),
    ):
        summary = Path(d) / "compare_summary.json"
        yearly = list(Path(d).glob("yearly_metrics_*.csv"))
        arm: dict[str, Any] = {"dir": str(d), "means": _arm_means_local(d, prefer)}
        if summary.exists():
            arm["summary"] = json.loads(summary.read_text(encoding="utf-8"))
        if yearly:
            arm["yearly_metrics"] = str(yearly[0])
        metrics["arms"][name] = arm

    holdout = _holdout_task_slice(rule_dir, orig_dir, hitl_dir, list(split.get("holdout_years") or []))
    metrics["holdout_compare"] = holdout
    (sp.metrics / "holdout_compare.json").write_text(
        json.dumps(holdout, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metrics_path = sp.metrics / "task_compare.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    out["metrics"] = str(metrics_path)
    out["holdout_compare"] = holdout
    out["absorption"] = compute_absorption(sp, hitl_model=hitl)

    compare_path = sp.compare / "three_way.json"
    slim = {
        "session": out["session"],
        "original_model": out["original_model"],
        "hitl_model": out["hitl_model"],
        "metrics": out["metrics"],
        "year_policy": out["year_policy"],
        "dirs": {"rule": str(rule_dir), "original_ppo": str(orig_dir), "hitl_ppo": str(hitl_dir)},
    }
    compare_path.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    out["compare"] = str(compare_path)
    sp.update_meta(compare_three=slim)
    return out
