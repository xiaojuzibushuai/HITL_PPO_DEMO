"""专家结构化改因的事后核查（等雨 / 允许偏干分开；不合成「成功率」）。

仅在审阅定稿后、用实际气象做事后评价。审阅标签必须来自 ``reason_code``，
禁止仅凭「预报有雨且当天未灌」推断等雨意图。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import P_CUMU7_WINDOW, REASON_CODE_LABELS
from .exec_layer import expert_action_filled, expert_action_float, norm_date
from .session import SessionPaths, resolve_session


def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default


def _load_actual_rain(sp: SessionPaths) -> pd.DataFrame:
    """事后用实测雨；优先会话 workdir / uploads / 一期 eval。"""
    cols = list(P_CUMU7_WINDOW["actual_cols"])
    candidates = [
        sp.workdir / "actual_eval.csv",
        sp.workdir / "actual.csv",
        sp.root / "uploads" / "actual_eval.csv",
        sp.root / "uploads" / "actual.csv",
        sp.phase1_eval / "_workdir" / "actual_eval.csv",
        sp.root / "02_eval" / "_workdir" / "actual_eval.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "date" not in df.columns:
            continue
        have = [c for c in cols if c in df.columns]
        if len(have) < 7 and "P_1_a" not in df.columns:
            continue
        out = df.copy()
        out["_d"] = out["date"].map(norm_date)
        for c in cols:
            if c not in out.columns:
                out[c] = 0.0
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        out["actual_rain_7d"] = out[cols].sum(axis=1)
        return out[["_d", "P_1_a", "actual_rain_7d"] + cols]
    return pd.DataFrame()


def _merge_wait_events(dates: list[pd.Timestamp]) -> list[int]:
    """重叠/相邻等待日合并为事件 id（按时间序）。返回与 dates 等长的 event_id。"""
    if not dates:
        return []
    order = sorted(range(len(dates)), key=lambda i: dates[i])
    event_ids = [-1] * len(dates)
    eid = 0
    prev: pd.Timestamp | None = None
    for i in order:
        d = dates[i]
        if prev is None or int((d.normalize() - prev.normalize()).days) > 1:
            eid += 1
        event_ids[i] = eid
        prev = d
    return event_ids


def analyze_expert_rationales(
    work: str | Path | None = None,
    *,
    review: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """写出 ``07_metrics/expert_rationale_audit.json``（三项分列，无合成成功率）。"""
    root = resolve_session(work, create=False)
    sp = SessionPaths(root).ensure()
    if review is None:
        path = sp.review_json if sp.review_json.exists() else sp.review_csv
        if not path.exists():
            raise FileNotFoundError(f"无审阅表: {sp.review_csv}")
        loaded = pd.read_json(path) if path.suffix.lower() == ".json" else pd.read_csv(path)
        if loaded is None or not isinstance(loaded, pd.DataFrame):
            raise FileNotFoundError(f"审阅表无法解析为 DataFrame: {path}")
        review = loaded
    df = review.copy()
    if "reason_code" not in df.columns:
        df["reason_code"] = ""
    df["reason_code"] = df["reason_code"].fillna("").astype(str).str.strip()
    df["_dt"] = pd.to_datetime(df["date"], errors="coerce")
    actual = _load_actual_rain(sp)
    actual_map = {}
    if len(actual):
        for _, r in actual.iterrows():
            actual_map[str(r["_d"])] = r

    # —— 1) 依据预报暂缓灌水 ——
    wait_rows = []
    for _, r in df.iterrows():
        if r["reason_code"] != "wait_forecast_rain":
            continue
        if "editable" in r and not bool(r.get("editable", True)):
            continue
        a_model = _f(r.get("a_model"))
        a_exp = expert_action_float(r.get("a_expert"))
        if a_exp is None:
            continue
        # 仅统计「相对 PPO 减灌或不灌」的标注日（正灌方向）
        reduced = a_model > 0.01 and a_exp < a_model - 1e-6
        dkey = norm_date(r.get("date"))
        act = actual_map.get(str(dkey))
        p_fc = _f(r.get("P_cumu7"))
        p_act7 = _f(act["actual_rain_7d"]) if act is not None else float("nan")
        wait_rows.append({
            "date": str(r.get("date")),
            "year": str(r.get("year")),
            "stage": str(r.get("stage") or ""),
            "h_before": _f(r.get("h_before")),
            "hmin": _f(r.get("hmin")),
            "P_cumu7": p_fc,
            "a_model": a_model,
            "a_expert": float(a_exp),
            "irrigation_reduced": bool(reduced),
            "delta_irrigation_mm": float(a_exp - a_model) if reduced else 0.0,
            "actual_rain_7d": p_act7,
            "forecast_minus_actual_7d": (p_fc - p_act7) if np.isfinite(p_act7) else None,
            "rain_realized_mm": p_act7 if np.isfinite(p_act7) else None,
        })

    wait_df = pd.DataFrame(wait_rows)
    wait_event_n = 0
    if len(wait_df):
        dts = list(pd.to_datetime(wait_df["date"], errors="coerce"))
        eids = _merge_wait_events(dts)
        wait_df["wait_event_id"] = eids
        wait_event_n = int(wait_df["wait_event_id"].nunique())

    wait_summary = {
        "n_labeled_days": int(len(wait_df)),
        "n_wait_events_merged": wait_event_n,
        "n_days_ppo_irrigation_reduced": int(wait_df["irrigation_reduced"].sum()) if len(wait_df) else 0,
        "total_irrigation_reduced_mm": float(
            (-wait_df.loc[wait_df["irrigation_reduced"], "delta_irrigation_mm"]).sum()
        ) if len(wait_df) else 0.0,
        "mean_P_cumu7": float(wait_df["P_cumu7"].mean()) if len(wait_df) else None,
        "mean_actual_rain_7d": float(wait_df["actual_rain_7d"].mean())
        if len(wait_df) and wait_df["actual_rain_7d"].notna().any()
        else None,
        "p_cumu7_window": dict(P_CUMU7_WINDOW),
        "report_separately": [
            "专家是否减少了 PPO 灌水（天数、水量）",
            "预报雨是否兑现（P_cumu7 vs 时窗实测合计，含决策日）",
            "等待期间胁迫与模拟产量响应（见 allow_low / 全季 metrics，不在此合成成功率）",
        ],
        "days": wait_rows,
    }

    # —— 2) 允许较低水层 ——
    low_rows = []
    for _, r in df.iterrows():
        if r["reason_code"] != "allow_low_water":
            continue
        if "editable" in r and not bool(r.get("editable", True)):
            continue
        h = _f(r.get("h_before"))
        hmin = _f(r.get("hmin"))
        below = max(0.0, hmin - h)
        low_rows.append({
            "date": str(r.get("date")),
            "year": str(r.get("year")),
            "stage": str(r.get("stage") or ""),
            "lamda": _f(r.get("lamda"), 0.0),
            "h_before": h,
            "hmin": hmin,
            "below_hmin_mm": below,
            "a_model": _f(r.get("a_model")),
            "a_expert": expert_action_float(r.get("a_expert")),
            "P_cumu7": _f(r.get("P_cumu7")),
        })

    # 标注日后短期轨迹：同季按日期向后看低于 hmin 的持续（用审阅表递推水深）
    df_sorted = df.dropna(subset=["_dt"]).sort_values(["year", "_dt"], kind="mergesort")
    by_year = {str(y): g.reset_index(drop=True) for y, g in df_sorted.groupby(df_sorted["year"].astype(str))}
    for row in low_rows:
        g = by_year.get(row["year"])
        if g is None or g.empty:
            row.update({"post_below_hmin_days": None, "post_min_h": None, "post_stress_days_ks": None})
            continue
        dts = pd.to_datetime(g["date"], errors="coerce")
        start = pd.to_datetime(row["date"], errors="coerce")
        # 从决策日起连续段内统计（到下一次标注或季末前最多 14 日窗口作描述）
        mask = dts >= start
        sub = g.loc[mask].head(14)
        hs = pd.to_numeric(sub.get("h_before", 0.0), errors="coerce").fillna(0.0)
        hmins = pd.to_numeric(sub.get("hmin", 0.0), errors="coerce").fillna(0.0)
        below_flags = (hmins - hs) > 0.01
        # 连续低于 hmin 天数（从决策日算起）
        run = 0
        for v in below_flags.tolist():
            if v:
                run += 1
            else:
                break
        # Ks 胁迫：若有 ks 列
        stress = 0
        if "ks" in sub.columns:
            ks = pd.to_numeric(sub["ks"], errors="coerce")
            stress = int((ks < 1.0 - 1e-6).sum())
        row["post_below_hmin_days"] = int(run)
        row["post_min_h"] = float(hs.min()) if len(hs) else None
        row["post_stress_days_ks"] = stress

    low_summary = {
        "n_labeled_days": int(len(low_rows)),
        "mean_below_hmin_mm": float(np.mean([r["below_hmin_mm"] for r in low_rows])) if low_rows else None,
        "note": (
            "低于 hmin 是管理水层指标，不自动等于作物受旱；"
            "胁迫看 ks/Ks 等模拟响应；相对产量见三方 yearly yield_factor，勿写成田间实测产量"
        ),
        "days": low_rows,
    }

    # —— 3) 合并小额（标注计数，段统计仍以 operation_runs 为准）——
    merge_n = int((df["reason_code"] == "merge_small_irrigation").sum())

    code_counts = {
        code: int((df["reason_code"] == code).sum()) for code in REASON_CODE_LABELS
    }
    out = {
        "ok": True,
        "session": str(sp.root),
        "reason_code_counts": code_counts,
        "labels": REASON_CODE_LABELS,
        "wait_forecast_rain": wait_summary,
        "allow_low_water": low_summary,
        "merge_small_irrigation_labeled_days": merge_n,
        "method_notes": [
            "等雨与允许偏干分开统计，不合成成功率",
            "等雨仅统计 reason_code=wait_forecast_rain 的显式标注日",
            "等待事件：相邻标注日合并，同时报告决策日数与事件数",
            "P_cumu7 时窗含决策日，与 actual.P_1_a…P_7_a 对齐",
        ],
    }
    sp.metrics.mkdir(parents=True, exist_ok=True)
    path = sp.metrics / "expert_rationale_audit.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    out["path"] = str(path)
    return out
