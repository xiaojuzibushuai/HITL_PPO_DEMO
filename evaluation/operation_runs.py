"""连续小额灌/排操作段统计（论文口径）。

主指标（连续小额灌溉段）
  - 实际灌水量满足 0 < m ≤ max_mm，日历连续，长度 ≥ 2 d
  - 分别汇总长度=2 与长度≥3 的段数、天数、水量

补充指标（连续小额续灌段）
  - 首日 m > max_mm，后续连续日均为 0 < m ≤ max_mm（至少 1 个后续小额日）
  - 识别如 60、20、20 mm；全日小额段只计入主指标

排水：连续小额排水段单独统计（0 < d ≤ max_mm，≥2 d），不与灌水合并。

中断条件：零操作、日期不连续、年份边界、固定规则接管日（editable=False / used_fixed_rule）。
统计一律用实际执行量 m/d（或由 a_exec 拆分），不用模型提案。
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

# 研究判据（动作档 10 mm 步长下的操作性定义，非普适农艺阈值）
DEFAULT_SMALL_MM = 20.0
DEFAULT_MIN_MAIN_DAYS = 2


def _as_float_series(s: Any, n: int) -> np.ndarray:
    if s is None:
        return np.zeros(n, dtype=float)
    arr = pd.to_numeric(pd.Series(s), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(arr) != n:
        raise ValueError(f"length mismatch: expected {n}, got {len(arr)}")
    return arr


def _break_mask(
    n: int,
    *,
    editable: Sequence[bool] | None = None,
    used_fixed_rule: Sequence[bool] | None = None,
) -> np.ndarray:
    """True = 该日打断连续段（固定规则/不可编辑）。"""
    br = np.zeros(n, dtype=bool)
    if editable is not None:
        ed = np.asarray(list(editable), dtype=bool)
        if len(ed) != n:
            raise ValueError("editable length mismatch")
        br |= ~ed
    if used_fixed_rule is not None:
        fx = np.asarray(list(used_fixed_rule), dtype=bool)
        if len(fx) != n:
            raise ValueError("used_fixed_rule length mismatch")
        br |= fx
    return br


def _dates_consecutive(d0: pd.Timestamp, d1: pd.Timestamp) -> bool:
    if pd.isna(d0) or pd.isna(d1):
        return False
    return int((d1.normalize() - d0.normalize()).days) == 1


def _empty_bucket() -> dict[str, float]:
    return {"segments": 0, "days": 0, "volume_mm": 0.0}


def _add_segment(bucket: dict[str, float], *, length: int, volume: float) -> None:
    bucket["segments"] += 1
    bucket["days"] += int(length)
    bucket["volume_mm"] += float(volume)


def _summarize_all_small_runs(
    amounts: np.ndarray,
    dates: Sequence[pd.Timestamp],
    breaks: np.ndarray,
    *,
    max_mm: float,
    min_days: int,
) -> tuple[dict[str, float], dict[str, float], list[bool]]:
    """全段均为 (0, max_mm] 的连续段 → 分 2d / ≥3d；并返回逐日是否落在 ≥min_days 段内。"""
    n = len(amounts)
    flags = [False] * n
    two = _empty_bucket()
    ge3 = _empty_bucket()
    i = 0
    while i < n:
        if breaks[i] or not (0.0 < float(amounts[i]) <= max_mm + 1e-9):
            i += 1
            continue
        j = i
        vol = float(amounts[i])
        while j + 1 < n:
            if breaks[j + 1]:
                break
            if not _dates_consecutive(dates[j], dates[j + 1]):
                break
            a = float(amounts[j + 1])
            if not (0.0 < a <= max_mm + 1e-9):
                break
            j += 1
            vol += a
        length = j - i + 1
        if length >= min_days:
            for k in range(i, j + 1):
                flags[k] = True
            if length == 2:
                _add_segment(two, length=length, volume=vol)
            else:
                _add_segment(ge3, length=length, volume=vol)
        i = j + 1
    return two, ge3, flags


def _summarize_topup_runs(
    amounts: np.ndarray,
    dates: Sequence[pd.Timestamp],
    breaks: np.ndarray,
    *,
    max_mm: float,
) -> tuple[dict[str, float], list[bool]]:
    """续灌：首日灌水量 > max_mm，其后至少一日为 (0, max_mm] 的连续小额。

    与「全日小额」主指标互补（例 60、20、20）；全日 ≤max_mm 的段只计入主指标。
    高亮：续灌段内后续小额日。
    """
    n = len(amounts)
    flags = [False] * n
    bucket = _empty_bucket()
    i = 0
    while i < n:
        a0 = float(amounts[i])
        # 首日须大于小额上限，才是「先灌一笔再小额续灌」
        if breaks[i] or not (a0 > max_mm + 1e-9):
            i += 1
            continue
        if i + 1 >= n or breaks[i + 1] or not _dates_consecutive(dates[i], dates[i + 1]):
            i += 1
            continue
        a1 = float(amounts[i + 1])
        if not (0.0 < a1 <= max_mm + 1e-9):
            i += 1
            continue
        j = i + 1
        vol = a0 + a1
        while j + 1 < n:
            if breaks[j + 1]:
                break
            if not _dates_consecutive(dates[j], dates[j + 1]):
                break
            a = float(amounts[j + 1])
            if not (0.0 < a <= max_mm + 1e-9):
                break
            j += 1
            vol += a
        length = j - i + 1
        _add_segment(bucket, length=length, volume=vol)
        for k in range(i + 1, j + 1):
            flags[k] = True
        i = j + 1
    return bucket, flags


def summarize_operation_runs(
    df: pd.DataFrame,
    *,
    max_mm: float = DEFAULT_SMALL_MM,
    min_main_days: int = DEFAULT_MIN_MAIN_DAYS,
) -> dict[str, Any]:
    """对已按时间排序的逐日表汇总一年（或一季）操作段。"""
    if df is None or len(df) == 0:
        return {
            "criterion": {
                "small_mm": max_mm,
                "min_main_days": min_main_days,
                "note": "研究判据；执行量 m/d；灌排分开",
            },
            "irrigation_small_runs_2d": _empty_bucket(),
            "irrigation_small_runs_ge3d": _empty_bucket(),
            "irrigation_topup_small_runs": _empty_bucket(),
            "drainage_small_runs_2d": _empty_bucket(),
            "drainage_small_runs_ge3d": _empty_bucket(),
            "irrigation_operation_days": 0,
            "drainage_operation_days": 0,
            "irrigation_contiguous_segments": 0,
            "drainage_contiguous_segments": 0,
        }

    x = df.copy()
    x["_dt"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.loc[x["_dt"].notna()].sort_values("_dt", kind="mergesort").reset_index(drop=True)
    n = len(x)
    if "m" not in x.columns and "a_exec" in x.columns:
        ae = pd.to_numeric(x["a_exec"], errors="coerce").fillna(0.0)
        x["m"] = ae.clip(lower=0.0)
        x["d"] = (-ae).clip(lower=0.0)
    m = _as_float_series(x.get("m", 0.0), n)
    d = _as_float_series(x.get("d", 0.0), n)
    dates = list(x["_dt"])
    editable = x["editable"].tolist() if "editable" in x.columns else None
    fixed = x["used_fixed_rule"].tolist() if "used_fixed_rule" in x.columns else None
    br = _break_mask(n, editable=editable, used_fixed_rule=fixed)

    irr2, irr3, _ = _summarize_all_small_runs(
        m, dates, br, max_mm=max_mm, min_days=min_main_days
    )
    topup, _ = _summarize_topup_runs(m, dates, br, max_mm=max_mm)
    dr2, dr3, _ = _summarize_all_small_runs(
        d, dates, br, max_mm=max_mm, min_days=min_main_days
    )

    def _count_op_segments(amounts: np.ndarray) -> tuple[int, int]:
        """操作日数 + 任意正量的连续段数（日历连续、遇 break 中断）。"""
        days = int(np.sum(amounts > 0.01))
        segs = 0
        i = 0
        while i < n:
            if br[i] or amounts[i] <= 0.01:
                i += 1
                continue
            segs += 1
            j = i
            while j + 1 < n:
                if br[j + 1] or amounts[j + 1] <= 0.01:
                    break
                if not _dates_consecutive(dates[j], dates[j + 1]):
                    break
                j += 1
            i = j + 1
        return days, segs

    irr_days, irr_segs = _count_op_segments(m)
    dr_days, dr_segs = _count_op_segments(d)

    return {
        "criterion": {
            "small_mm": float(max_mm),
            "min_main_days": int(min_main_days),
            "source": "executed_m_d",
            "note": (
                "主：连续小额灌溉/排水段（全日 ≤small_mm，≥min_main_days）；"
                "补：续灌段（首日任意正灌，后续小额）；灌排分开"
            ),
        },
        "irrigation_small_runs_2d": irr2,
        "irrigation_small_runs_ge3d": irr3,
        "irrigation_topup_small_runs": topup,
        "drainage_small_runs_2d": dr2,
        "drainage_small_runs_ge3d": dr3,
        "irrigation_operation_days": irr_days,
        "drainage_operation_days": dr_days,
        "irrigation_contiguous_segments": irr_segs,
        "drainage_contiguous_segments": dr_segs,
    }


def yearly_operation_metric_row(summary: dict[str, Any]) -> dict[str, float]:
    """压平为 yearly_metrics 可写列。"""
    def _g(key: str, field: str) -> float:
        return float((summary.get(key) or {}).get(field) or 0)

    return {
        "irrigation_operation_days": float(summary.get("irrigation_operation_days") or 0),
        "drainage_operation_days": float(summary.get("drainage_operation_days") or 0),
        "irrigation_contiguous_segments": float(summary.get("irrigation_contiguous_segments") or 0),
        "drainage_contiguous_segments": float(summary.get("drainage_contiguous_segments") or 0),
        "irrigation_small_run_2d_segments": _g("irrigation_small_runs_2d", "segments"),
        "irrigation_small_run_2d_days": _g("irrigation_small_runs_2d", "days"),
        "irrigation_small_run_2d_mm": _g("irrigation_small_runs_2d", "volume_mm"),
        "irrigation_small_run_ge3d_segments": _g("irrigation_small_runs_ge3d", "segments"),
        "irrigation_small_run_ge3d_days": _g("irrigation_small_runs_ge3d", "days"),
        "irrigation_small_run_ge3d_mm": _g("irrigation_small_runs_ge3d", "volume_mm"),
        "irrigation_topup_small_segments": _g("irrigation_topup_small_runs", "segments"),
        "irrigation_topup_small_days": _g("irrigation_topup_small_runs", "days"),
        "irrigation_topup_small_mm": _g("irrigation_topup_small_runs", "volume_mm"),
        "drainage_small_run_2d_segments": _g("drainage_small_runs_2d", "segments"),
        "drainage_small_run_2d_days": _g("drainage_small_runs_2d", "days"),
        "drainage_small_run_2d_mm": _g("drainage_small_runs_2d", "volume_mm"),
        "drainage_small_run_ge3d_segments": _g("drainage_small_runs_ge3d", "segments"),
        "drainage_small_run_ge3d_days": _g("drainage_small_runs_ge3d", "days"),
        "drainage_small_run_ge3d_mm": _g("drainage_small_runs_ge3d", "volume_mm"),
    }


def mark_small_irrigation_review_days(
    irrig_mm: Iterable[float],
    *,
    dates: Iterable[Any] | None = None,
    editable: Iterable[bool] | None = None,
    used_fixed_rule: Iterable[bool] | None = None,
    max_mm: float = DEFAULT_SMALL_MM,
    min_main_days: int = DEFAULT_MIN_MAIN_DAYS,
) -> list[bool]:
    """审阅高亮：落在主指标小额灌溉段，或落在续灌段的后续小额日。"""
    m = np.asarray(list(irrig_mm), dtype=float)
    n = len(m)
    if n == 0:
        return []
    if dates is None:
        # 无日期时退化为索引相邻即连续（审阅表同行序）
        dates_ts = [pd.Timestamp("2000-01-01") + pd.Timedelta(days=i) for i in range(n)]
    else:
        dates_ts = list(pd.to_datetime(pd.Series(list(dates)), errors="coerce"))
    br = _break_mask(
        n,
        editable=list(editable) if editable is not None else None,
        used_fixed_rule=list(used_fixed_rule) if used_fixed_rule is not None else None,
    )
    _, _, main_flags = _summarize_all_small_runs(
        m, dates_ts, br, max_mm=max_mm, min_days=min_main_days
    )
    _, topup_flags = _summarize_topup_runs(m, dates_ts, br, max_mm=max_mm)
    return [bool(a or b) for a, b in zip(main_flags, topup_flags)]
