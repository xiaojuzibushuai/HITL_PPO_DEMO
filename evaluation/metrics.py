"""评估指标（冻结表）
唯一计算入口；报告/看板只读本模块输出。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common.scenario import get_scenario, ks_factor


def _default_raw_taw() -> tuple[float, float]:
    """每次取当前情景，避免 import 时冻死旧 scenario。"""
    sc = get_scenario()
    return sc.raw, sc.taw


def _max_run_length(flags: np.ndarray) -> int:
    best = cur = 0
    for v in flags.astype(bool):
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _count_events(flags: np.ndarray) -> int:
    """连续 True 段的个数。"""
    f = flags.astype(bool)
    if len(f) == 0:
        return 0
    prev = False
    n = 0
    for v in f:
        if v and not prev:
            n += 1
        prev = bool(v)
    return int(n)


def compute_yearly_metrics(
    df: pd.DataFrame,
    *,
    raw: float | None = None,
    taw: float | None = None,
) -> pd.DataFrame:
    """按年汇总。要求逐日表至少含 date, m/d 或 a_exec, P_1_a, et0_a, lamda, Hc, h_before, h_after。

    raw/taw 缺省取当前 scenario（勿在模块 import 时写死）。
    """
    if raw is None or taw is None:
        dr, dt = _default_raw_taw()
        raw = dr if raw is None else raw
        taw = dt if taw is None else taw
    x = df.copy()
    x["date_dt"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.loc[x["date_dt"].notna()].copy()
    x["year"] = x["date_dt"].dt.year.astype(int)

    x["P_1_a"] = pd.to_numeric(x.get("P_1_a", 0.0), errors="coerce").fillna(0.0)
    x["et0_a"] = pd.to_numeric(x.get("et0_a", 0.0), errors="coerce").fillna(0.0)
    if "m" not in x.columns and "a_exec" in x.columns:
        ae = pd.to_numeric(x["a_exec"], errors="coerce").fillna(0.0)
        x["m"] = ae.clip(lower=0.0)
        x["d"] = (-ae).clip(lower=0.0)
    x["m"] = pd.to_numeric(x.get("m", 0.0), errors="coerce").fillna(0.0)
    x["d"] = pd.to_numeric(x.get("d", 0.0), errors="coerce").fillna(0.0)
    x["lamda"] = pd.to_numeric(x.get("lamda", 1.0), errors="coerce")
    x["Hc"] = pd.to_numeric(x.get("Hc", np.nan), errors="coerce")
    x["h_before"] = pd.to_numeric(x.get("h_before", 0.0), errors="coerce").fillna(0.0)
    x["h_after"] = pd.to_numeric(x.get("h_after", x["h_before"]), errors="coerce").fillna(0.0)

    # 执行后水位近似：h + a_exec（若有）；否则用 h_before
    if "a_exec" in x.columns:
        a_exec = pd.to_numeric(x["a_exec"], errors="coerce").fillna(0.0)
        h_ctrl = x["h_before"] + a_exec
    else:
        h_ctrl = x["h_before"] + x["m"] - x["d"]
    x["ks_eff"] = [ks_factor(float(h), raw, taw) for h in h_ctrl]
    x["ETc_actual"] = x["ks_eff"] * x["et0_a"]

    x["irrig_event"] = (x["m"] > 0.01).astype(int)
    x["drain_event"] = (x["d"] > 0.01).astype(int)
    x["stress_day"] = (x["ks_eff"] < 1.0 - 1e-6).astype(int)
    x["stress_day_lam"] = ((x["ks_eff"] < 1.0 - 1e-6) & (x["lamda"].fillna(0) > 0)).astype(int)
    x["over_hc"] = ((x["Hc"].notna()) & (x["h_after"] - x["Hc"] > 0.01)).astype(int)
    x["over_depth"] = np.where(x["Hc"].notna(), np.maximum(x["h_after"] - x["Hc"], 0.0), 0.0)
    x["no_drain_high"] = (
        (x["Hc"].notna()) & (x["h_before"] >= x["Hc"]) & (x["d"] <= 0.01)
    ).astype(int)

    rows: list[dict[str, Any]] = []
    for year, g in x.groupby("year", sort=True):
        rain_total = float(g["P_1_a"].sum())
        drain_total = float(g["d"].sum())
        irrig_total = float(g["m"].sum())
        rain_util = (rain_total - drain_total) / rain_total if rain_total > 1e-9 else float("nan")

        # 产量系数：按 (连续段, lamda) 分组更稳；简化按 lamda 四舍五入
        yield_factor = 1.0
        gg = g.dropna(subset=["lamda"]).copy()
        gg["lamda_round"] = gg["lamda"].round(6)
        stage_seq = 0
        prev_lam = None
        gg = gg.reset_index(drop=True)
        keys = []
        for _, row in gg.iterrows():
            lam = float(row["lamda_round"])
            if prev_lam is None or lam != prev_lam:
                stage_seq += 1
                prev_lam = lam
            keys.append((stage_seq, lam))
        gg["_stage"] = keys
        for (_, lam), sg in gg.groupby("_stage", sort=True):
            et0_sum = float(sg["et0_a"].sum())
            etc_sum = float(sg["ETc_actual"].sum())
            if et0_sum <= 0:
                continue
            ratio = max(0.0, min(1.0, etc_sum / et0_sum))
            yield_factor *= ratio ** float(lam)
        yield_factor = float(np.clip(yield_factor, 0.0, 1.0))

        over_flags = g["over_hc"].to_numpy()
        from evaluation.operation_runs import summarize_operation_runs, yearly_operation_metric_row

        op = yearly_operation_metric_row(summarize_operation_runs(g))
        rows.append(
            {
                "year": int(year),
                "rain_total": rain_total,
                "irrigation_total": irrig_total,
                "drainage_total": drain_total,
                "rain_utilization": rain_util,
                "yield_factor": yield_factor,
                "drought_risk_factor": yield_factor,
                "managed_water": irrig_total + drain_total,
                # irrigation_times / drainage_times = 有灌/排发生的日数（非连续段事件数）
                "irrigation_times": int(g["irrig_event"].sum()),
                "drainage_times": int(g["drain_event"].sum()),
                "stress_days": int(g["stress_day_lam"].sum()),
                "stress_days_all": int(g["stress_day"].sum()),
                "overpond_days": int(g["over_hc"].sum()),
                "overpond_exposure": float(g["over_depth"].sum()),
                "overpond_events": _count_events(over_flags),
                "max_consecutive_overpond_days": _max_run_length(over_flags),
                "max_h_end": float(g["h_after"].max()),
                "no_drain_high_days": int(g["no_drain_high"].sum()),
                **op,
            }
        )
    return pd.DataFrame(rows)


def compare_summary(yearly_ppo: pd.DataFrame, yearly_rule: pd.DataFrame) -> dict[str, Any]:
    """多年均值 + 相对固定规则 Δ%。

    delta_pct：越低越好 → (rule-ppo)/|rule|*100（正=PPO 更省）；
    越高越好 → (ppo-rule)/|rule|*100（正=PPO 更好）。
    """
    keys_lower = [
        "irrigation_total",
        "drainage_total",
        "managed_water",
        "irrigation_times",
        "drainage_times",
    ]
    keys_higher = ["rain_utilization", "yield_factor", "drought_risk_factor"]
    extra = [
        "stress_days",
        "stress_days_all",
        "overpond_days",
        "overpond_exposure",
        "overpond_events",
        "max_consecutive_overpond_days",
        "max_h_end",
        "no_drain_high_days",
        "rain_total",
    ]
    all_keys = keys_lower + keys_higher + extra

    summary: dict[str, Any] = {}
    for k in all_keys:
        if k in yearly_ppo.columns:
            summary[f"ppo_{k}"] = float(np.nanmean(yearly_ppo[k].astype(float)))
        if k in yearly_rule.columns:
            summary[f"rule_{k}"] = float(np.nanmean(yearly_rule[k].astype(float)))

    def improv_lower(pk, rk):
        a, b = summary.get(pk), summary.get(rk)
        if a is None or b is None or abs(b) < 1e-9:
            return None
        return float((b - a) / abs(b) * 100.0)

    def improv_higher(pk, rk):
        a, b = summary.get(pk), summary.get(rk)
        if a is None or b is None or abs(b) < 1e-9:
            return None
        return float((a - b) / abs(b) * 100.0)

    # 扩展键：越低越好（胁迫 / 超淹 / 未及时排等）；max_h_end 亦按「越低越好」展示 Δ%
    keys_lower_extra = [
        "stress_days",
        "stress_days_all",
        "overpond_days",
        "overpond_exposure",
        "overpond_events",
        "max_consecutive_overpond_days",
        "max_h_end",
        "no_drain_high_days",
    ]

    delta: dict[str, Any] = {}
    for k in keys_lower:
        delta[k] = improv_lower(f"ppo_{k}", f"rule_{k}")
    for k in keys_higher:
        delta[k] = improv_higher(f"ppo_{k}", f"rule_{k}")
    for k in keys_lower_extra:
        if f"ppo_{k}" in summary or f"rule_{k}" in summary:
            delta[k] = improv_lower(f"ppo_{k}", f"rule_{k}")
    summary["delta_pct"] = delta
    summary["n_years_ppo"] = int(len(yearly_ppo))
    summary["n_years_rule"] = int(len(yearly_rule))
    return summary
