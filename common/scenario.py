"""情景参数 + 农艺小工具（土壤 RAW/TAW、ks）。

唯一真相源：configs/scenario.json（可用 --scenario 覆盖）。
训练环境 / 决策 / 评估都从这里读日历与土壤，禁止各处写死。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from .paths import ROOT


@dataclass(frozen=True)
class Scenario:
    """一次运行绑定的农艺情景（日历、土壤、数据路径）。"""

    name: str
    start: str  # "MM/DD"，与 forecast 每年首日对齐
    terminal: str  # "MM/DD"，与 forecast 每年末日对齐
    duration_days: int
    raw: float
    taw: float
    default_perc_mm_d: float
    variety: str = ""
    forecast_rel: str = "data/forecast.csv"
    actual_rel: str = "data/actual.csv"
    growth_para_rel: str = "data/growth_para.csv"
    # 评估留出表（默认 *_policy）；与训练表分开配置
    forecast_eval_rel: str = "data/forecast_policy.csv"
    actual_eval_rel: str = "data/actual_policy.csv"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def duration_inclusive(start_md: str, terminal_md: str, ref_year: int = 2001) -> int:
    """含首尾的生育期天数（非闰年参考年算 MM/DD）。"""
    s = datetime.strptime(f"{ref_year}/{start_md}", "%Y/%m/%d")
    e = datetime.strptime(f"{ref_year}/{terminal_md}", "%Y/%m/%d")
    if e < s:
        raise ValueError(f"terminal {terminal_md} 早于 start {start_md}")
    return (e - s).days + 1


def load_scenario(path: str | Path | None = None) -> Scenario:
    """从 JSON 加载情景；duration_days 为空则按 start~terminal 自动算。"""
    cfg_path = Path(path) if path else ROOT / "configs" / "scenario.json"
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    cal = raw.get("calendar") or {}
    soil = raw.get("soil") or {}
    hydro = raw.get("hydrology") or {}
    crop = raw.get("crop") or {}
    data = raw.get("data") or {}

    # 默认与当前 data/（6/15～10/4）一致
    start = str(cal.get("start", "06/15"))
    terminal = str(cal.get("terminal", "10/04"))
    dur = cal.get("duration_days", None)
    if dur is None or dur == "" or float(dur) == 0:
        duration_days = duration_inclusive(start, terminal)
    else:
        duration_days = int(dur)

    return Scenario(
        name=str(raw.get("name", cfg_path.name)),
        start=start,
        terminal=terminal,
        duration_days=duration_days,
        raw=float(soil.get("RAW", 16.0)),
        taw=float(soil.get("TAW", 80.0)),
        default_perc_mm_d=float(hydro.get("default_perc_mm_d", 1.5)),
        variety=str(crop.get("variety", "")),
        forecast_rel=str(data.get("forecast", "data/forecast.csv")),
        actual_rel=str(data.get("actual", "data/actual.csv")),
        growth_para_rel=str(data.get("growth_para", "data/growth_para.csv")),
        forecast_eval_rel=str(
            data.get("forecast_eval", data.get("forecast_policy", "data/forecast_policy.csv"))
        ),
        actual_eval_rel=str(
            data.get("actual_eval", data.get("actual_policy", "data/actual_policy.csv"))
        ),
    )


# 进程内「当前情景」：env / ks_factor 的 get_scenario() 读这里，避免 --scenario 只拷了表却仍用默认日历
_ACTIVE: Scenario | None = None


@lru_cache(maxsize=4)
def _cached_load(path: Optional[str]) -> Scenario:
    return load_scenario(path)


def get_scenario(path: Optional[str] = None) -> Scenario:
    """取情景。``path is None`` 时优先返回 ``set_active_scenario`` 激活的实例。"""
    global _ACTIVE
    if path is not None:
        return _cached_load(str(path))
    if _ACTIVE is not None:
        return _ACTIVE
    _ACTIVE = _cached_load(None)
    return _ACTIVE


def set_active_scenario(path: str | Path | None = None) -> Scenario:
    """激活情景（训练/评估/预测入口必须先调），保证 env 与拷表用同一套日历与路径。

    相对路径相对仓库根 ``ROOT`` 解析（避免 train chdir 后相对路径找错）。
    """
    global _ACTIVE
    clear_scenario_cache()
    if path is None:
        key = None
    else:
        p = Path(path)
        key = str((p if p.is_absolute() else ROOT / p).resolve())
    _ACTIVE = load_scenario(key)
    return _ACTIVE


def activate_scenario_obj(sc: Scenario) -> Scenario:
    """直接激活内存中的 Scenario（看板 uploads 覆盖路径时用）。"""
    global _ACTIVE
    _cached_load.cache_clear()
    _ACTIVE = sc
    return _ACTIVE


def clear_scenario_cache() -> None:
    global _ACTIVE
    _ACTIVE = None
    _cached_load.cache_clear()


def _abs_data_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else ROOT / p


def validate_scenario_data(
    sc: Scenario | None = None,
    *,
    need_train: bool = True,
    need_eval: bool = True,
) -> dict[str, Any]:
    """开训/评估前校验：表存在 + **每一年**季内逐日齐全。

    必须全员对齐，不能只抽样本年：``Irrigation.reset`` 按年均匀随机，
    缺任一年的 ``start`` 都会在训练中途炸。

    日历角色（与 env 一致）::
      - ``start``：reset 定位季初日；``i`` = 距 start 天数（奖励 7 日前瞻下标）
      - ``terminal``：当日 ``MM/DD`` 命中则 episode done
      - ``duration_days``：奖励里裁剪季末前瞻窗（``Duration``）；应 = start～terminal 含首尾天数
    """
    import pandas as pd

    sc = sc or get_scenario()
    expected_n = int(sc.duration_days)
    # 与 duration_inclusive 对齐，防止 JSON 手填 duration 与日历打架
    auto_n = duration_inclusive(sc.start, sc.terminal)
    report: dict[str, Any] = {
        "name": sc.name,
        "start": sc.start,
        "terminal": sc.terminal,
        "duration_days": expected_n,
        "duration_from_calendar": auto_n,
    }
    errors: list[str] = []
    if expected_n != auto_n:
        errors.append(
            f"duration_days={expected_n} 与 start～terminal 含首尾天数 {auto_n} 不一致"
        )

    def _season_days(year: str | int) -> list[str]:
        y = int(year)
        m0, d0 = (int(x) for x in sc.start.split("/"))
        m1, d1 = (int(x) for x in sc.terminal.split("/"))
        cur = datetime(y, m0, d0)
        end = datetime(y, m1, d1)
        days: list[str] = []
        while cur <= end:
            days.append(f"{cur.year}/{cur.month}/{cur.day}")
            cur += timedelta(days=1)
        return days

    def _check_pair(label: str, fc_rel: str, ac_rel: str) -> dict[str, Any]:
        info: dict[str, Any] = {"forecast": fc_rel, "actual": ac_rel}
        fc_p, ac_p = _abs_data_path(fc_rel), _abs_data_path(ac_rel)
        if not fc_p.exists():
            errors.append(f"{label} forecast 不存在: {fc_p}")
            return info
        if not ac_p.exists():
            errors.append(f"{label} actual 不存在: {ac_p}")
            return info
        fc = pd.read_csv(fc_p, usecols=["date"])
        ac = pd.read_csv(ac_p, usecols=["date"])
        fc_dates = set(fc["date"].astype(str))
        ac_dates = set(ac["date"].astype(str))
        years = sorted(fc["date"].astype(str).str.split("/").str[0].unique())
        ac_years = set(ac["date"].astype(str).str.split("/").str[0].unique())
        if not years:
            errors.append(f"{label} forecast 无有效年份: {fc_p}")
            return info
        missing_year = [y for y in years if y not in ac_years]
        if missing_year:
            errors.append(
                f"{label} actual 缺年份（相对 forecast）: {missing_year[:5]}"
                + ("..." if len(missing_year) > 5 else "")
            )
        info["n_years"] = int(len(years))
        bad_years: list[str] = []
        for y in years:
            need = _season_days(y)
            need_set = set(need)
            if len(need) != expected_n:
                errors.append(
                    f"{label} 年 {y} 日历展开天数 {len(need)} ≠ duration_days {expected_n}"
                )
            # 该年日期集合必须 == 季内逐日（多一天也会让 reward 的 iloc[i] 错位）
            year_fc = {d for d in fc_dates if d.split("/")[0] == y}
            year_ac = {d for d in ac_dates if d.split("/")[0] == y}
            miss_fc = sorted(need_set - year_fc)
            extra_fc = sorted(year_fc - need_set)
            miss_ac = sorted(need_set - year_ac)
            extra_ac = sorted(year_ac - need_set)
            if miss_fc or extra_fc or miss_ac or extra_ac:
                bad_years.append(y)
                if len(bad_years) <= 8:
                    if miss_fc:
                        errors.append(
                            f"{label} forecast 年 {y} 缺 {len(miss_fc)} 日（例 {miss_fc[0]}）"
                        )
                    if extra_fc:
                        errors.append(
                            f"{label} forecast 年 {y} 多出季外 {len(extra_fc)} 日（例 {extra_fc[0]}）"
                        )
                    if miss_ac:
                        errors.append(
                            f"{label} actual 年 {y} 缺 {len(miss_ac)} 日（例 {miss_ac[0]}）"
                        )
                    if extra_ac:
                        errors.append(
                            f"{label} actual 年 {y} 多出季外 {len(extra_ac)} 日（例 {extra_ac[0]}）"
                        )
        if len(bad_years) > 8:
            errors.append(f"{label} 另有 {len(bad_years) - 8} 年日期不齐（共 {len(bad_years)} 年）")
        info["bad_years"] = bad_years
        info["ok"] = not bad_years and not missing_year
        return info

    gp = _abs_data_path(sc.growth_para_rel)
    if not gp.exists():
        errors.append(f"growth_para 不存在: {gp}")
    else:
        report["growth_para"] = str(gp)

    if need_train:
        report["train"] = _check_pair("训练", sc.forecast_rel, sc.actual_rel)
    if need_eval:
        report["eval"] = _check_pair("评估", sc.forecast_eval_rel, sc.actual_eval_rel)

    if errors:
        raise ValueError("情景数据校验失败:\n  - " + "\n  - ".join(errors))
    return report


def ks_factor(h: float, raw: float | None = None, taw: float | None = None) -> float:
    """水分胁迫系数 ks ∈ [0, 1]。

    h > -RAW → 1；否则 (TAW + h) / (TAW - RAW)，再下限夹 0。
    raw/taw 缺省取当前 scenario。
    """
    if raw is None or taw is None:
        sc = get_scenario()
        raw = sc.raw if raw is None else raw
        taw = sc.taw if taw is None else taw
    h, raw, taw = float(h), float(raw), float(taw)
    if h > -raw:
        return 1.0
    return max((taw + h) / (taw - raw), 0.0)
