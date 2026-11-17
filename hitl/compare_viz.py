"""三方对比看板可视化数据：年份角色、整季轨迹、生育期汇总、吸收概率。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import (
    BC_DEFAULT,
    HITL_HOLDOUT_YEARS,
    HITL_REVIEW_YEARS,
    action_to_idx,
)
from .session import SessionPaths

STAGE_CN = {
    "Transplanting": "移栽期",
    "Tillering": "分蘖期",
    "Tillering_late": "分蘖末期",
    "Mid_drying": "晒田期",
    "Panicle_initiation": "拔节孕穗期",
    "Flowering": "抽穗开花期",
    "Grain_filling": "乳熟期",
    "Harvest": "落干收获期",
}

STAGE_ORDER = [
    "Transplanting",
    "Tillering",
    "Tillering_late",
    "Mid_drying",
    "Panicle_initiation",
    "Flowering",
    "Grain_filling",
    "Harvest",
]

YEARLY_VIZ_KEYS = (
    "irrigation_total",
    "drainage_total",
    "managed_water",
    "irrigation_times",
    "drainage_times",
    "irrigation_operation_days",
    "drainage_operation_days",
    "irrigation_contiguous_segments",
    "drainage_contiguous_segments",
    "stress_days",
    "stress_days_all",
    "rain_utilization",
    "yield_factor",
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


def year_role(year: str | int) -> dict[str, Any]:
    """三类评估年：审阅训练 / 审阅+BC验证 / 未审阅留出。审阅年同样参与评估对比。"""
    y = str(year).strip()
    val_years = {str(v) for v in (BC_DEFAULT.val_years or [])}
    holdout = {str(v) for v in HITL_HOLDOUT_YEARS}
    review = {str(v) for v in HITL_REVIEW_YEARS}
    if y in holdout:
        role = "holdout"
        label = "未审阅留出年"
        short = "留出"
        is_review = False
    elif y in val_years:
        role = "bc_val"
        label = "审阅年 · 亦为BC内部验证年（标签不进训练）"
        short = "审阅·验证"
        is_review = True
    elif y in review:
        role = "bc_train"
        label = "审阅年 · 亦为BC训练标签年"
        short = "审阅·训练"
        is_review = True
    else:
        role = "other"
        label = "其他评估年"
        short = "其他"
        is_review = y in review
    return {
        "year": y,
        "role": role,
        "label": label,
        "short": short,
        "is_review": bool(is_review),
    }


def resolve_growth_para(sp: SessionPaths) -> Path | None:
    cands = [
        sp.root / "_workdir" / "growth_para.csv",
        sp.root / "02_eval" / "_workdir" / "growth_para.csv",
        Path("data") / "growth_para.csv",
    ]
    for p in cands:
        if p.exists():
            return p
    return None


def _parse_growth_ends(gp_path: Path) -> list[tuple[str, str]]:
    """返回 [(period_name, end_MM/DD), ...] 按日历顺序。"""
    gp = pd.read_csv(gp_path, header=0)
    if "period_name" not in gp.columns:
        return []
    gp = gp.set_index("period_name")
    if "end" not in gp.index:
        return []
    ends: list[tuple[str, str]] = []
    for col in gp.columns:
        try:
            end = str(gp.loc["end", col]).strip()
        except Exception:
            continue
        if not end or end.lower() == "nan":
            continue
        ends.append((str(col), end))
    return ends


def stage_for_date(date_str: str, ends: list[tuple[str, str]]) -> str:
    s = str(date_str).replace("-", "/").strip()
    parts = s.split("/")
    if len(parts) < 3:
        return "Unknown"
    md = f"{int(parts[1]):02d}/{int(parts[2]):02d}"
    try:
        from datetime import datetime

        md_obj = datetime.strptime(md, "%m/%d")
    except Exception:
        return "Unknown"
    for name, end in ends:
        try:
            from datetime import datetime

            if md_obj <= datetime.strptime(str(end), "%m/%d"):
                return name
        except Exception:
            continue
    return "Unknown"


def load_policy_year_df(path: Path, year: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    mask = df["date"].astype(str).str.replace("-", "/", regex=False).str.startswith(str(year))
    return df.loc[mask].copy()


def season_rows_enriched(
    path: Path,
    year: str,
    *,
    growth_para: Path | None = None,
    include_actual_rain: bool = True,
) -> list[dict[str, Any]]:
    df = load_policy_year_df(path, year)
    if df.empty:
        return []
    ends = _parse_growth_ends(growth_para) if growth_para else []
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        date = str(r.get("date", ""))
        stage = stage_for_date(date, ends) if ends else "Unknown"
        item: dict[str, Any] = {
            "date": date,
            "year": str(year),
            "stage": stage,
            "stage_cn": STAGE_CN.get(stage, stage),
        }
        for c in (
            "h_before",
            "h_after",
            "hmin",
            "hmax",
            "Hp",
            "Hc",
            "a_mm",
            "a_exec",
            "a_cmd",
            "m",
            "d",
            "ks",
            "P_cumu7",
            "et0_a",
            "lamda",
        ):
            if c not in df.columns:
                continue
            v = r.get(c)
            if pd.isna(v):
                item[c] = None
            else:
                try:
                    item[c] = float(v)
                except Exception:
                    item[c] = None
        if include_actual_rain:
            for rk in ("P_1_a", "P_actual_display"):
                if rk in df.columns and not pd.isna(r.get(rk)):
                    try:
                        item["rain"] = float(r.get(rk))
                        item[rk] = item["rain"]
                        break
                    except Exception:
                        pass
        # 执行量：灌正排负（优先 m/d）
        m = item.get("m")
        d = item.get("d")
        if m is not None or d is not None:
            item["exec_signed"] = float(m or 0.0) - float(d or 0.0)
        else:
            ae = item.get("a_exec")
            item["exec_signed"] = float(ae) if ae is not None else 0.0
        rows.append(item)
    return rows


def _num(v: Any) -> float:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def stage_metrics_from_df(df: pd.DataFrame, ends: list[tuple[str, str]]) -> list[dict[str, Any]]:
    if df.empty:
        return []
    work = df.copy()
    work["_stage"] = [stage_for_date(str(d), ends) for d in work["date"]]
    out: list[dict[str, Any]] = []
    for st in STAGE_ORDER:
        g = work[work["_stage"] == st]
        if g.empty:
            continue
        m = g["m"].map(_num) if "m" in g.columns else pd.Series(0.0, index=g.index)
        d = g["d"].map(_num) if "d" in g.columns else pd.Series(0.0, index=g.index)
        h = g["h_before"].map(_num) if "h_before" in g.columns else pd.Series(np.nan, index=g.index)
        # 胁迫：优先用轨迹 ks；否则无法算则空
        if "ks" in g.columns:
            ks = g["ks"].map(lambda x: _num(x) if pd.notna(x) else 1.0)
            stress = int((ks < 1.0 - 1e-6).sum())
        else:
            stress = None
        below_hmin = None
        if "hmin" in g.columns and "h_before" in g.columns:
            below_hmin = int(
                ((g["h_before"].map(_num) + 1e-9) < g["hmin"].map(_num)).sum()
            )
        irrig_days = int((m > 1e-9).sum())
        drain_days = int((d > 1e-9).sum())
        out.append({
            "stage": st,
            "stage_cn": STAGE_CN.get(st, st),
            "n_days": int(len(g)),
            "irrigation_total": float(m.sum()),
            "drainage_total": float(d.sum()),
            "irrigation_times": irrig_days,
            "drainage_times": drain_days,
            "operation_days": int(((m > 1e-9) | (d > 1e-9)).sum()),
            "h_before_mean": float(h.mean()) if h.notna().any() else None,
            "stress_days": stress,
            "below_hmin_days": below_hmin,
            "note_stress": "stress_days 按轨迹 ks<1；below_hmin_days 为晨水层低于 hmin，二者不同",
        })
    return out


def build_stage_compare(
    sp: SessionPaths,
    *,
    year: str | None = None,
    years: list[str] | None = None,
) -> dict[str, Any]:
    gp = resolve_growth_para(sp)
    ends = _parse_growth_ends(gp) if gp else []
    arms = {
        "rule": sp.compare / "rule" / "policy_output_RULE.csv",
        "original_ppo": sp.compare / "original_ppo" / "policy_output_PPO.csv",
        "hitl_ppo": sp.compare / "hitl_ppo" / "policy_output_PPO.csv",
    }
    # 兼容规则臂也可能叫 PPO 文件名
    if not arms["rule"].exists():
        alt = sp.compare / "rule" / "policy_output_PPO.csv"
        if alt.exists():
            arms["rule"] = alt

    target_years = [str(year)] if year else [str(y) for y in (years or [])]
    if not target_years:
        return {"ok": False, "error": "未指定年份"}

    per_arm: dict[str, list[dict[str, Any]]] = {}
    for name, path in arms.items():
        if not path.exists():
            per_arm[name] = []
            continue
        frames = [load_policy_year_df(path, y) for y in target_years]
        frames = [f for f in frames if not f.empty]
        if not frames:
            per_arm[name] = []
            continue
        df = pd.concat(frames, ignore_index=True)
        # 多年：按阶段先逐年再平均
        if len(target_years) == 1:
            per_arm[name] = stage_metrics_from_df(df, ends)
        else:
            by_y = []
            for y in target_years:
                by_y.append(stage_metrics_from_df(load_policy_year_df(path, y), ends))
            # average numeric fields by stage
            acc: dict[str, dict[str, list[float]]] = {}
            meta: dict[str, dict[str, Any]] = {}
            for one in by_y:
                for row in one:
                    st = row["stage"]
                    meta[st] = {"stage": st, "stage_cn": row["stage_cn"]}
                    bucket = acc.setdefault(st, {})
                    for k, v in row.items():
                        if k in ("stage", "stage_cn", "note_stress"):
                            continue
                        if v is None:
                            continue
                        bucket.setdefault(k, []).append(float(v))
            merged = []
            for st in STAGE_ORDER:
                if st not in acc:
                    continue
                item = dict(meta[st])
                for k, vals in acc[st].items():
                    item[k] = float(sum(vals) / len(vals)) if vals else None
                item["note_stress"] = (
                    "多年均值；stress_days 按轨迹 ks<1；below_hmin_days 与胁迫不同"
                )
                item["n_years"] = len(target_years)
                merged.append(item)
            per_arm[name] = merged

    return {
        "ok": True,
        "years": target_years,
        "mode": "year" if len(target_years) == 1 else "mean",
        "growth_para": str(gp) if gp else None,
        "arms": per_arm,
        "defs": {
            "irrigation_times": "有灌水执行(m>0)的日数",
            "drainage_times": "有排水执行(d>0)的日数",
            "operation_days": "当日有灌或排的日数",
            "stress_days": "轨迹 ks<1 日数（环境胁迫）",
            "below_hmin_days": "晨水层 h_before < hmin 日数（管理下限，≠胁迫）",
        },
    }


def _yearly_map(arm_dir: Path, prefer: str) -> dict[str, dict[str, Any]]:
    path = arm_dir / f"yearly_metrics_{prefer}.csv"
    rows: list[dict[str, Any]] = []
    if path.exists():
        rows = pd.read_csv(path).to_dict(orient="records")
    else:
        cands = list(arm_dir.glob("yearly_metrics_*.csv"))
        if cands:
            rows = pd.read_csv(cands[0]).to_dict(orient="records")
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        y = r.get("year")
        if y is None:
            continue
        ys = str(int(y) if not isinstance(y, str) else y)
        slim: dict[str, Any] = {"year": ys}
        for k in YEARLY_VIZ_KEYS:
            if k in r and pd.notna(r[k]):
                try:
                    slim[k] = float(r[k])
                except Exception:
                    slim[k] = r[k]
            elif (
                k.startswith("irrigation_small")
                or k.startswith("drainage_small")
                or k.startswith("irrigation_topup")
            ):
                slim[k] = 0.0
        slim.update(year_role(ys))
        out[ys] = slim
    return out


def build_yearly_pairs(sp: SessionPaths) -> dict[str, Any]:
    rule = _yearly_map(sp.compare / "rule", "RULE")
    ppo = _yearly_map(sp.compare / "original_ppo", "PPO")
    hitl = _yearly_map(sp.compare / "hitl_ppo", "PPO")
    years = sorted(set(rule) | set(ppo) | set(hitl), key=lambda x: int(x))
    pairs = []
    for y in years:
        role = year_role(y)
        pairs.append({
            **role,
            "rule": rule.get(y),
            "original_ppo": ppo.get(y),
            "hitl_ppo": hitl.get(y),
        })
    return {
        "ok": True,
        "years": years,
        "pairs": pairs,
        "roles": {
            "bc_train": [y for y in years if year_role(y)["role"] == "bc_train"],
            "bc_val": [y for y in years if year_role(y)["role"] == "bc_val"],
            "holdout": [y for y in years if year_role(y)["role"] == "holdout"],
        },
        "role_guide": [
            {
                "role": "bc_train",
                "years": [y for y in years if year_role(y)["role"] == "bc_train"],
                "label": "2012—2014：审阅年，也是BC训练标签年",
            },
            {
                "role": "bc_val",
                "years": [y for y in years if year_role(y)["role"] == "bc_val"],
                "label": "2016：审阅年，也是BC内部验证年（标签用于验证，不用于训练）",
            },
            {
                "role": "holdout",
                "years": [y for y in years if year_role(y)["role"] == "holdout"],
                "label": "2017—2020：未审阅留出年",
            },
        ],
        "defs": {
            "irrigation_times": "灌水日数（当日执行灌水量 m>0）",
            "drainage_times": "排水日数（当日执行排水量 d>0）",
            "irrigation_contiguous_segments": "连续灌水操作段数（相邻有灌日合并为一段）",
            "irrigation_small_run_ge3d_*": "连续≥3日且每日 0<m≤20mm 的小额灌溉段",
            "stress_days": "环境 ks 胁迫日（λ>0）；≠低于 hmin 日数",
            "note": "审阅年同样参与评估对比（与原PPO同气象情景）；不是「仅训练不评估」。",
        },
    }


def load_expert_edit_dates(sp: SessionPaths, year: str) -> list[dict[str, Any]]:
    """审阅年专家改动日（相对原模型提案）；留出年通常为空。"""
    path = sp.labels if sp.labels.exists() else sp.review_json
    if not path.exists():
        if sp.review_csv.exists():
            df = pd.read_csv(sp.review_csv)
        else:
            return []
    else:
        df = pd.read_json(path) if path.suffix == ".json" else pd.read_csv(path)
    if df.empty or "date" not in df.columns:
        return []
    y = str(year)
    out: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        d = str(r.get("date", "")).replace("-", "/")
        if not d.startswith(y):
            continue
        flag = str(r.get("flag", "") or "")
        is_edit = flag == "edited"
        if not is_edit and "a_expert" in r.index and "a_model" in r.index:
            try:
                is_edit = abs(float(r["a_expert"]) - float(r["a_model"])) > 1e-6
            except Exception:
                is_edit = False
        if not is_edit:
            continue
        try:
            a_exp = float(r.get("a_expert")) if pd.notna(r.get("a_expert")) else None
        except Exception:
            a_exp = None
        try:
            a_mod = float(r.get("a_model")) if pd.notna(r.get("a_model")) else None
        except Exception:
            a_mod = None
        out.append({
            "date": d,
            "a_expert": a_exp,
            "a_model": a_mod,
            "reason_code": str(r.get("reason_code", "") or ""),
        })
    return out


def build_stage_year_heatmap(
    sp: SessionPaths,
    *,
    metric: str = "irrigation_total",
) -> dict[str, Any]:
    """生育期 × 年份：HITL − 原PPO 的差值矩阵（0=无变化为中心）。"""
    allowed = {
        "irrigation_total",
        "drainage_total",
        "operation_days",
        "irrigation_times",
        "drainage_times",
        "h_before_mean",
        "stress_days",
    }
    if metric not in allowed:
        metric = "irrigation_total"
    pairs = build_yearly_pairs(sp)
    years = pairs.get("years") or []
    gp = resolve_growth_para(sp)
    ends = _parse_growth_ends(gp) if gp else []
    ppo_path = sp.compare / "original_ppo" / "policy_output_PPO.csv"
    hitl_path = sp.compare / "hitl_ppo" / "policy_output_PPO.csv"

    # stage order from first available year
    stage_keys: list[str] = list(STAGE_ORDER)
    matrix: list[dict[str, Any]] = []
    values: list[float] = []

    for y in years:
        ppo_m = {
            r["stage"]: r
            for r in stage_metrics_from_df(load_policy_year_df(ppo_path, y), ends)
        }
        hitl_m = {
            r["stage"]: r
            for r in stage_metrics_from_df(load_policy_year_df(hitl_path, y), ends)
        }
        row_cells = []
        for st in stage_keys:
            a = ppo_m.get(st, {})
            b = hitl_m.get(st, {})
            va = a.get(metric)
            vb = b.get(metric)
            if va is None and vb is None:
                delta = None
            else:
                delta = float(vb or 0.0) - float(va or 0.0)
                values.append(delta)
            row_cells.append({
                "stage": st,
                "stage_cn": STAGE_CN.get(st, st),
                "ppo": va,
                "hitl": vb,
                "delta": delta,
            })
        matrix.append({
            **year_role(y),
            "cells": row_cells,
        })

    # symmetric scale around 0
    abs_max = max((abs(v) for v in values), default=0.0)
    if abs_max < 1e-9:
        abs_max = 1.0

    return {
        "ok": True,
        "metric": metric,
        "metric_label": {
            "irrigation_total": "灌水量 Δmm（HITL−原PPO）",
            "drainage_total": "排水量 Δmm（HITL−原PPO）",
            "operation_days": "操作日数 Δ（HITL−原PPO）",
            "irrigation_times": "灌水日数 Δ",
            "drainage_times": "排水日数 Δ",
            "h_before_mean": "晨水层均值 Δmm",
            "stress_days": "胁迫日数 Δ",
        }.get(metric, metric),
        "years": years,
        "stages": [
            {"stage": st, "stage_cn": STAGE_CN.get(st, st)} for st in stage_keys
        ],
        "rows": matrix,
        "scale": {"min": -abs_max, "max": abs_max, "center": 0.0},
        "note": "颜色以无变化(0)为中心；暖色=HITL相对原PPO增大，冷色=减小。晒田/落干多为固定规则。",
        "role_guide": pairs.get("role_guide"),
    }


def compute_absorption_probs(sp: SessionPaths, *, force: bool = False) -> dict[str, Any]:
    out_path = sp.metrics / "absorption_probs.json"
    labels_path = sp.labels if sp.labels.exists() else sp.review_json
    base = sp.model if sp.model.exists() else None
    hitl = sp.hitl_model if sp.hitl_model.exists() else None

    def _cache_stale() -> bool:
        """HITL / labels / absorption.json 任一比缓存新 → 必须重算。"""
        if not out_path.exists():
            return True
        try:
            cache_m = out_path.stat().st_mtime
        except OSError:
            return True
        for p in (hitl, labels_path, sp.metrics / "absorption.json"):
            if p is None:
                continue
            try:
                if p.exists() and p.stat().st_mtime > cache_m + 1e-6:
                    return True
            except OSError:
                continue
        return False

    if out_path.exists() and not force and not _cache_stale():
        try:
            cached = json.loads(out_path.read_text(encoding="utf-8"))
            if cached.get("ok"):
                return cached
        except Exception:
            pass

    if not labels_path.exists():
        out = {"ok": False, "error": "无 labels/review"}
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    labels = pd.read_json(labels_path) if labels_path.suffix == ".json" else pd.read_csv(labels_path)
    if labels.empty:
        out = {"ok": False, "error": "标签为空"}
        return out

    if base is None or hitl is None:
        out = {"ok": False, "error": "缺少原 PPO 或 HITL 权重"}
        return out

    from common.action_mask import ACTION_SPACE, mask_policy_logits, valid_action_mask
    from common.model_io import load_ppo_checkpoint
    import torch
    import torch.nn.functional as F

    ppo_agent, _, _ = load_ppo_checkpoint(base)
    hitl_agent, _, _ = load_ppo_checkpoint(hitl)
    ppo_agent.policy_net.eval()
    hitl_agent.policy_net.eval()
    action_space = list(ACTION_SPACE)

    sv_col = "s_vec" if "s_vec" in labels.columns else ("state_vec" if "state_vec" in labels.columns else None)
    if not sv_col:
        out = {"ok": False, "error": "无 s_vec/state_vec"}
        return out

    include = labels.get("include_in_bc", pd.Series(True, index=labels.index)).fillna(False).astype(bool)
    flags = labels.get("flag", pd.Series("", index=labels.index)).astype(str)
    edited_mask = flags.eq("edited")
    # fallback: compare a_expert vs a_model
    if not edited_mask.any() and "a_expert" in labels.columns and "a_model" in labels.columns:
        def _edited(r):
            try:
                return abs(float(r["a_expert"]) - float(r["a_model"])) > 1e-6
            except Exception:
                return False
        edited_mask = labels.apply(_edited, axis=1)

    def _target_idx(row) -> Optional[int]:
        if "a_idx" in row and pd.notna(row.get("a_idx")):
            try:
                return int(row["a_idx"])
            except Exception:
                pass
        for col in ("a_label_mm", "a_expert", "a_model"):
            if col in row and pd.notna(row.get(col)):
                try:
                    return action_to_idx(row[col])
                except Exception:
                    continue
        return None

    def _prob(agent, state, target_idx: int) -> dict[str, Any]:
        device = agent.device
        x = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.inference_mode():
            logits, _ = agent.policy_net(x)
            masked = mask_policy_logits(logits, x, action_space)
            valid = valid_action_mask(x, action_space)[0]
            probs = F.softmax(masked, dim=-1)[0]
            n_valid = int(valid.sum().item())
            if target_idx < 0 or target_idx >= len(action_space) or not bool(valid[target_idx].item()):
                return {"ok": False, "n_valid": n_valid}
            p = float(probs[target_idx].item())
            top = int(masked.argmax(dim=-1).item())
            return {
                "ok": True,
                "prob": p,
                "top1_mm": float(action_space[top]),
                "top1_match": bool(top == target_idx),
                "n_valid": n_valid,
            }

    edited_rows: list[dict[str, Any]] = []
    confirm_rows: list[dict[str, Any]] = []
    for _, row in labels.loc[include].iterrows():
        sv = row.get(sv_col)
        if sv is None:
            continue
        try:
            state = np.asarray(sv, dtype=np.float32).reshape(-1)
            if state.size == 0 or not np.isfinite(state).all():
                continue
        except Exception:
            continue
        tidx = _target_idx(row)
        if tidx is None:
            continue
        before = _prob(ppo_agent, state, tidx)
        after = _prob(hitl_agent, state, tidx)
        if not before.get("ok") or not after.get("ok"):
            continue
        if int(before["n_valid"]) <= 1:
            continue  # 强制单档不计入概率诊断
        rec = {
            "date": str(row.get("date", "")),
            "year": str(row.get("year", "") or ""),
            "stage": str(row.get("stage", "") or ""),
            "expert_mm": float(ACTION_SPACE[tidx]),
            "ppo_prob": before["prob"],
            "hitl_prob": after["prob"],
            "delta": after["prob"] - before["prob"],
            "ppo_top1_match": before["top1_match"],
            "hitl_top1_match": after["top1_match"],
        }
        if bool(edited_mask.loc[row.name]):
            edited_rows.append(rec)
        else:
            # 确认日：看「原模型动作」概率变化
            try:
                model_mm = float(row.get("a_model"))
                model_idx = action_to_idx(model_mm)
            except Exception:
                model_idx = tidx
            b2 = _prob(ppo_agent, state, model_idx)
            a2 = _prob(hitl_agent, state, model_idx)
            if b2.get("ok") and a2.get("ok"):
                confirm_rows.append({
                    "date": rec["date"],
                    "year": rec["year"],
                    "stage": rec["stage"],
                    "model_mm": float(ACTION_SPACE[model_idx]),
                    "ppo_prob": b2["prob"],
                    "hitl_prob": a2["prob"],
                    "delta": a2["prob"] - b2["prob"],
                    "ppo_top1_match": b2["top1_match"],
                    "hitl_top1_match": a2["top1_match"],
                })

    def _summary(rows: list[dict[str, Any]], match_key_before: str, match_key_after: str) -> dict[str, Any]:
        if not rows:
            return {"n": 0}
        deltas = [r["delta"] for r in rows]
        return {
            "n": len(rows),
            "mean_ppo_prob": float(np.mean([r["ppo_prob"] for r in rows])),
            "mean_hitl_prob": float(np.mean([r["hitl_prob"] for r in rows])),
            "mean_delta": float(np.mean(deltas)),
            "prob_increased_share": float(np.mean([d > 1e-12 for d in deltas])),
            "top1_match_rate_ppo": float(np.mean([r[match_key_before] for r in rows])),
            "top1_match_rate_hitl": float(np.mean([r[match_key_after] for r in rows])),
        }

    # 并入 absorption 主率（若已有）
    abs_path = sp.metrics / "absorption.json"
    abs_summary = None
    if abs_path.exists():
        try:
            abs_summary = json.loads(abs_path.read_text(encoding="utf-8"))
        except Exception:
            abs_summary = None

    out = {
        "ok": True,
        "edited": edited_rows,
        "confirmed": confirm_rows,
        "summary_edited": _summary(edited_rows, "ppo_top1_match", "hitl_top1_match"),
        "summary_confirmed": _summary(confirm_rows, "ppo_top1_match", "hitl_top1_match"),
        "absorption": {
            "edit_reproduce_rate": (abs_summary or {}).get("edit_reproduce_rate"),
            "confirm_hold_rate_policy": (abs_summary or {}).get("confirm_hold_rate_policy"),
        },
        "note": "改动日：专家标签动作概率；确认日：原模型动作概率。仅 n_valid>1。",
    }
    sp.metrics.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
