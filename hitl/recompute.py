"""状态—动作一致重算：定稿动作序列 → 约定季初连续重放 → 与审阅轨迹校验 → BC。

设计要点
--------
- **不调用 PPO**。驱动动作为审阅表请求档（已填 ``a_expert``，否则 ``a_model``）。
- **季初**：与 ``rollout`` / ``reset_policy`` 一致，从 ``hmax`` 起连续递推（不拿表内
  ``h_before`` 直接绑标签，以免掩盖水量断裂）。
- **BC 标签**：合法**请求**动作类别；非法请求报错，要求重新审阅（不自动改成执行量档）。
- **执行量** ``a_actual``：同一执行链落地，供递推与效果评价。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch

import train.PPO_Rice_env as PPO_Rice_env
from common.action_mask import ACTION_SPACE, valid_action_mask
from common.scenario import ks_factor, get_scenario
from common.execution import EXEC_PROTOCOL_VERSION

from .config import EXEC_DEFAULT, ExecConfig, action_to_idx, snap_to_action_space
from .exec_layer import (
    apply_action,
    apply_expert_action,
    date_ymd,
    expert_action_filled,
    is_fixed_rule_stage,
    norm_date,
    stage_name_from_state,
)

PathLike = Union[str, Path]

# 执行/导出协议版本（渗漏修复等环境变更时递增；与旧检查点重评区分）


def _is_finite_number(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def request_valid_under_mask(s_vec: np.ndarray, a_req_mm: float) -> bool:
    """请求档是否落在当前状态的动作掩码内。"""
    if not _is_finite_number(a_req_mm) or float(a_req_mm) not in ACTION_SPACE:
        return False
    idx = action_to_idx(a_req_mm)
    x = torch.as_tensor(np.asarray(s_vec, dtype=np.float32).reshape(1, -1))
    mask = valid_action_mask(x, ACTION_SPACE).reshape(-1)
    return bool(mask[idx].item())


def _resolve_request_action(row: pd.Series, *, editable: bool) -> tuple[float, str]:
    """定稿请求动作（mm 档）与样本旗标。"""
    if not editable:
        raw = row.get("a_expert", np.nan)
        if pd.notna(raw) and raw != "":
            return float(raw), "fixed_rule"
        return float(row.get("a_model", 0)), "fixed_rule"

    raw = row.get("a_expert", np.nan)
    a_model = float(row["a_model"])
    if not expert_action_filled(raw):
        return a_model, "confirmed"
    a = float(raw)
    if abs(a - a_model) < 1e-6:
        return float(a), "confirmed"
    return float(a), "edited"


def validate_review_table(review_df: pd.DataFrame) -> list[str]:
    """缺日/重复日/缺字段/非有限数值 → 错误列表（空=通过）。"""
    errs: list[str] = []
    if review_df is None or review_df.empty:
        return ["审阅表为空"]
    need = ["date", "h_before", "h_after", "a_model", "a_exec", "state_vec"]
    for c in need:
        if c not in review_df.columns:
            errs.append(f"缺列: {c}")
    if errs:
        return errs

    dates = [norm_date(d) for d in review_df["date"].tolist()]
    sc = get_scenario()
    by_year = {}
    for d in dates:
        by_year.setdefault(date_ymd(d)[0], set()).add(d)
    for y, observed in by_year.items():
        expected = {norm_date(t) for t in pd.date_range(f"{y}/{sc.start}", f"{y}/{sc.terminal}")}
        if observed != expected:
            errs.append(f"{y} 季日历不完整: 缺 {sorted(expected-observed)[:5]}，多 {sorted(observed-expected)[:5]}")
    if len(dates) != len(set(dates)):
        from collections import Counter

        dup = [d for d, n in Counter(dates).items() if n > 1]
        errs.append(f"重复日期: {dup[:10]}")

    for i, (_, r) in enumerate(review_df.iterrows()):
        d = dates[i]
        if "year" in r and str(r["year"]) != str(date_ymd(d)[0]):
            errs.append(f"{d} year 字段不符")
        for col in ("h_before", "h_after", "a_model", "a_exec"):
            if not _is_finite_number(r.get(col)):
                errs.append(f"{d} 列 {col} 非有限数值")
        if expert_action_filled(r.get("a_expert")) and not _is_finite_number(r.get("a_expert")):
            errs.append(f"{d} 列 a_expert 非有限数值")
        if "state_vec" in review_df.columns:
            sv = r.get("state_vec")
            if sv is not None and not (
                isinstance(sv, str) and not str(sv).strip()
            ):
                try:
                    arr = np.asarray(sv if not isinstance(sv, str) else json.loads(sv), dtype=np.float64)
                    if arr.shape != (8,) or not np.isfinite(arr).all():
                        errs.append(f"{d} state_vec 须为 8 维有限向量")
                except Exception as exc:
                    errs.append(f"{d} state_vec 无法解析: {exc}")
            else:
                errs.append(f"{d} state_vec 缺失")
    return errs[:80]


def trajectory_fingerprint(review_df: pd.DataFrame, *, year: Optional[str] = None) -> str:
    """定稿轨迹指纹：日期 / 晨水 / 请求 / 执行 / 日末（改动作后旧确认应失效）。"""
    df = review_df.copy()
    if year is not None and "year" in df.columns:
        df = df[df["year"].astype(str) == str(year)]
    df = df.copy()
    df["_d"] = df["date"].map(norm_date)
    df["_ord"] = df["_d"].map(date_ymd)
    df = df.sort_values("_ord")
    rows = []
    for _, r in df.iterrows():
        req = r.get("a_expert")
        if not expert_action_filled(req):
            req = r.get("a_model")
        rows.append({
            "d": norm_date(r["date"]),
            "hb": round(float(r["h_before"]), 6) if _is_finite_number(r.get("h_before")) else None,
            "req": round(float(req), 6) if _is_finite_number(req) else None,
            "ex": round(float(r["a_exec"]), 6) if _is_finite_number(r.get("a_exec")) else None,
            "ha": round(float(r["h_after"]), 6) if _is_finite_number(r.get("h_after")) else None,
            "sv": [round(float(v), 6) for v in (json.loads(r["state_vec"]) if isinstance(r.get("state_vec"), str) else r.get("state_vec", []))],
            "confirmed": bool(r.get("confirmed", False)),
            "pending": bool(r.get("needs_reconfirm", False)),
        })
    blob = json.dumps({"protocol": EXEC_PROTOCOL_VERSION, "rows": rows}, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _recompute_one_season(
    season_df: pd.DataFrame,
    env: PPO_Rice_env.Irrigation,
    *,
    cfg: ExecConfig,
    force_hc_guard: Optional[bool],
    include_fixed_rule_in_bc: bool,
) -> list[dict[str, Any]]:
    """单季：从季初 hmax 连续重放定稿请求序列（不调用 PPO）。"""
    if season_df.empty:
        return []

    state_space = env.state_space_policy.copy()
    state_space["_d"] = state_space["date"].map(norm_date)
    actual_space = env.actual_state_space_policy.copy()
    actual_space["_d"] = actual_space["date"].map(norm_date)
    season_df = season_df.copy()
    season_df["_d"] = season_df["date"].map(norm_date)
    season_df["_ord"] = season_df["_d"].map(date_ymd)
    season_df = season_df.sort_values("_ord").reset_index(drop=True)

    first_date = season_df.iloc[0]["_d"]
    matched = state_space[state_space["_d"] == first_date]
    if matched.empty:
        raise ValueError(f"重算找不到季初日: {first_date}")
    state = matched.iloc[0].copy()
    # 约定季初：与 rollout / reset_policy 一致
    state["h_before"] = float(state["hmax"])

    samples: list[dict[str, Any]] = []
    illegal: list[dict[str, Any]] = []
    records = list(season_df.iterrows())
    for i, (_, row) in enumerate(records):
        date = row["_d"]
        base_m = state_space[state_space["_d"] == date]
        if base_m.empty:
            raise ValueError(f"重放缺环境日: {date}")
        base = base_m.iloc[0]
        for col in base.index:
            if col not in ("h_before", "h_after", "ks", "_d"):
                state[col] = base[col]

        stage = stage_name_from_state(state)
        fixed = is_fixed_rule_stage(state, stage_name=stage, cfg=cfg)
        if "editable" in row and pd.notna(row["editable"]):
            editable = bool(row["editable"]) and not fixed
        else:
            editable = not fixed

        a_req, flag = _resolve_request_action(row, editable=editable)
        st = state.drop(labels=["_d"], errors="ignore")
        actual_m = actual_space[actual_space["_d"] == date]
        if actual_m.empty:
            raise ValueError(f"重放缺实测日: {date}")
        actual = actual_m.iloc[0].drop(labels=["_d"], errors="ignore")

        s_vec = np.array(env.process_series(st), dtype=np.float32)
        if fixed:
            h_after, a_actual, info = apply_action(
                env, st, actual, a_req,
                force_hc_guard=force_hc_guard, fixed_rule_stage=True, cfg=cfg,
            )
            flag = "fixed_rule"
            a_label_mm = float(a_req)
            label_ok = True
        else:
            h_after, a_actual, info = apply_expert_action(
                env, st, actual, a_req, cfg=cfg, force_hc_guard=force_hc_guard,
            )
            a_label_mm = float(a_req)
            label_ok = request_valid_under_mask(s_vec, a_label_mm)
            if not label_ok:
                illegal.append({
                    "date": date,
                    "a_requested_mm": a_label_mm,
                    "a_executed_mm": float(a_actual),
                    "h_before": float(st.h_before),
                    "Hp": float(st.Hp),
                    "Hc": float(st.Hc),
                    "hmax": float(st.hmax),
                    "note": "请求档不在动作掩码内；请重新审阅，勿用执行量冒充专家意见",
                })

        row_confirmed = bool(row.get("confirmed", False)) if "confirmed" in row.index else False
        if fixed:
            include = bool(include_fixed_rule_in_bc)
        else:
            include = bool(row_confirmed) and not bool(row.get("needs_reconfirm", False)) and label_ok

        samples.append({
            "date": date,
            "year": str(date).split("/")[0] if "/" in date else str(row.get("year", "")),
            "stage": stage,
            "h_before": float(st.h_before),
            "s_vec": s_vec.tolist(),
            "a_model": float(row["a_model"]),
            "a_expert": float(a_req),
            "a_requested_mm": float(a_req),
            "a_label_mm": float(a_label_mm),
            "a_idx": action_to_idx(a_label_mm) if label_ok or fixed else -1,
            "a_actual": float(a_actual),
            "a_executed_mm": float(a_actual),
            "flag": flag,
            "weight_type": flag,
            "editable": editable,
            "confirmed": row_confirmed,
            "h_after": float(h_after),
            "boundary_break": (not fixed) and abs(float(a_actual) - float(a_req)) > 1e-3,
            "label_valid_under_mask": bool(label_ok),
            "used_hc_guard": info["used_hc_guard"],
            "used_fixed_rule": info["used_fixed_rule"],
            "include_in_bc": include,
            "exec_protocol": EXEC_PROTOCOL_VERSION,
        })

        if i + 1 >= len(records):
            break
        next_d = records[i + 1][1]["_d"]
        nxt = state_space[state_space["_d"] == next_d]
        if nxt.empty:
            raise ValueError(f"重放缺下一日 {next_d}")
        next_state = nxt.iloc[0].copy()
        next_state["h_before"] = float(h_after)
        next_state["ks"] = ks_factor(float(next_state.h_before))
        state = next_state

    if illegal:
        msg = (
            f"非法请求标签 {len(illegal)} 日（BC 只学合法请求档，不自动改标签）。"
            f" 例: {illegal[0]}"
        )
        raise ValueError(msg)

    return samples


def recompute_consistent_trajectory(
    review_df: pd.DataFrame,
    env: Optional[PPO_Rice_env.Irrigation] = None,
    *,
    cfg: Optional[ExecConfig] = None,
    force_hc_guard: Optional[bool] = None,
    include_fixed_rule_in_bc: bool = False,
) -> pd.DataFrame:
    """按定稿请求动作从季初连续重放（不调 PPO）。

    多年按 ``year`` 分段；季初均为 ``hmax``。
    """
    table_errs = validate_review_table(review_df)
    if table_errs:
        raise ValueError("审阅表校验失败: " + "; ".join(table_errs[:8]))

    if review_df.empty:
        return pd.DataFrame()

    cfg = cfg or EXEC_DEFAULT
    env = env or PPO_Rice_env.Irrigation()
    df = review_df.copy()
    if "year" not in df.columns:
        df["year"] = df["date"].map(lambda d: norm_date(d).split("/")[0])

    samples: list[dict[str, Any]] = []
    for _, season in df.groupby("year", sort=True):
        samples.extend(
            _recompute_one_season(
                season,
                env,
                cfg=cfg,
                force_hc_guard=force_hc_guard,
                include_fixed_rule_in_bc=include_fixed_rule_in_bc,
            )
        )
    return pd.DataFrame(samples)


def to_bc_arrays(consistent_df: pd.DataFrame, *, only_include_in_bc: bool = True):
    df = consistent_df
    if only_include_in_bc and "include_in_bc" in df.columns:
        df = df[df["include_in_bc"].astype(bool)].reset_index(drop=True)
    if df.empty:
        raise ValueError("没有可用于 BC 的样本（可能全部为固定规则日）")
    if "label_valid_under_mask" in df.columns and not bool(df["label_valid_under_mask"].all()):
        bad = df.loc[~df["label_valid_under_mask"].astype(bool), "date"].astype(str).tolist()
        raise ValueError(f"存在非法掩码标签，禁止进训: {bad[:10]}")
    if (df["a_idx"].to_numpy(dtype=np.int64) < 0).any():
        raise ValueError("存在无效 a_idx，禁止进训")
    states = np.asarray(df["s_vec"].tolist(), dtype=np.float32)
    actions = df["a_idx"].to_numpy(dtype=np.int64)
    flags = df["flag"].to_numpy() if "flag" in df.columns else np.array(["confirmed"] * len(df))
    return states, actions, flags


def save_labels(consistent_df: pd.DataFrame, path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = path if path.suffix.lower() == ".json" else path.with_suffix(".json")
    consistent_df.to_json(out, orient="records", force_ascii=False)
    return out


def verify_review_vs_labels(
    review_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    atol: float = 1e-3,
) -> dict[str, Any]:
    """校验定稿轨迹与连续重放导出：日期、晨水、请求、执行量、日末、状态向量。"""
    if review_df is None or review_df.empty or labels_df is None or labels_df.empty:
        return {
            "ok": False,
            "n_compared": 0,
            "n_mismatch": 0,
            "mismatches": [],
            "error": "审阅表或 labels 为空",
        }

    table_errs = validate_review_table(review_df)
    if table_errs:
        return {
            "ok": False,
            "n_compared": 0,
            "n_mismatch": len(table_errs),
            "mismatches": [{"field": "table", "error": e} for e in table_errs[:30]],
            "error": "审阅表结构/数值校验失败",
        }

    rev = review_df.copy()
    lab = labels_df.copy()
    rev["_d"] = rev["date"].map(norm_date)
    lab["_d"] = lab["date"].map(norm_date)

    rev_dates = set(rev["_d"])
    lab_dates = set(lab["_d"])
    missing = sorted(rev_dates - lab_dates)
    extra = sorted(lab_dates - rev_dates)
    mismatches: list[dict[str, Any]] = []
    for d in missing[:20]:
        mismatches.append({"date": d, "field": "missing_in_labels"})
    for d in extra[:20]:
        mismatches.append({"date": d, "field": "extra_in_labels"})

    lab_by = {r["_d"]: r for _, r in lab.iterrows()}
    n_compared = 0
    for _, r in rev.iterrows():
        d = r["_d"]
        L = lab_by.get(d)
        if L is None:
            continue
        n_compared += 1
        req_r = r.get("a_expert") if expert_action_filled(r.get("a_expert")) else r.get("a_model")
        req_l = L.get("a_requested_mm", L.get("a_expert"))
        for field, rv, lv in (
            ("h_before", r.get("h_before"), L.get("h_before")),
            ("h_after", r.get("h_after"), L.get("h_after")),
            ("a_exec", r.get("a_exec"), L.get("a_executed_mm", L.get("a_actual"))),
            ("a_requested", req_r, req_l),
        ):
            if pd.isna(rv) or pd.isna(lv):
                mismatches.append({"date": d, "field": field, "error": "缺失"})
                continue
            if abs(float(rv) - float(lv)) > atol:
                mismatches.append({
                    "date": d,
                    "field": field,
                    "review": float(rv),
                    "labels": float(lv),
                })
        sv_r = r.get("state_vec") if "state_vec" in r.index else None
        sv_l = L.get("s_vec") if "s_vec" in L.index else None
        if sv_r is not None and sv_l is not None:
            try:
                a = np.asarray(json.loads(sv_r) if isinstance(sv_r, str) else sv_r, dtype=np.float64).reshape(-1)
                b = np.asarray(sv_l, dtype=np.float64).reshape(-1)
                if a.shape != (8,) or b.shape != (8,) or not np.isfinite(a).all() or not np.isfinite(b).all():
                    mismatches.append({"date": d, "field": "state_vec/s_vec", "error": "维度或有限值校验失败"})
                elif not np.allclose(a, b, atol=atol, rtol=0):
                    mismatches.append({
                        "date": d,
                        "field": "state_vec/s_vec",
                        "max_abs": float(np.nanmax(np.abs(a - b))),
                    })
            except Exception as e:
                mismatches.append({"date": d, "field": "state_vec/s_vec", "error": str(e)})
        else:
            mismatches.append({"date": d, "field": "state_vec/s_vec", "error": "缺失"})
        if "label_valid_under_mask" in L.index and not bool(L.get("label_valid_under_mask", True)):
            mismatches.append({"date": d, "field": "illegal_request_label"})

    return {
        "ok": len(mismatches) == 0 and n_compared > 0,
        "n_compared": n_compared,
        "n_mismatch": len(mismatches),
        "mismatches": mismatches[:50],
        "exec_protocol": EXEC_PROTOCOL_VERSION,
    }
