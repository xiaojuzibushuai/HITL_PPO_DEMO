"""预训练 PPO 整季 rollout → 审阅表（决策走一期 ``decision.infer``）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import train.PPO_Rice_env as PPO_Rice_env
from common.action_mask import mask_policy_logits
from decision.infer import decide_from_logits

from .config import ACTION_SPACE, EXEC_DEFAULT, REVIEW_DEFAULT, ExecConfig, ReviewConfig
from .exec_layer import fixed_rule_action_mm, is_fixed_rule_stage, stage_name_from_state
from common.execution import apply_action, EXEC_PROTOCOL_VERSION, allowed_actions


def _mark_small_runs(
    irrig_mm: list[float],
    *,
    dates=None,
    editable=None,
    used_fixed_rule=None,
    max_abs_mm: float = 20.0,
    min_days: int = 2,
) -> list[bool]:
    """审阅高亮：连续小额灌溉（执行灌水量）主指标 + 续灌后续小额日。

    灌排分开；排水不并入本标记。默认 ≥2 d、每日 0<m≤20 mm（与论文主口径一致）。
    """
    from evaluation.operation_runs import mark_small_irrigation_review_days

    return mark_small_irrigation_review_days(
        irrig_mm,
        dates=dates,
        editable=editable,
        used_fixed_rule=used_fixed_rule,
        max_mm=max_abs_mm,
        min_main_days=min_days,
    )


def list_policy_years(env: Optional[PPO_Rice_env.Irrigation] = None) -> list[str]:
    env = env or PPO_Rice_env.Irrigation()
    dates = env.state_space_policy["date"].astype(str)
    return sorted({d.split("/")[0] for d in dates if "/" in d})


def rollout_season(
    agent,
    env: Optional[PPO_Rice_env.Irrigation] = None,
    *,
    max_steps: int = 200,
    year: Optional[str] = None,
    cfg: Optional[ExecConfig] = None,
    review_cfg: Optional[ReviewConfig] = None,
) -> pd.DataFrame:
    cfg = cfg or EXEC_DEFAULT
    review_cfg = review_cfg or REVIEW_DEFAULT
    env = env or PPO_Rice_env.Irrigation()
    state_space = env.state_space_policy
    actual_state_space = env.actual_state_space_policy

    env.current_year_idx = 0
    state, actual_state_pd = env.reset_policy(state_space)
    if year is not None:
        for _ in list_policy_years(env):
            if str(state.date).split("/")[0] == str(year):
                break
            state, actual_state_pd = env.reset_policy(state_space)

    rows: List[dict[str, Any]] = []
    done = False
    step = 0
    use_mask = bool(getattr(agent, "use_action_mask", True))
    action_space = list(getattr(agent, "action_space", None) or ACTION_SPACE)

    while not done and step < max_steps:
        s_vec = np.array(env.process_series(state), dtype=np.float32)
        stage = stage_name_from_state(state)
        fixed = is_fixed_rule_stage(state, stage_name=stage, cfg=cfg)

        x = torch.FloatTensor(s_vec).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            logits, _ = agent.policy_net(x)
        dec = decide_from_logits(
            logits,
            s_vec,
            action_space=action_space,
            use_action_mask=use_mask,
            hmin=float(state.hmin),
            hmax=float(state.hmax),
            hp=float(state.Hp),
            hc=float(state.Hc),
            p_cumu7=float(state.P_cumu7),
            hc_guard=bool(cfg.force_hc_guard),
            physical_clip=True,
            drying_rule=False,
            low_water_refill=False,
        )
        a_model = float(dec["raw_action"])
        a_idx = int(dec["action_idx"])
        masked = mask_policy_logits(logits, x, action_space) if use_mask else logits
        probs = F.softmax(masked, dim=1).detach().cpu().numpy()[0]

        used_guard = used_fixed = False
        if fixed:
            # 与 apply_action(fixed=True) 同序：离散请求 → correct_policy → env.step/correct
            from decision.rules import correct_policy_action

            a_req = float(fixed_rule_action_mm(state))
            a_idx = int(np.argmin([abs(a - a_req) for a in action_space]))
            a_mm = float(
                correct_policy_action(
                    a_req,
                    float(state.h_before),
                    float(state.hmax),
                    float(state.Hp),
                    float(state.Hc),
                )
            )
            used_fixed = True
            editable = False
            a_expert = float(a_req)
            a_model = float(a_req)
        else:
            a_mm = float(dec["action"])  # 经守卫/限幅后的执行量
            used_guard = bool(dec.get("safety_override"))
            editable = True
            # 专家列默认空：仅人在看板选择后才写入 a_expert
            a_expert = None

        actual = actual_state_space.loc[actual_state_space["date"] == state.date].iloc[0]
        expected_h, a_mm, exec_info = apply_action(env, state, actual, a_model, cfg=cfg)
        next_state, reward, done, a_exec = env.step(
            actual_state_pd, actual_state_space, state_space, state, a_mm
        )
        row = {
            "step": step,
            "year": str(state.date).split("/")[0],
            "date": state.date,
            "stage": stage,
            "h_before": float(state.h_before),
            "h_after": float(next_state.h_before),
            "exec_protocol": EXEC_PROTOCOL_VERSION,
            "allowed_actions": allowed_actions(state),
            "needs_reconfirm": False,
            "hmin": float(state.hmin),
            "hmax": float(state.hmax),
            "Hp": float(state.Hp),
            "Hc": float(state.Hc),
            "lamda": float(state.lamda),
            "ks": float(state.ks),
            "P_cumu7": float(state.P_cumu7) if review_cfg.show_forecast_rain else None,
            "a_model": a_model,
            "a_idx": a_idx,
            "a_exec": float(a_exec),
            "a_applied": float(a_mm),
            "a_expert": a_expert,
            "confirmed": False,
            "editable": editable,
            "used_fixed_rule": used_fixed,
            "used_hc_guard": used_guard,
            "state_vec": s_vec.tolist(),
            "reward": float(reward[0]) if isinstance(reward, (tuple, list)) else float(reward),
        }
        for i, p in enumerate(probs):
            row[f"prob_{i}"] = float(p)
        rows.append(row)
        state = next_state
        step += 1

    df = pd.DataFrame(rows)
    if len(df):
        # 用实际执行灌水量（正 a_exec），固定规则日打断连续段
        irrig = [
            max(0.0, float(a)) if e else 0.0
            for a, e in zip(df["a_exec"].tolist(), df["editable"].tolist())
        ]
        df["is_small_run"] = _mark_small_runs(
            irrig,
            dates=df["date"].tolist(),
            editable=df["editable"].tolist(),
            used_fixed_rule=df["used_fixed_rule"].tolist()
            if "used_fixed_rule" in df.columns
            else None,
        )
    return df


def rollout_all_years(
    agent,
    env: Optional[PPO_Rice_env.Irrigation] = None,
    *,
    years: Optional[List[str]] = None,
    cfg: Optional[ExecConfig] = None,
    review_cfg: Optional[ReviewConfig] = None,
) -> pd.DataFrame:
    env = env or PPO_Rice_env.Irrigation()
    all_years = list_policy_years(env)
    target = [str(y) for y in years] if years else all_years
    frames = []
    for y in target:
        if y not in all_years:
            continue
        df_y = rollout_season(agent, env, year=y, cfg=cfg, review_cfg=review_cfg)
        if len(df_y):
            frames.append(df_y)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_review_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import json
    out = df.copy()
    if "state_vec" in out:
        out["state_vec"] = out["state_vec"].map(lambda v: v if isinstance(v, str) else json.dumps(v))
    if path.suffix.lower() in {".xlsx", ".xls"}:
        out.to_excel(path, index=False)
    else:
        out.to_csv(path, index=False, encoding="utf-8-sig")
    return path
