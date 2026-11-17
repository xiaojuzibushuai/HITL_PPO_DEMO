"""执行层：生育期判定、日期归一、动作落地（复用一期 ``decision.rules``）。

``rollout`` / ``resync`` / ``recompute`` 共用，避免各写一套 Hc/晒田逻辑。
"""
from __future__ import annotations

from typing import Any, Optional
from types import SimpleNamespace
import numbers
import numpy as np
import torch

import pandas as pd
import train.PPO_Rice_env as PPO_Rice_env
from decision.rules import (
    apply_safety_drainage_guard,
    correct_policy_action,
    drying_stage_action,
)

from common.action_mask import ACTION_SPACE, valid_action_mask, snap_to_action_space
FIXED_RULE_STAGE_NAMES = ["晒田", "落干", "Sun_dry", "SunDry", "Drainage", "Mid_drainage", "Mid_drying", "Final_drainage", "Harvest"]
EXEC_PROTOCOL_VERSION = "2026.10.25-hitl-exec-v3"
ExecConfig = Any
EXEC_DEFAULT = SimpleNamespace(force_hc_guard=True, enable_fixed_rule_stages=True,
    fixed_rule_stage_names=FIXED_RULE_STAGE_NAMES, fixed_rule_hmin_threshold=-50.0)


def expert_action_filled(val) -> bool:
    """审阅表专家列是否已由人填写（空/NaN 表示未填，递推仍用模型动作）。"""
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(val, str) and not str(val).strip():
        return False
    return True


def expert_action_float(val) -> float | None:
    """已填专家列 → 离散档位 mm；未填返回 ``None``（供 API / 类型检查用）。"""
    if not expert_action_filled(val):
        return None
    return float(snap_to_action_space(val))


def norm_date(date) -> str:
    """统一成 ``Y/M/D``（无前导零），兼容 Timestamp / ISO。

    注意：无前导零时字符串序错误（``2012/10/1`` < ``2012/6/15``）。
    凡按日排序请用 :func:`date_ymd`，勿对 ``norm_date`` 结果做字典序排序。
    """
    if isinstance(date, numbers.Real):
        date = pd.to_datetime(date, unit="ms", utc=True)
    if hasattr(date, "strftime"):
        return f"{date.year}/{date.month}/{date.day}"
    s = str(date).strip().replace("-", "/")
    if " " in s:
        s = s.split(" ")[0]
    parts = s.split("/")
    if len(parts) != 3:
        raise ValueError(f"无法识别日期: {date!r}")
    y, m, d = parts
    return f"{int(y)}/{int(m)}/{int(d)}"


def date_ymd(date) -> tuple[int, int, int]:
    """日历序键 ``(年, 月, 日)``，供季内按真实时间排序。"""
    s = norm_date(date)
    parts = s.split("/")
    if len(parts) != 3:
        return (0, 0, 0)
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return (0, 0, 0)


def stage_name_from_state(state) -> str:
    try:
        name, _ = PPO_Rice_env.Irrigation.get_growth_stage(str(state.date))
        return str(name)
    except Exception:
        return ""


def is_fixed_rule_stage(
    state,
    stage_name: Optional[str] = None,
    cfg: Optional[ExecConfig] = None,
) -> bool:
    """晒田/落干/收获排水等：不进模型决策、默认不进 BC。"""
    return fixed_rule_label(state, stage_name=stage_name, cfg=cfg) is not None


def fixed_rule_label(
    state=None,
    stage_name: Optional[str] = None,
    *,
    hmin: Optional[float] = None,
    cfg: Optional[ExecConfig] = None,
) -> Optional[str]:
    """固定规则日的短标签；非固定日返回 ``None``。

    - 晒田：Mid_drying / 名称含 dry·晒田
    - 落干：Harvest / hmin 极低 / 名称含落干·drainage
    """
    cfg = cfg or EXEC_DEFAULT
    if not cfg.enable_fixed_rule_stages:
        return None
    name = stage_name
    if name is None and state is not None:
        name = stage_name_from_state(state)
    name_l = (name or "").lower()
    names = {n.lower() for n in cfg.fixed_rule_stage_names}
    if name_l in names or any(k.lower() in name_l for k in FIXED_RULE_STAGE_NAMES):
        if any(k in name_l for k in ("dry", "晒", "sun")):
            return "晒田"
        if any(k in name_l for k in ("harvest", "落干", "drain")):
            return "落干"
        return "固定规则"
    try:
        hm = float(hmin) if hmin is not None else float(state.hmin)  # type: ignore[union-attr]
        if hm <= float(cfg.fixed_rule_hmin_threshold):
            return "落干" if (name_l.startswith("harvest") or "落" in (name or "")) else "晒田"
    except Exception:
        pass
    return None


def fixed_rule_action_mm(state) -> float:
    """晒田/落干日的离散动作（mm 档）。"""
    h = float(state.h_before)
    hmax = float(state.hmax)
    hp = float(state.Hp)
    hc = float(state.Hc)
    da = drying_stage_action(h, hmax, hp, hc)
    if da is not None:
        return float(snap_to_action_space(da))
    if h > max(hp, 0.0):
        return float(snap_to_action_space(max(-100.0, min(0.0, hp - h))))
    return 0.0


def apply_action(
    env: Optional[PPO_Rice_env.Irrigation],
    state,
    actual_state,
    action_mm: float,
    *,
    force_hc_guard: Optional[bool] = None,
    fixed_rule_stage: Optional[bool] = None,
    cfg: Optional[ExecConfig] = None,
) -> tuple[Any, float, dict]:
    """模型/固定规则「日常」链：请求 →（晒田离散档 | Hc 守卫）→ ``correct_policy_action`` → ``env.correct``。

    固定规则日与 ``rollout`` 同序：``fixed_rule_action_mm`` → 物理限幅 → ``env.correct``。
    不含：``decide_from_logits`` 里的 ``low_water_refill``（审阅 resync 的 PPO 建议也未开）。
    """
    cfg = cfg or EXEC_DEFAULT
    env = env or PPO_Rice_env.Irrigation()
    use_guard = cfg.force_hc_guard if force_hc_guard is None else force_hc_guard
    if fixed_rule_stage is None:
        fixed_rule_stage = is_fixed_rule_stage(state, cfg=cfg)

    h = float(state.h_before)
    hmax = float(state.hmax)
    hp = float(state.Hp)
    hc = float(state.Hc)
    p7 = float(getattr(state, "P_cumu7", 0.0) or 0.0)

    requested = float(action_mm)
    action = requested
    used_fixed = used_guard = False

    if fixed_rule_stage:
        # 与 rollout 统一：先取离散晒田/落干请求档，再 correct → env.correct
        used_fixed = True
        action = float(fixed_rule_action_mm(state))
        requested = action
    elif use_guard:
        guarded = apply_safety_drainage_guard(action, h, hp, hc, p7)
        if guarded != action:
            used_guard = True
            action = guarded

    action = correct_policy_action(action, h, hmax, hp, hc)
    h_after, actual_action = env.correct(state, actual_state, action)
    return h_after, float(actual_action), {
        "requested_action": requested,
        "used_fixed_rule": used_fixed,
        "used_hc_guard": used_guard,
        "editable": not used_fixed,
        "exec_protocol": EXEC_PROTOCOL_VERSION,
    }


def apply_expert_action(
    env: Optional[PPO_Rice_env.Irrigation],
    state,
    actual_state,
    expert_mm: float,
    *,
    cfg: Optional[ExecConfig] = None,
    force_hc_guard: Optional[bool] = None,
) -> tuple[Any, float, dict]:
    """专家**请求**动作走与 PPO 相同的执行约束（验收：同状态+同请求 → 同执行量/下一状态）。

    链路与 ``apply_action(..., fixed_rule_stage=False)`` 一致：
    Hc 守卫（默认开）→ ``correct_policy_action`` → ``env.correct``。
    返回的 ``a_exec`` 是环境实际执行量，可能与口头请求档位不同。
    """
    return apply_action(
        env,
        state,
        actual_state,
        validate_requested_action(state, expert_mm),
        force_hc_guard=force_hc_guard,
        fixed_rule_stage=False,
        cfg=cfg,
    )


def same_request_same_exec(
    env: Optional[PPO_Rice_env.Irrigation],
    state,
    actual_state,
    action_mm: float,
    *,
    cfg: Optional[ExecConfig] = None,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """校验：同一请求经 ``apply_action`` 与 ``apply_expert_action`` 结果一致。"""
    cfg = cfg or EXEC_DEFAULT
    h1, a1, i1 = apply_action(
        env, state, actual_state, action_mm, fixed_rule_stage=False, cfg=cfg,
    )
    h2, a2, i2 = apply_expert_action(
        env, state, actual_state, action_mm, cfg=cfg,
    )
    ok = abs(float(h1) - float(h2)) <= atol and abs(float(a1) - float(a2)) <= atol
    return {
        "ok": ok,
        "h_after_action": float(h1),
        "h_after_expert": float(h2),
        "a_exec_action": float(a1),
        "a_exec_expert": float(a2),
        "used_hc_guard": bool(i1.get("used_hc_guard")),
    }


def allowed_actions(state) -> list[int]:
    x = torch.tensor([[float(state[c]) for c in ("h_before", "hmin", "hmax", "Hp", "Hc")]], dtype=torch.float32)
    return [a for a, ok in zip(ACTION_SPACE, valid_action_mask(x)[0].tolist()) if ok]


def validate_requested_action(state, requested) -> float:
    a = float(requested)
    if not np.isfinite(a) or a not in allowed_actions(state):
        raise ValueError(f"{state.date}: 非法请求 {a} mm；当前合法档 {allowed_actions(state)}")
    return a

