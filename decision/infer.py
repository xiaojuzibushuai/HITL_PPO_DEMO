"""统一决策入口：掩码后动作 →（可选晒田）→ Hc 守卫 → 物理限幅 →（可选低水层补灌）。"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from common.action_mask import ACTION_SPACE, mask_policy_logits
from decision.rules import (
    apply_safety_drainage_guard,
    correct_policy_action,
    drying_stage_action,
)


def fixed_rule_action(h: float, hmin: float, hmax: float, hp: float, hc: float) -> float:
    """智灌式固定规则基线（当日水层阈值，mm；不做 7 日前瞻）。

    - h < hmin → 灌到 hmax（返回 hmax - h）
    - h ≥ Hc 或 h > Hp → 排到目标（超 Hc 用 min(Hp,Hc)，否则 Hp）
    - 否则 0

    评估时与 PPO 共用 env.step 水量递推；本函数只负责「选动作」。
    """
    h, hmin, hmax, hp, hc = map(float, (h, hmin, hmax, hp, hc))
    if h < hmin:
        return hmax - h
    if h >= hc or h > hp:
        target = min(hp, hc) if h >= hc else hp
        return target - h
    return 0.0


def apply_decision_pipeline(
    raw_action: float,
    *,
    h: float,
    hmin: float,
    hmax: float,
    hp: float,
    hc: float,
    p_cumu7: float = 0.0,
    hc_guard: bool = True,
    physical_clip: bool = True,
    drying_rule: bool = False,
    low_water_refill: bool = False,
) -> dict[str, Any]:
    """对模型/规则原始动作做后处理，返回最终可执行 mm。

    顺序：可选晒田规则 → Hc 守卫(-100) → 物理限幅 → 可选低水补灌。
    返回含 raw_action / action / dtype / desc / safety_reason 等。
    """
    h, hmin, hmax, hp, hc = map(float, (h, hmin, hmax, hp, hc))
    raw = float(raw_action)
    stage_rule = False
    applied = raw

    if drying_rule and hmax <= 0.0:
        da = drying_stage_action(h, hmax, hp, hc)
        if da is not None:
            applied = float(da)
            stage_rule = True

    guarded = applied
    if hc_guard and not stage_rule:
        guarded = apply_safety_drainage_guard(applied, h, hp, hc, p_cumu7)

    action = correct_policy_action(guarded, h, hmax, hp, hc) if physical_clip else float(guarded)

    safety_reason = ""
    if hc_guard and guarded != applied:
        # 文案对齐智灌 decide_day
        safety_reason = f"当前水深达到 Hc={hc:.0f}mm，安全守卫请求最大排水"
    elif low_water_refill and h < hmin - 20:
        needed = min(60.0, max(10.0, hmin - h))
        action = correct_policy_action(needed, h, hmax, hp, hc) if physical_clip else needed
        safety_reason = f"low-water refill (h < hmin-20)"

    if action > 1e-9:
        dtype, desc = "irrigate", f"灌水 {int(round(action))} mm"
    elif action < -1e-9:
        dtype, desc = "drain", f"排水 {int(round(abs(action)))} mm"
    else:
        dtype, desc = "noop", "维持水位"

    return {
        "raw_action": raw,
        "applied_action": applied,
        "guarded_action": guarded,
        "action": float(action),
        "dtype": dtype,
        "desc": desc,
        "stage_rule": stage_rule,
        "safety_override": bool(guarded != applied or (low_water_refill and safety_reason.startswith("low"))),
        "safety_reason": safety_reason,
    }


def decide_from_logits(
    logits: torch.Tensor,
    state_vec: np.ndarray | torch.Tensor,
    *,
    action_space=None,
    use_action_mask: bool = True,
    hmin: float,
    hmax: float,
    hp: float,
    hc: float,
    p_cumu7: float = 0.0,
    hc_guard: bool = True,
    physical_clip: bool = True,
    drying_rule: bool = False,
    low_water_refill: bool = False,
) -> dict[str, Any]:
    """策略 logits →（可选掩码）argmax → ``apply_decision_pipeline``。

    Args:
        logits: 网络原始动作打分，形状 (1, n_actions) 或 (n_actions,)
        state_vec: 至少含 h 的状态（取第 0 维作水深）；用于掩码
        action_space: mm 档列表，默认 ACTION_SPACE
        use_action_mask: True 时非法档 logits→-inf（如 h≥Hc 只留 -100）
        hmin/hmax/hp/hc: 当日水位约束（mm）
        p_cumu7: 7 日雨预报（守卫接口保留，当前守卫主要看 h≥Hc）
        hc_guard: 超 Hc 强制请求 -100
        physical_clip: 再截到灌≤hmax、排≥目标
        drying_rule: hmax≤0 晒田阶段规则
        low_water_refill: h < hmin-20 时强制补灌（在线 predict 开，eval 默认关）
    """
    action_space = list(action_space) if action_space is not None else list(ACTION_SPACE)
    if isinstance(state_vec, np.ndarray):
        st = torch.FloatTensor(state_vec).unsqueeze(0)
    else:
        st = state_vec if state_vec.dim() == 2 else state_vec.unsqueeze(0)

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    h = float(st[0, 0].item())
    unmasked_idx = int(torch.argmax(logits, dim=1).item())
    masked = mask_policy_logits(logits, st, action_space) if use_action_mask else logits
    probs = F.softmax(masked, dim=1).detach().cpu().numpy()[0]
    idx = int(np.argmax(probs))
    raw_action = float(action_space[idx])

    post = apply_decision_pipeline(
        raw_action,
        h=h,
        hmin=hmin,
        hmax=hmax,
        hp=hp,
        hc=hc,
        p_cumu7=p_cumu7,
        hc_guard=hc_guard,
        physical_clip=physical_clip,
        drying_rule=drying_rule,
        low_water_refill=low_water_refill,
    )
    post.update(
        {
            "action_idx": idx,
            "confidence": float(probs[idx]),
            "unmasked_action": float(action_space[unmasked_idx]),
            "masked": bool(use_action_mask),
        }
    )
    return post
