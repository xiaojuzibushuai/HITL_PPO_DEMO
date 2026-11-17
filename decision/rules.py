"""决策后处理规则（与智灌复现包 zhiguan_decision.rules 对齐）。

执行顺序（复现包注释原文）：
  模型动作 → 安全守卫（h >= Hc 强制 -100）→ 物理限幅 → 执行

为何先 -100 再限幅是合理的：
  - 守卫只表达「必须大排」的意图（动作空间最大排水档）；
  - correct_policy_action 再按目标水位 min(Hp,Hc) 截到物理可执行量，
    避免一次排穿目标（例如 h=80、目标 60 时 -100 会被截成 -20）。
  掩码在 h>=Hc 时已只留 -100；守卫是 unmasked / 旁路时的硬保险，与复现包 decide_day 一致。

ks 统一用 common.scenario.ks_factor。
"""
from __future__ import annotations

import numpy as np

from common.scenario import ks_factor  # noqa: F401 — 再导出供外部 from rules import ks_factor


def safety_drainage_required(h: float, hp: float, hc: float, p_cumu7: float = 0.0) -> bool:
    """水层达到耐淹上限 Hc 时必须排水（与复现包同名函数一致）。"""
    return float(h) >= float(hc)


def apply_safety_drainage_guard(
    raw_action: float, h: float, hp: float, hc: float, p_cumu7: float = 0.0
) -> float:
    """安全守卫：h >= Hc → -100；随后由 correct_policy_action 截到目标水层。"""
    return -100.0 if safety_drainage_required(h, hp, hc, p_cumu7) else float(raw_action)


def drying_stage_action(h: float, hmax: float, hp: float, hc: float) -> float | None:
    """晒田/落干（hmax<=0）：有水则排到目标，否则不动；非晒田返回 None。"""
    if float(hmax) > 0:
        return None
    h = float(h)
    target = min(float(hp), float(hc)) if h >= float(hc) else float(hp)
    return max(-100.0, target - h) if h > target else 0.0


def correct_policy_action(raw_action: float, h: float, hmax: float, hp: float, hc: float) -> float:
    """物理限幅：灌不超过 hmax；排不低于目标（h>=Hc 时 min(Hp,Hc)，否则 Hp）。"""
    raw_action = float(np.clip(float(raw_action), -100.0, 100.0))
    h = float(h)
    if raw_action > 0:
        return min(raw_action, max(0.0, float(hmax) - h))
    if raw_action < 0:
        target = min(float(hp), float(hc)) if h >= float(hc) else float(hp)
        if h <= target:
            return 0.0
        return max(raw_action, target - h)
    return 0.0
