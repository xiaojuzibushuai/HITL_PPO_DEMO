"""动作掩码（与智灌复现包 valid_action_mask 一致）。训练与推理共用。"""
from __future__ import annotations

import torch
import math

ACTION_SPACE = [
    -100, -90, -80, -70, -60, -50, -40, -30, -20, -10,
    0,
    10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
]


def snap_to_action_space(action_mm):
    """仅供规则请求量量化；专家输入须另做严格校验。"""
    a = float(action_mm)
    if not math.isfinite(a):
        raise ValueError("请求量必须为有限数值")
    return max(-100, min(100, int(round(a / 10) * 10)))


def valid_action_mask(states: torch.Tensor, action_space=None) -> torch.Tensor:
    """状态前 5 维：h, hmin, hmax, Hp, Hc。"""
    if action_space is None:
        action_space = ACTION_SPACE
    if states.dim() == 1:
        states = states.unsqueeze(0)
    actions = torch.as_tensor(action_space, dtype=states.dtype, device=states.device)
    h = states[:, 0:1]
    hmax = states[:, 2:3]
    hp = states[:, 3:4]
    hc = states[:, 4:5]
    grid = actions.unsqueeze(0)

    emergency = h >= hc
    emergency_valid = emergency & (grid == actions.min())

    drain_capacity = torch.clamp(h - hp, min=0.0)
    drain_limit = torch.clamp(torch.ceil(drain_capacity / 10.0) * 10.0, max=100.0)
    drain_valid = (grid < 0) & (h > hp) & (-grid <= drain_limit)

    irrigation_capacity = torch.clamp(hmax - h, min=0.0)
    irrigation_limit = torch.clamp(torch.ceil(irrigation_capacity / 10.0) * 10.0, max=100.0)
    irrigation_valid = (grid > 0) & (h < hmax) & (grid <= irrigation_limit) & (h < hp)

    ordinary_valid = drain_valid | irrigation_valid | (grid == 0)
    return emergency_valid | ((~emergency) & ordinary_valid)


def mask_policy_logits(logits: torch.Tensor, states: torch.Tensor, action_space=None) -> torch.Tensor:
    if action_space is None:
        action_space = ACTION_SPACE
    mask = valid_action_mask(states, action_space)
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
