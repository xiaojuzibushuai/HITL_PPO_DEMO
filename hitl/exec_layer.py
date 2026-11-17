"""兼容入口：审阅、导出、评估统一使用 common.execution。"""
from common.execution import (
    EXEC_PROTOCOL_VERSION, apply_action, apply_expert_action, same_request_same_exec,
    fixed_rule_action_mm, fixed_rule_label, is_fixed_rule_stage, stage_name_from_state,
    expert_action_filled, expert_action_float, norm_date, date_ymd,
    allowed_actions, validate_requested_action,
)
