"""HITL 配置 + 动作档工具（与 ``common.action_mask`` 一致）。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from common.action_mask import ACTION_SPACE

# 晒田 / 落干 / 收获排水：固定规则，不进模型决策、默认不进行为克隆(BC)
FIXED_RULE_STAGE_NAMES = [
    "晒田",
    "落干",
    "Sun_dry",
    "SunDry",
    "Drainage",
    "Mid_drainage",
    "Mid_drying",
    "Final_drainage",
    "Harvest",  # 收获期 hmin 极低，有水就排，不需专家审阅
]

# ---------------------------------------------------------------------------
# 评估集年份划分（当前评估集约 8 年：2012,2013,2014,2016,2017,2018,2019,2020）
#
# - 审阅年 HITL_REVIEW_YEARS：专家改写 + 行为克隆监督样本来源
# - 留出验证年 HITL_HOLDOUT_YEARS：专家默认不改；用于报告「未见年」泛化
#
# 二方对比（原 PPO vs 规则）：始终用评估集全部年份（与训评 02_eval 一致），无此划分。
# 三方对比（规则 / 原 PPO / HITL）：默认也跑评估集全部年份（与二方可对齐）；
#   同时另汇总「仅留出验证年」子集，避免只在专家改过的年上夸大 HITL。
# ---------------------------------------------------------------------------
HITL_REVIEW_YEARS: list[str] = ["2012", "2013", "2014", "2016"]
HITL_HOLDOUT_YEARS: list[str] = ["2017", "2018", "2019", "2020"]

# 结构化改因（论文机制标签）。等雨 / 允许偏干必须分开，禁止用「没灌水+有预报」事后推断。
REASON_CODE_LABELS: dict[str, str] = {
    "merge_small_irrigation": "合并小额灌水",
    "wait_forecast_rain": "依据预报暂缓灌水",
    "allow_low_water": "允许较低水层",
    "other": "其他",
}
REASON_CODE_ORDER: list[str] = [
    "merge_small_irrigation",
    "wait_forecast_rain",
    "allow_low_water",
    "other",
]

# P_cumu7 时窗（与 actual 表 P_1_a…P_7_a 对齐）：含决策日共 7 日 = 当日 + 随后 6 日。
P_CUMU7_WINDOW = {
    "includes_decision_day": True,
    "n_days": 7,
    "actual_cols": ["P_1_a", "P_2_a", "P_3_a", "P_4_a", "P_5_a", "P_6_a", "P_7_a"],
    "note": "P_1_a=决策日实测雨；P_2_a…P_7_a=随后 6 日。核对预报兑现时用同行这 7 列之和。",
}

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_SPACE)}


def action_to_idx(action_mm: float | int) -> int:
    a = int(round(float(action_mm) / 10.0) * 10)
    a = max(-100, min(100, a))
    if a not in ACTION_TO_IDX:
        raise ValueError(f"动作 {action_mm} 不在 ACTION_SPACE 中")
    return ACTION_TO_IDX[a]


def snap_to_action_space(action_mm: float | int) -> int:
    a = int(round(float(action_mm) / 10.0) * 10)
    return max(-100, min(100, a))


@dataclass
class ExecConfig:
    force_hc_guard: bool = True
    enable_fixed_rule_stages: bool = True
    fixed_rule_stage_names: list[str] = field(default_factory=lambda: list(FIXED_RULE_STAGE_NAMES))
    fixed_rule_hmin_threshold: float = -50.0


@dataclass
class ReviewConfig:
    show_forecast_rain: bool = True
    # 旧配置兼容字段；rollout 忽略此开关，永远不导出实测 P_1_a
    show_actual_rain: bool = False
    actual_rain_placeholder: str = "?"


@dataclass
class BCConfig:
    """在原 PPO 权重上做 CE+KL 监督微调（不调用 PPO.update）。

    - ``train_policy_head_only`` + 冻结 value：保住表征与价值估计
    - ``kl_beta``：相对微调前旧策略的 KL，防止忘掉 PPO
    - ``w_edit``：专家改动日样本权重，拉向人类习惯
    """

    epochs: int = 80
    batch_size: int = 32
    lr: float = 3e-5
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    max_grad_norm: float = 0.5
    w_edit: float = 6.0
    kl_beta: float = 0.08
    freeze_value_head: bool = True
    train_policy_head_only: bool = True
    # 第一版关闭平滑：避免把概率质量摊到掩码外类别；以后再做「仅合法档平滑」
    label_smoothing: float = 0.0
    early_stop_patience: int = 6
    min_delta: float = 1e-4
    val_ratio: float = 0.15
    min_val_size: int = 8
    seed: int = 42
    # BC 内部验证/开发季（默认 2016）：仅早停与调参，不是独立测试季。
    # 真正留出评价季是 HITL_HOLDOUT_YEARS（2017–2020），勿写入此列表。
    # 空列表时 bc_finetune 会对 ≥2 季回退为「最后一季」，单季则行级拆分。
    val_years: list[str] = field(default_factory=lambda: ["2016"])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EXEC_DEFAULT = ExecConfig()
REVIEW_DEFAULT = ReviewConfig()
BC_DEFAULT = BCConfig()


def describe_bc_rationale() -> str:
    """静态说明；看板会在其后追加本会话已确认年 / 改动年（见 ``_session_status``）。"""
    return (
        "行为克隆(BC)：用专家确认动作做监督微调（交叉熵 + 相对原策略 KL），"
        f"不调用 PPO.update。"
        f" lr={BC_DEFAULT.lr}; kl_beta={BC_DEFAULT.kl_beta}; "
        f"policy_head_only={BC_DEFAULT.train_policy_head_only}; "
        f"W_edit={BC_DEFAULT.w_edit}; patience={BC_DEFAULT.early_stop_patience}。"
        f" 规范审阅年={','.join(HITL_REVIEW_YEARS)}（须逐年确认并形成改动后再 BC）；"
        f"留出评价季={','.join(HITL_HOLDOUT_YEARS)}（不进审阅/标签/BC 早停，仅最终泛化对照）；"
        f"BC 内部验证/开发季={','.join(BC_DEFAULT.val_years) or '行级拆分'}"
        f"（仅早停，非独立测试季）。"
    )


def resolve_year_split(available: list[str] | None = None) -> dict[str, list[str]]:
    """按可用年份裁剪默认审阅/留出划分。"""
    avail = [str(y) for y in (available or [])]
    if not avail:
        return {
            "review_years": list(HITL_REVIEW_YEARS),
            "holdout_years": list(HITL_HOLDOUT_YEARS),
            "all_eval_years": list(HITL_REVIEW_YEARS) + list(HITL_HOLDOUT_YEARS),
        }
    review = [y for y in HITL_REVIEW_YEARS if y in avail]
    holdout = [y for y in HITL_HOLDOUT_YEARS if y in avail]
    # 若配置年与当前评估集无交集，则按时间对半分
    if not review and not holdout:
        mid = max(1, len(avail) // 2)
        review, holdout = avail[:mid], avail[mid:]
    elif not review:
        review = [y for y in avail if y not in holdout] or avail[: max(1, len(avail) // 2)]
    elif not holdout:
        holdout = [y for y in avail if y not in review]
    return {
        "review_years": review,
        "holdout_years": holdout,
        "all_eval_years": avail,
    }
