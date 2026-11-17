"""审阅期：专家改动作后，从该日执行起重推到季末，并用冻结 PPO 刷新后续建议。

规则（对齐终稿）
----------------
- 本轮**改动作**的最早一天记为 T：**T 之前整行原样**（含晒田/固定规则日，不从季初重放）。
- **T 日**：晨水深不动；专家**请求**经与 PPO 相同执行链（Hc 守卫 + 物理限幅 + ``env.correct``）得 ``a_exec``/日末水。
- **T 之后**：旧专家改动作废，按新水深链 PPO 预演并标需重看。
- **经验规则 a_rule**、**实测雨 P_1_a** 不随 resync 改写（对照基线固定）。
- 仅「确认」、未改动作（无 override）→ 不作废后续。
- 同一请求里若同时修改 T 与其后的日期，只应用最早修改；T 后动作/确认均按新状态作废并重新标为待复核。
- 季内按真实日历序（勿对无前导零 Y/M/D 字符串排序）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch

import train.PPO_Rice_env as PPO_Rice_env
from common.scenario import ks_factor
from decision.infer import decide_from_logits

from .config import ACTION_SPACE, EXEC_DEFAULT, ExecConfig, snap_to_action_space
from .exec_layer import (
    apply_action,
    apply_expert_action,
    date_ymd,
    expert_action_filled,
    fixed_rule_action_mm,
    is_fixed_rule_stage,
    norm_date,
    stage_name_from_state,
    allowed_actions, EXEC_PROTOCOL_VERSION,
)


def _suggest(agent, s_vec: np.ndarray, state, *, use_guard: bool) -> tuple[float, int, float]:
    action_space = list(getattr(agent, "action_space", None) or ACTION_SPACE)
    use_mask = bool(getattr(agent, "use_action_mask", True))
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
        p_cumu7=float(getattr(state, "P_cumu7", 0.0) or 0.0),
        hc_guard=use_guard,
        physical_clip=True,
        drying_rule=False,
        low_water_refill=False,
    )
    return float(dec["raw_action"]), int(dec["action_idx"]), float(dec["action"])


def resync_review_with_policy(
    review_df: pd.DataFrame,
    agent,
    env: Optional[PPO_Rice_env.Irrigation] = None,
    *,
    cfg: Optional[ExecConfig] = None,
    fresh_indices: Optional[set[int]] = None,
    override_indices: Optional[set[int]] = None,
) -> pd.DataFrame:
    """从季初按日推进。

    - ``override_indices``：本轮**改了专家动作**的行（决定作废起点 T）。
    - ``fresh_indices``：本轮确认/改动行。早于最早动作修改的确认保留；受上游修改影响的后续输入需重看。
    """
    if review_df.empty:
        return review_df.copy()

    cfg = cfg or EXEC_DEFAULT
    env = env or PPO_Rice_env.Irrigation()
    df = review_df.copy().reset_index(drop=True)
    fresh = {int(i) for i in (fresh_indices or set())}
    overrides = {int(i) for i in (override_indices or set())}
    if "confirmed" not in df.columns:
        df["confirmed"] = False
    if "year" not in df.columns:
        df["year"] = df["date"].map(lambda d: norm_date(d).split("/")[0])
    df["_d"] = df["date"].map(norm_date)
    df["_ord"] = df["_d"].map(date_ymd)

    state_space = env.state_space_policy.copy()
    state_space["_d"] = state_space["date"].map(norm_date)
    actual_space = env.actual_state_space_policy.copy()
    actual_space["_d"] = actual_space["date"].map(norm_date)

    out_rows: list[dict] = []
    for _, season in df.groupby("year", sort=True):
        season = season.sort_values("_ord", kind="mergesort")
        records = list(season.iterrows())
        if not records:
            continue

        # 本季本轮改动作的最早日历点 T；本季无 override → 整季原样保留（勿从季初重放）
        ov_ords = [tuple(row["_ord"]) for row_i, row in records if int(row_i) in overrides]
        t0: tuple[int, int, int] | None = min(ov_ords) if ov_ords else None
        if t0 is None:
            for _row_i, row in records:
                new_row = {k: row[k] for k in row.index if k not in ("_d", "_ord")}
                out_rows.append(new_row)
            continue

        h_carry: float | None = None

        for row_i, row in records:
            date = row["_d"]
            ord_t = tuple(row["_ord"])

            # T 之前：审阅表整行不动（含固定规则日；避免从季初重放改掉 T 晨水深）
            if ord_t < t0:
                new_row = {k: row[k] for k in row.index if k not in ("_d", "_ord")}
                out_rows.append(new_row)
                continue

            base_m = state_space[state_space["_d"] == date]
            if base_m.empty:
                raise ValueError(f"重算缺环境日: {date}")
            if ord_t == t0:
                h_morning = float(row["h_before"])
            elif h_carry is not None:
                h_morning = float(h_carry)
            else:
                h_morning = float(row["h_before"])

            state = base_m.iloc[0].copy()
            state["h_before"] = float(h_morning)
            state["ks"] = ks_factor(float(h_morning))
            st = state.drop(labels=["_d"], errors="ignore")

            stage = stage_name_from_state(st)
            fixed = is_fixed_rule_stage(st, stage_name=stage, cfg=cfg)
            editable = (
                bool(row["editable"]) if "editable" in row and pd.notna(row["editable"]) else not fixed
            ) and not fixed

            prev_expert = row.get("a_expert", np.nan)
            was_confirmed = bool(row.get("confirmed", False))
            a_expert_hist = row.get("a_expert_hist", np.nan)
            is_override = int(row_i) in overrides
            is_fresh = int(row_i) in fresh or is_override
            anchor_t = ord_t == t0

            store_expert: float | None = None
            drive_action = 0.0
            if fixed:
                a_model = float(fixed_rule_action_mm(st))
                a_idx = int(np.argmin([abs(a - a_model) for a in ACTION_SPACE]))
                a_applied = a_model
                a_expert = a_model
                drive_action = float(a_expert)
                store_expert = float(a_expert)
                confirmed = True
                source = "fixed_rule"
                needs = False
            else:
                s_vec = np.array(env.process_series(st), dtype=np.float32)
                a_model, a_idx, a_applied = _suggest(
                    agent, s_vec, st, use_guard=bool(cfg.force_hc_guard),
                )
                filled = expert_action_filled(prev_expert)
                expert_mm: float | None = (
                    float(prev_expert) if filled else None
                )
                drive_action = float(a_model)
                store_expert = expert_mm
                source = "ppo_refresh"
                needs = False
                confirmed = bool(was_confirmed)

                # 最早动作修改后状态已改变；同批次旧提案/旧专家请求均不可沿用。
                after_t = ord_t > t0
                if after_t and editable:
                    hist_src = (
                        expert_mm
                        if expert_mm is not None
                        else row.get("a_expert_hist", np.nan)
                    )
                    if pd.notna(hist_src):
                        a_expert_hist = float(snap_to_action_space(hist_src))
                    drive_action = float(a_model)
                    store_expert = None
                    confirmed = False
                    source = "ppo_refresh"
                    needs = True
                elif editable and expert_mm is not None and (
                    anchor_t or is_override or is_fresh
                ):
                    drive_action = expert_mm
                    store_expert = expert_mm
                    source = "expert" if (
                        is_override
                        or abs(expert_mm - float(a_model)) > 1e-6
                        or was_confirmed
                    ) else "follow"
                    needs = False
                    # 显式选择了该日专家动作（包括与 PPO 同档）即完成该日审阅。
                    confirmed = bool(was_confirmed or is_override)
                elif expert_mm is not None:
                    drive_action = expert_mm
                    store_expert = expert_mm
                    source = (
                        "expert" if abs(expert_mm - float(a_model)) > 1e-6 else "follow"
                    )
                else:
                    drive_action = float(a_model)
                    store_expert = None
                    source = "ppo_refresh"
                    needs = False
                    confirmed = False

            if fixed:
                s_vec = np.array(env.process_series(st), dtype=np.float32)

            actual_m = actual_space[actual_space["_d"] == date]
            if actual_m.empty:
                raise ValueError(f"重算缺实测日: {date}")
            actual = actual_m.iloc[0].drop(labels=["_d"], errors="ignore")

            state["h_before"] = float(h_morning)
            state["ks"] = ks_factor(float(h_morning))
            st = state.drop(labels=["_d"], errors="ignore")
            # —— 日末水深 / 执行量（验收：同状态+同请求 → 同 a_exec / h_after）——
            # ① 固定规则日：apply_action(fixed=True)
            # ② 专家/跟表：apply_expert_action ≡ apply_action(fixed=False)（Hc+物理限幅+correct）
            # ③ 模型日 ppo_refresh：同上 apply_action(fixed=False)
            if fixed:
                h_after, a_exec, info = apply_action(
                    env, st, actual, drive_action, fixed_rule_stage=True, cfg=cfg,
                )
            elif source in ("expert", "follow"):
                h_after, a_exec, info = apply_expert_action(
                    env, st, actual, drive_action, cfg=cfg,
                )
            else:
                h_after, a_exec, info = apply_action(
                    env, st, actual, drive_action, fixed_rule_stage=False, cfg=cfg,
                )

            # T 日：晨水深与模型提案列保持改前表内值；执行/日末水深按专家重算
            if anchor_t and not fixed:
                h_morning = float(row["h_before"])
                if pd.notna(row.get("a_model", np.nan)):
                    a_model = float(row["a_model"])
                if pd.notna(row.get("a_applied", np.nan)):
                    a_applied = float(row["a_applied"])
                if pd.notna(row.get("a_idx", np.nan)):
                    a_idx = int(row["a_idx"])

            new_row = {k: row[k] for k in row.index if k not in ("_d", "_ord")}
            new_row.update({
                "date": date,
                "year": str(date).split("/")[0],
                "stage": stage,
                "h_before": float(h_morning),
                "ks": float(st.ks),
                "a_model": float(a_model),
                "a_idx": int(a_idx),
                "a_applied": float(a_applied),
                "a_exec": float(a_exec),
                "a_expert": store_expert,
                "a_expert_hist": a_expert_hist if pd.notna(a_expert_hist) else None,
                "editable": bool(editable),
                "confirmed": bool(confirmed),
                "source": source,
                "needs_reconfirm": bool(needs),
                "used_fixed_rule": bool(info["used_fixed_rule"]),
                "used_hc_guard": bool(info["used_hc_guard"]),
                "state_vec": s_vec.tolist(),
                "h_after": float(h_after),
                "allowed_actions": allowed_actions(st),
                "exec_protocol": EXEC_PROTOCOL_VERSION,
            })
            out_rows.append(new_row)
            h_carry = float(h_after)

    out = pd.DataFrame(out_rows)
    if len(out):
        from .rollout import _mark_small_runs

        irrig = [
            max(0.0, float(a)) if e else 0.0
            for a, e in zip(out["a_exec"].tolist(), out["editable"].tolist())
        ]
        out["is_small_run"] = _mark_small_runs(
            irrig,
            dates=out["date"].tolist(),
            editable=out["editable"].tolist(),
            used_fixed_rule=out["used_fixed_rule"].tolist()
            if "used_fixed_rule" in out.columns
            else None,
        )
    return out
