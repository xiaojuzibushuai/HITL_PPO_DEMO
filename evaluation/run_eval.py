"""训后评估：PPO vs 固定规则（第一期）。

推荐::

    python run.py eval --model output/runs/run_<ts>/01_train/ppo_rice_best.pth

兼容：``python -m evaluation.run_eval --model ... --out-dir ...``  
评估表一律来自 scenario.forecast_eval / actual_eval（开跑前 validate）。  
正式产物应落在 ``output/runs/run_*/02_eval/``（经 ``api.evaluate`` / ``run.py eval``）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data_prep import prepare_eval_workdir  # noqa: E402
from common.model_io import load_model_config, load_ppo_checkpoint  # noqa: E402
from common.paths import mkdir  # noqa: E402
from common.scenario import set_active_scenario  # noqa: E402
from common.execution import apply_action, EXEC_PROTOCOL_VERSION  # noqa: E402
from common.action_mask import ACTION_SPACE, valid_action_mask  # noqa: E402
from decision.infer import apply_decision_pipeline, decide_from_logits, fixed_rule_action  # noqa: E402
from evaluation.metrics import compare_summary, compute_yearly_metrics  # noqa: E402
from train.PPO_RL_brain import PPO  # noqa: E402
import train.PPO_Rice_env as PPO_Rice_env  # noqa: E402


def _greedy_ppo_mm(agent: PPO, state_vec: np.ndarray, state_row, cfg: dict) -> tuple[float, float]:
    """返回 (cmd_mm, exec_mm)：掩码 argmax + 评估默认后处理。"""
    x = torch.FloatTensor(state_vec).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        logits, _ = agent.policy_net(x)
    eval_opts = (cfg.get("evaluation_defaults") if isinstance(cfg, dict) else None) or {}
    post = decide_from_logits(
        logits,
        x,
        action_space=agent.action_space,
        use_action_mask=bool(agent.use_action_mask),
        hmin=float(state_row.hmin),
        hmax=float(state_row.hmax),
        hp=float(state_row.Hp),
        hc=float(state_row.Hc),
        p_cumu7=float(getattr(state_row, "P_cumu7", 0.0) or 0.0),
        hc_guard=bool(eval_opts.get("hc_guard", True)),
        physical_clip=bool(eval_opts.get("physical_clip", True)),
        drying_rule=bool(eval_opts.get("drying_rule", False)),
        low_water_refill=bool(eval_opts.get("low_water_refill", False)),
    )
    return float(post["raw_action"]), float(post["action"])


def _rule_mm(state_row, cfg: dict) -> tuple[float, float]:
    """固定规则当日动作 → 与 PPO 相同后处理；返回 (raw_mm, exec_mm)。"""
    raw = fixed_rule_action(
        float(state_row.h_before),
        float(state_row.hmin),
        float(state_row.hmax),
        float(state_row.Hp),
        float(state_row.Hc),
    )
    # 请求取能达到同一阈值目标的离散档，执行器再限幅；与 PPO 共用21档接口。
    raw = float(np.sign(raw) * min(100.0, np.ceil(max(abs(raw)-1e-7, 0.0) / 10.0) * 10.0))
    mask_state = torch.tensor([[float(state_row[c]) for c in ("h_before", "hmin", "hmax", "Hp", "Hc")]], dtype=torch.float32)
    legal = [a for a, ok in zip(ACTION_SPACE, valid_action_mask(mask_state)[0].tolist()) if ok]
    if raw not in legal:
        raw = float(min(legal, key=lambda a: abs(a-raw)))
    eval_opts = (cfg.get("evaluation_defaults") if isinstance(cfg, dict) else None) or {}
    post = apply_decision_pipeline(
        raw,
        h=float(state_row.h_before),
        hmin=float(state_row.hmin),
        hmax=float(state_row.hmax),
        hp=float(state_row.Hp),
        hc=float(state_row.Hc),
        hc_guard=bool(eval_opts.get("hc_guard", True)),
        physical_clip=bool(eval_opts.get("physical_clip", True)),
        drying_rule=bool(eval_opts.get("drying_rule", False)),
        low_water_refill=False,
    )
    return float(raw), float(post["action"])


def _rollout(
    *,
    policy: str,
    agent: Optional[PPO],
    cfg: dict,
    max_years: Optional[int] = None,
) -> pd.DataFrame:
    """整季闭环：每日选动作 → env.step → 记日志。policy='ppo'|'rule'。"""
    env = PPO_Rice_env.Irrigation()
    state_space = env.state_space_policy
    actual_state_space = env.actual_state_space_policy
    rows: list[dict[str, Any]] = []
    env.current_year_idx = 0
    n_years = int(state_space["date"].astype(str).str.split("/").str[0].nunique())
    n_run = n_years if max_years is None else min(int(max_years), n_years)

    for ep in range(n_run):
        state, actual_state_pd = env.reset_policy(state_space)
        done = False
        step = 0
        while not done and step < 200:
            h0 = float(state.h_before)
            s_vec = np.array(env.process_series(state), dtype=np.float32)
            if policy == "ppo":
                assert agent is not None
                a_cmd, a_mm = _greedy_ppo_mm(agent, s_vec, state, cfg)
            else:
                a_cmd, a_mm = _rule_mm(state, cfg)

            actual = actual_state_space.loc[actual_state_space["date"] == state.date].iloc[0]
            expected_h, a_mm, exec_info = apply_action(env, state, actual, a_cmd)

            next_state, reward, done, a_exec = env.step(
                actual_state_pd, actual_state_space, state_space, state, a_mm
            )
            a_exec = float(a_exec)
            m = max(0.0, a_exec)
            d = max(0.0, -a_exec)
            r_total = float(reward[0]) if isinstance(reward, (tuple, list)) else float(reward)
            rows.append(
                {
                    "episode": ep,
                    "step": step,
                    "date": state.date,
                    "h_before": h0,
                    "hmin": float(state.hmin),
                    "hmax": float(state.hmax),
                    "Hp": float(state.Hp),
                    "Hc": float(state.Hc),
                    "lamda": float(state.lamda),
                    "ks": float(state.ks),
                    "P_cumu7": float(getattr(state, "P_cumu7", 0.0) or 0.0),
                    "a_policy_proposal": float(a_cmd),
                    "a_cmd": float(exec_info["requested_action"]),
                    "a_mm": float(a_mm),
                    "a_exec": a_exec,
                    "m": m,
                    "d": d,
                    "reward_total": r_total,
                    "h_after": float(next_state.h_before),
                    "used_fixed_rule": exec_info["used_fixed_rule"],
                    "exec_protocol": EXEC_PROTOCOL_VERSION,
                }
            )
            state = next_state
            step += 1

    df = pd.DataFrame(rows)
    out = df.copy()
    out["date_dt"] = pd.to_datetime(out["date"], errors="coerce")
    actual = actual_state_space.copy()
    cols = [c for c in ["date", "P_1_a", "et0_a"] if c in actual.columns]
    actual = actual.loc[actual["date"].notna(), cols].copy()
    actual["date_dt"] = pd.to_datetime(actual["date"], errors="coerce")
    actual = actual.loc[actual["date_dt"].notna()].drop_duplicates(subset=["date_dt"], keep="first")
    merged = out.merge(actual[["date_dt", "P_1_a", "et0_a"]], on="date_dt", how="left")
    merged["P_1_a"] = pd.to_numeric(merged.get("P_1_a", 0.0), errors="coerce").fillna(0.0)
    merged["et0_a"] = pd.to_numeric(merged.get("et0_a", 0.0), errors="coerce").fillna(0.0)
    return merged.drop(columns=["date_dt"])


def _write_daily_disagreement(out_dir: Path, ppo_csv: Path, rule_csv: Path) -> tuple[Path, dict[str, Any]]:
    """合并同日 PPO/规则动作，供二期 HITL 标「分歧日」。"""
    ppo = pd.read_csv(ppo_csv)
    rule = pd.read_csv(rule_csv)
    keys = ["date", "h_before", "hmin", "hmax", "Hp", "Hc", "P_1_a", "et0_a"]
    left = ppo[[c for c in keys + ["a_mm", "a_exec", "h_after", "reward_total"] if c in ppo.columns]].rename(
        columns={"a_mm": "a_ppo", "a_exec": "a_exec_ppo", "h_after": "h_after_ppo", "reward_total": "reward_ppo"}
    )
    right = rule[[c for c in ["date", "a_mm", "a_exec", "h_after"] if c in rule.columns]].rename(
        columns={"a_mm": "a_rule", "a_exec": "a_exec_rule", "h_after": "h_after_rule"}
    )
    m = left.merge(right, on="date", how="inner")
    # 分歧 = PPO 执行量与规则动作不相等（浮点容差 1e-6；优先 a_exec_ppo）
    exec_col = "a_exec_ppo" if "a_exec_ppo" in m.columns else "a_ppo"
    m["abs_diff_mm"] = (m[exec_col] - m["a_rule"]).abs()
    m["disagree"] = (m["abs_diff_mm"] > 1e-6).astype(int)
    out_dir = Path(mkdir(out_dir))
    path = out_dir / "policy_compare_daily.csv"
    m.to_csv(path, index=False, encoding="utf-8-sig")
    stats = {
        "n_days": int(len(m)),
        "n_disagree": int(m["disagree"].sum()),
        "disagree_rate": float(m["disagree"].mean()) if len(m) else 0.0,
        "mean_abs_diff_mm": float(m["abs_diff_mm"].mean()) if len(m) else 0.0,
    }
    return path, stats


def run_eval(
    model_path: str | Path | None,
    out_dir: str | Path,
    *,
    only: str = "both",
    max_years: Optional[int] = None,
    scenario: str | Path | None = None,
    keep_active_scenario: bool = False,
) -> dict[str, Any]:
    """多年闭环评估：表路径只认 scenario.forecast_eval / actual_eval。"""
    from common.scenario import get_scenario

    sc = get_scenario() if keep_active_scenario else set_active_scenario(scenario)
    out_dir = Path(mkdir(out_dir))
    if model_path is not None:
        model_path = Path(model_path).resolve()

    work = Path(mkdir(out_dir / "_workdir"))
    prepare_eval_workdir(work, scenario=sc, validate=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_cfg = load_model_config()
    prev = Path.cwd()
    os.chdir(work)
    result: dict[str, Any] = {
        "out_dir": str(out_dir),
        "eval_tables": {
            "forecast": sc.forecast_eval_rel,
            "actual": sc.actual_eval_rel,
        },
    }
    yearly_ppo = yearly_rule = None
    daily_ppo_path = daily_rule_path = None
    try:
        if only in ("ppo", "both"):
            if model_path is None:
                raise ValueError("only=ppo/both 时必须提供 model_path")
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            agent, cfg, _ = load_ppo_checkpoint(model_path, device=device)
            if "evaluation_defaults" not in cfg:
                cfg["evaluation_defaults"] = global_cfg.get("evaluation_defaults", {})
            daily = _rollout(policy="ppo", agent=agent, cfg=cfg, max_years=max_years)
            yearly_ppo = compute_yearly_metrics(daily)
            daily_ppo_path = out_dir / "policy_output_PPO.csv"
            daily.to_csv(daily_ppo_path, index=False, encoding="utf-8-sig")
            yearly_ppo.to_csv(out_dir / "yearly_metrics_PPO.csv", index=False, encoding="utf-8-sig")
            result["ppo_daily"] = str(daily_ppo_path)
            result["ppo_yearly"] = str(out_dir / "yearly_metrics_PPO.csv")
            result["policy_mode"] = cfg.get("policy_mode", "masked")
            result["n_years_run"] = int(yearly_ppo["year"].nunique()) if len(yearly_ppo) else 0

        if only in ("rule", "both"):
            cfg = {"evaluation_defaults": global_cfg.get("evaluation_defaults", {})}
            daily = _rollout(policy="rule", agent=None, cfg=cfg, max_years=max_years)
            yearly_rule = compute_yearly_metrics(daily)
            daily_rule_path = out_dir / "policy_output_RULE.csv"
            daily.to_csv(daily_rule_path, index=False, encoding="utf-8-sig")
            yearly_rule.to_csv(out_dir / "yearly_metrics_RULE.csv", index=False, encoding="utf-8-sig")
            result["rule_daily"] = str(daily_rule_path)
            result["rule_yearly"] = str(out_dir / "yearly_metrics_RULE.csv")
            if "n_years_run" not in result:
                result["n_years_run"] = int(yearly_rule["year"].nunique()) if len(yearly_rule) else 0
    finally:
        os.chdir(prev)

    if yearly_ppo is not None and yearly_rule is not None:
        summary = compare_summary(yearly_ppo, yearly_rule)
        (out_dir / "compare_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result["summary"] = summary
        result["compare_summary"] = str(out_dir / "compare_summary.json")
        if daily_ppo_path and daily_rule_path:
            cmp_path, disagree = _write_daily_disagreement(out_dir, daily_ppo_path, daily_rule_path)
            result["daily_compare"] = str(cmp_path)
            result["disagree"] = disagree
            summary["disagree"] = disagree
            (out_dir / "compare_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    art_keys = (
        "ppo_daily", "ppo_yearly", "rule_daily", "rule_yearly",
        "compare_summary", "daily_compare", "policy_mode", "disagree",
    )
    manifest = {
        "out_dir": str(out_dir),
        "eval_tables": result["eval_tables"],
        "only": only,
        "max_years": max_years,
        "n_years_run": result.get("n_years_run"),
        "artifacts": {k: result[k] for k in art_keys if k in result},
    }
    man_path = out_dir / "run_manifest.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    result["manifest"] = str(man_path)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="PPO_Rice_Env 训后评估：PPO vs 固定规则")
    p.add_argument("--model", default=None, help="PPO .pth；only=rule 时可省略")
    p.add_argument(
        "--out-dir",
        default=None,
        help="输出目录；省略则写入新建 runs/run_*/02_eval（与 api.evaluate 一致）",
    )
    p.add_argument("--only", choices=["ppo", "rule", "both"], default="both")
    p.add_argument("--max-years", type=int, default=None, help="最多评估年数；默认=全部验证年")
    args = p.parse_args()
    if args.only != "rule" and not args.model:
        p.error("only=ppo/both 时必须 --model")
    out_dir = args.out_dir
    if out_dir is None:
        from common.run_store import RunPaths, create_run

        out_dir = str(RunPaths(create_run(note="run_eval.cli")).ensure().eval)
    info = run_eval(args.model, out_dir, only=args.only, max_years=args.max_years)
    print(
        json.dumps(
            info.get("summary", {k: v for k, v in info.items() if k != "summary"}),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
