"""v3 定向回归检查；只写入独立验收会话，不覆盖旧案例。"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common.paths import ROOT
from common.data_prep import prepare_eval_workdir
from common.model_io import load_ppo_checkpoint
from common.execution import apply_action, apply_expert_action, norm_date, allowed_actions
from evaluation.run_eval import _rollout
from .rollout import rollout_all_years
from .recompute import recompute_consistent_trajectory, verify_review_vs_labels, validate_review_table
from .resync import resync_review_with_policy
from .session import create_session, SessionPaths
from .pipeline import assess_bc_readiness, finalize_review, bc
from .config import BCConfig
import train.PPO_Rice_env as env_module


def run_checks():
    torch.set_num_threads(2)
    model = ROOT / "output/runs/run_20261024_170858/01_train/ppo_rice_best.pth"
    sp = SessionPaths(create_session("sess_v3_acceptance_20261025", note="独立回归验收，不作为效果结果")).ensure()
    prepare_eval_workdir(sp.workdir)
    initial_cwd = Path.cwd()
    checks = {}
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    try:
        os.chdir(sp.workdir)
        agent, cfg, _ = load_ppo_checkpoint(model)
        env = env_module.Irrigation()
        review = rollout_all_years(agent, env, years=["2012"])
        review["confirmed"] = True
        labels = recompute_consistent_trajectory(review, env)
        checks["fresh_review_parity"] = verify_review_vs_labels(review, labels)
        assert checks["fresh_review_parity"]["ok"]
        daily = _rollout(policy="ppo", agent=agent, cfg=cfg, max_years=1)
        checks["review_evaluation_max_h_error"] = float(np.max(np.abs(daily.h_before.values-review.h_before.values)))
        checks["review_evaluation_max_exec_error"] = float(np.max(np.abs(daily.a_exec.values-review.a_exec.values)))
        assert checks["review_evaluation_max_h_error"] < 1e-5
        assert checks["review_evaluation_max_exec_error"] < 1e-5
        rejected = {}
        for kind in ("missing_day", "missing_vector", "bad_vector", "invalid_request"):
            damaged = review.copy(deep=True)
            if kind == "missing_day":
                damaged = damaged.drop(damaged.index[7])
            elif kind == "missing_vector":
                damaged = damaged.drop(columns=["state_vec"])
            elif kind == "bad_vector":
                damaged.at[0, "state_vec"] = [1, 2]
            else:
                damaged.at[0, "a_expert"] = 999
            try:
                recompute_consistent_trajectory(damaged, env)
                rejected[kind] = False
            except (ValueError, TypeError):
                rejected[kind] = True
        assert all(rejected.values()), rejected
        checks["invalid_inputs_rejected"] = rejected
        state, _ = env.reset_policy(env.state_space_policy)
        state.h_before = -5.0
        actual = env.actual_state_space_policy.iloc[0].copy()
        actual.P_1_a, actual.et0_a = 0.0, 0.0
        checks["dry_no_percolation_h_after"] = float(env.correct(state, actual, 0)[0])
        assert checks["dry_no_percolation_h_after"] == -5
        same = []
        for action in allowed_actions(state):
            h1, a1, _ = apply_action(env, state, actual, action, fixed_rule_stage=False)
            h2, a2, _ = apply_expert_action(env, state, actual, action)
            same.append(abs(h1-h2) < 1e-8 and abs(a1-a2) < 1e-8)
        assert all(same)
        checks["same_legal_request_same_execution"] = True
        # 只改一天：此前保留，之后失效；重确认后导出必须与新轨迹一致。
        anchor = next(i for i, r in review.iterrows() if r.editable and len(allowed_actions(pd.Series({**r.to_dict(), "date": norm_date(r.date)}))) > 1 and i > 3)
        changed = review.copy(deep=True)
        allowed = allowed_actions(pd.Series(changed.loc[anchor].to_dict()))
        action = next(a for a in allowed if a != changed.at[anchor, "a_model"])
        changed.at[anchor, "a_expert"] = action
        resynced = resync_review_with_policy(changed, agent, env, override_indices={anchor}, fresh_indices={anchor})
        assert np.array_equal(resynced.h_before.values[:anchor], review.h_before.values[:anchor])
        assert bool(resynced.loc[(resynced.index > anchor) & resynced.editable, "needs_reconfirm"].all())
        resynced["confirmed"], resynced["needs_reconfirm"] = True, False
        assert verify_review_vs_labels(resynced, recompute_consistent_trajectory(resynced, env))["ok"]
        checks["resync_and_reconfirmation"] = True
    finally:
        os.chdir(initial_cwd)
    meta0 = json.loads((sp.root / "meta.json").read_text(encoding="utf-8"))
    gate0 = assess_bc_readiness(
        review, meta0, workdir=sp.workdir, require_edits=False, check_parity=False,
    )
    assert not gate0["ok"] and gate0["fingerprint_bad_years"], gate0
    checks["assess_bc_readiness_blocks_without_finalize"] = True
    # 缺指纹拦截，定稿后短程 BC、重载、恢复 cwd。
    try:
        bc(review, work=sp.root, model_path=model, epochs=1, require_edits=False)
        raise AssertionError("未定稿 BC 被错误放行")
    except ValueError:
        checks["missing_fingerprint_rejected"] = True
    finalized = finalize_review(sp.root, review, source="acceptance_only")
    meta1 = json.loads((sp.root / "meta.json").read_text(encoding="utf-8"))
    gate1 = assess_bc_readiness(
        finalized, meta1, workdir=sp.workdir, require_edits=False, check_parity=True,
    )
    assert gate1["ok"] and gate1["bc_ready"], gate1
    checks["assess_bc_readiness_ok_after_finalize"] = True
    # 验收会话可无专家改动；正式实验仍默认 require_edits=True（与看板一致）
    out = bc(
        work=sp.root,
        model_path=model,
        bc_cfg=BCConfig(epochs=2),
        require_edits=False,
    )
    assert Path.cwd() == initial_cwd
    loaded, loaded_cfg, _ = load_ppo_checkpoint(out["hitl_model"])
    assert loaded_cfg["hidden_dim"] == cfg["hidden_dim"]
    checks["bc_reload_cwd_and_metadata"] = True
    mutated = finalized.copy(deep=True)
    mutated.at[0, "h_before"] += 1
    try:
        bc(mutated, work=sp.root, model_path=model, epochs=1, require_edits=False)
        raise AssertionError("过期指纹 BC 被错误放行")
    except ValueError:
        checks["stale_fingerprint_rejected"] = True
    assert hashlib.sha256(model.read_bytes()).hexdigest() == model_hash
    checks["original_checkpoint_unchanged"] = True
    checks["ok"] = True
    path = sp.metrics / "acceptance.json"
    path.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"acceptance": str(path), "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_checks()
