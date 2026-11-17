"""Real-human HITL follow-up: action-probability diagnosis + PPO-label BC control.

This analysis never edits the source HITL session. It trains one control policy
from a fresh copy of the frozen PPO checkpoint, replacing expert-edited targets
with the PPO proposal recorded for the same reviewed state while keeping the
same samples, sample weights, BC configuration, validation year, and seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.action_mask import ACTION_SPACE, mask_policy_logits, valid_action_mask
from common.model_io import load_ppo_checkpoint, sidecar_config_path
from hitl.bc_finetune import finetune_bc
from hitl.config import BCConfig, action_to_idx


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _year_group(year: Any) -> str:
    y = str(year)
    if y in {"2012", "2013", "2014"}:
        return "train_review_2012_2014"
    if y == "2016":
        return "internal_validation_2016"
    if y in {"2017", "2018", "2019", "2020"}:
        return "holdout_2017_2020"
    return "other"


def _probability_rows(
    labels: pd.DataFrame,
    policies: dict[str, tuple[Any, list[float]]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare masked expert-label probabilities/ranks and greedy actions."""
    include = labels.get("include_in_bc", pd.Series(False, index=labels.index)).fillna(False).astype(bool)
    edited = labels.get("flag", pd.Series("", index=labels.index)).astype(str).eq("edited")
    rows = labels.loc[include & edited].copy()
    if rows.empty:
        raise ValueError("No expert-edited BC rows found in labels.json")

    records: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        state = np.asarray(row["s_vec"], dtype=np.float32).reshape(-1)
        if not np.isfinite(state).all():
            raise ValueError(f"Non-finite state at {row.get('date')}")
        target_idx = int(row["a_idx"])
        if target_idx < 0 or target_idx >= len(ACTION_SPACE):
            raise ValueError(f"Invalid expert action index {target_idx} at {row.get('date')}")
        target_mm = float(ACTION_SPACE[target_idx])
        target_label_mm = float(row["a_label_mm"])
        if abs(target_mm - target_label_mm) > 1e-6:
            raise ValueError(f"a_idx/a_label_mm mismatch at {row.get('date')}")

        rec: dict[str, Any] = {
            "date": str(row.get("date", "")),
            "year": str(row.get("year", "")),
            "split": _year_group(row.get("year", "")),
            "stage": str(row.get("stage", "")),
            "state": state.tolist(),
            "expert_action_idx": target_idx,
            "expert_action_mm": target_mm,
            "ppo_proposal_mm": float(row["a_model"]),
            "valid_action_count": None,
        }
        for policy_name, (agent, action_space) in policies.items():
            device = agent.device
            x = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.inference_mode():
                logits, _ = agent.policy_net(x)
                masked = mask_policy_logits(logits, x, action_space)
                valid = valid_action_mask(x, action_space)[0]
                probs = F.softmax(masked, dim=-1)[0]
                n_valid = int(valid.sum().item())
                if not bool(valid[target_idx].item()):
                    raise ValueError(
                        f"Expert action is masked for {row.get('date')} under {policy_name}"
                    )
                target_prob = float(probs[target_idx].item())
                valid_probs = probs[valid]
                valid_indices = torch.arange(len(action_space), device=device)[valid]
                rank = int((valid_probs > target_prob + 1e-12).sum().item()) + 1
                top_idx = int(masked.argmax(dim=-1).item())
            rec[f"{policy_name}_expert_prob"] = target_prob
            rec[f"{policy_name}_expert_rank"] = rank
            rec[f"{policy_name}_top1_idx"] = top_idx
            rec[f"{policy_name}_top1_mm"] = float(action_space[top_idx])
            rec[f"{policy_name}_top1_matches_expert"] = bool(top_idx == target_idx)
            rec["valid_action_count"] = n_valid
        rec["choice_available"] = bool(rec["valid_action_count"] > 1)
        for policy_name in policies:
            if policy_name != "ppo":
                rec[f"{policy_name}_expert_prob_delta_vs_ppo"] = (
                    rec[f"{policy_name}_expert_prob"] - rec["ppo_expert_prob"]
                )
                rec[f"{policy_name}_rank_change_vs_ppo"] = (
                    rec[f"{policy_name}_expert_rank"] - rec["ppo_expert_rank"]
                )
        rec.pop("state")
        records.append(rec)

    frame = pd.DataFrame(records)
    summary: dict[str, Any] = {}
    for group_name, group_df in [("all", frame), *frame.groupby("split", sort=True)]:
        evaluable = group_df[group_df["choice_available"]]
        item: dict[str, Any] = {
            "n_edited": int(len(group_df)),
            "n_action_choice": int(len(evaluable)),
        }
        for policy_name in policies:
            prob_col = f"{policy_name}_expert_prob"
            rank_col = f"{policy_name}_expert_rank"
            top_col = f"{policy_name}_top1_matches_expert"
            item[policy_name] = {
                "mean_expert_action_probability": (
                    float(evaluable[prob_col].mean()) if len(evaluable) else None
                ),
                "median_expert_action_probability": (
                    float(evaluable[prob_col].median()) if len(evaluable) else None
                ),
                "mean_expert_action_rank": (
                    float(evaluable[rank_col].mean()) if len(evaluable) else None
                ),
                "top1_expert_matches": int(evaluable[top_col].sum()),
                "top1_expert_match_rate": (
                    float(evaluable[top_col].mean()) if len(evaluable) else None
                ),
            }
            if policy_name != "ppo":
                delta = evaluable[f"{policy_name}_expert_prob_delta_vs_ppo"]
                rank_delta = evaluable[f"{policy_name}_rank_change_vs_ppo"]
                item[policy_name]["mean_expert_probability_delta_vs_ppo"] = (
                    float(delta.mean()) if len(delta) else None
                )
                item[policy_name]["expert_probability_increased_share_vs_ppo"] = (
                    float((delta > 1e-12).mean()) if len(delta) else None
                )
                item[policy_name]["mean_expert_rank_change_vs_ppo"] = (
                    float(rank_delta.mean()) if len(rank_delta) else None
                )
                item[policy_name]["expert_rank_improved_share_vs_ppo"] = (
                    float((rank_delta < 0).mean()) if len(rank_delta) else None
                )
        summary[group_name] = item
    return frame, summary


def _grouped_metrics(session: Path, control_eval: Path, out_dir: Path) -> pd.DataFrame:
    sources = {
        "original_ppo": session / "06_compare" / "original_ppo" / "yearly_metrics_PPO.csv",
        "human_feedback_bc": session / "06_compare" / "hitl_ppo" / "yearly_metrics_PPO.csv",
        "ppo_label_bc_control": control_eval / "yearly_metrics_PPO.csv",
    }
    metrics = [
        "irrigation_total", "drainage_total", "managed_water", "rain_utilization",
        "yield_factor", "irrigation_times", "drainage_times", "stress_days",
        "stress_days_all", "overpond_days", "overpond_exposure",
    ]
    records: list[dict[str, Any]] = []
    expected_years: set[str] | None = None
    for arm, path in sources.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing yearly evaluation for {arm}: {path}")
        df = pd.read_csv(path)
        df["year"] = df["year"].astype(str)
        years = set(df["year"])
        if expected_years is None:
            expected_years = years
        elif years != expected_years:
            raise ValueError(f"Evaluation-year mismatch for {arm}: {sorted(years)}")
        df["split"] = df["year"].map(_year_group)
        for split, sub in [("all_8_years", df), *df.groupby("split", sort=True)]:
            rec: dict[str, Any] = {"arm": arm, "split": split, "n_years": int(sub["year"].nunique())}
            for metric in metrics:
                if metric in sub.columns:
                    rec[metric] = float(pd.to_numeric(sub[metric], errors="coerce").mean())
            records.append(rec)
    frame = pd.DataFrame(records)
    base = frame[frame["arm"] == "original_ppo"].set_index("split")
    for arm in ("human_feedback_bc", "ppo_label_bc_control"):
        mask = frame["arm"] == arm
        for metric in metrics:
            if metric not in frame.columns:
                continue
            frame.loc[mask, f"delta_{metric}_vs_ppo"] = frame.loc[mask].apply(
                lambda row: float(row[metric] - base.loc[row["split"], metric]), axis=1
            )
    out_path = out_dir / "grouped_year_metrics.csv"
    frame.to_csv(out_path, index=False, encoding="utf-8-sig")
    return frame


def run(session: Path, output_dir: Path | None = None) -> Path:
    session = session.resolve()
    if not session.is_dir():
        raise FileNotFoundError(session)
    meta_path = session / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    human_sources = meta.get("feedback_sources_by_year", {})
    review_years = set((meta.get("year_split") or {}).get("review_years", []))
    if not review_years or any(human_sources.get(y) != "human_whole_season_acceptance" for y in review_years):
        raise ValueError("Session is not fully recorded as genuine whole-season human review")
    if set(meta.get("finalized_years", [])) != review_years:
        raise ValueError("Review years are not all finalized")

    labels_path = session / "04_bc_export" / "labels.json"
    parity_path = session / "04_bc_export" / "review_labels_parity.json"
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    if not parity.get("ok") or int(parity.get("n_mismatch", -1)) != 0:
        raise ValueError("Review/export parity is not clean; refusing follow-up experiment")
    labels = pd.read_json(labels_path)
    base_model = session / "01_train" / "ppo_rice_best.pth"
    human_model = session / "05_bc" / "ppo_hitl.pth"
    human_sidecar = sidecar_config_path(human_model)
    if not all(path.exists() for path in (labels_path, base_model, human_model, human_sidecar)):
        raise FileNotFoundError("Required human-session labels or model files are missing")

    if output_dir is None:
        output_dir = ROOT / "output" / "paper_experiments" / f"human_feedback_followup_{session.name}"
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite follow-up results: {output_dir}")
    output_dir.mkdir(parents=True)

    bc_side = json.loads(human_sidecar.read_text(encoding="utf-8"))
    cfg_data = bc_side.get("bc") or {}
    cfg = BCConfig(**cfg_data)
    if list(map(str, cfg.val_years)) != ["2016"]:
        raise ValueError(f"Expected frozen internal validation year 2016, got {cfg.val_years}")

    ppo_agent, base_cfg, _ = load_ppo_checkpoint(base_model)
    human_agent, _, _ = load_ppo_checkpoint(human_model)
    policies: dict[str, tuple[Any, list[float]]] = {
        "ppo": (ppo_agent, list(ppo_agent.action_space)),
        "human_feedback_bc": (human_agent, list(human_agent.action_space)),
    }
    probability_frame, _ = _probability_rows(labels, policies)
    prob_path = output_dir / "action_probability_diagnostic.csv"
    probability_frame.to_csv(prob_path, index=False, encoding="utf-8-sig")

    # PPO-label BC control: same rows and edit weights, but no changed target is
    # allowed to carry the expert action. Retain human labels in a separate audit column.
    control = labels.copy()
    include = control.get("include_in_bc", pd.Series(False, index=control.index)).fillna(False).astype(bool)
    edited = control.get("flag", pd.Series("", index=control.index)).astype(str).eq("edited")
    edit_rows = control.index[include & edited]
    if not len(edit_rows):
        raise ValueError("No edited rows available for the PPO-label BC control")
    control["human_expert_label_mm"] = pd.to_numeric(control["a_label_mm"], errors="coerce")
    control["bc_control_target_mm"] = control["human_expert_label_mm"]
    for idx in edit_rows:
        proposal = float(control.at[idx, "a_model"])
        target_idx = action_to_idx(proposal)
        control.at[idx, "a_idx"] = target_idx
        control.at[idx, "bc_control_target_mm"] = float(ACTION_SPACE[target_idx])
    control["bc_control_type"] = "edited_expert_targets_reverted_to_same_state_ppo_proposals"
    target_audit = control.loc[edit_rows, [
        "date", "year", "human_expert_label_mm", "a_model", "bc_control_target_mm", "a_idx"
    ]].copy()
    target_audit.to_csv(output_dir / "ppo_label_control_targets.csv", index=False, encoding="utf-8-sig")

    control_agent, _, _ = load_ppo_checkpoint(base_model)
    control_model = output_dir / "ppo_label_bc_control.pth"
    history = finetune_bc(control_agent, control, cfg=cfg, save_path=control_model)
    control_sidecar = {
        **base_cfg,
        "kind": "bc_finetune",
        "experiment_arm": "ppo_label_bc_control",
        "control_definition": (
            "Same BC rows, sample weights, config, validation year and seed as human BC; "
            "each edited target is replaced with that row's PPO proposal."
        ),
        "exec_protocol": meta.get("exec_protocol"),
        "bc": cfg.to_dict(),
        "base_model": str(base_model),
        "base_model_sha256": _sha256(base_model),
        "base_model_cfg": base_cfg,
        "labels_sha256": _sha256(labels_path),
        "best_epoch": history.get("best_epoch"),
        "n_samples": history.get("n_samples"),
        "train_indices": history.get("train_indices"),
        "val_indices": history.get("val_indices"),
        "validation_years": history.get("validation_years"),
        "split_unit": history.get("split_unit"),
        "n_train": history.get("n_train"),
        "n_validation": history.get("n_validation"),
        "control_target_rows": int(len(edit_rows)),
    }
    sidecar_config_path(control_model).write_text(
        json.dumps(control_sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from api import evaluate

    control_eval = output_dir / "ppo_label_control_eval"
    control_eval.mkdir(parents=True)
    eval_result = evaluate(
        control_model,
        out_dir=control_eval,
        only="ppo",
        scenario=None,
    )

    control_agent_eval, _, _ = load_ppo_checkpoint(control_model)
    policies["ppo_label_bc_control"] = (control_agent_eval, list(control_agent_eval.action_space))
    probability_frame, probability_summary = _probability_rows(labels, policies)
    probability_frame.to_csv(prob_path, index=False, encoding="utf-8-sig")
    _write_json(output_dir / "action_probability_summary.json", probability_summary)

    grouped_metrics = _grouped_metrics(session, control_eval, output_dir)
    summary = {
        "experiment": "real_human_feedback_bc_followup",
        "session": str(session),
        "output_dir": str(output_dir),
        "feedback_source": meta.get("feedback_source"),
        "review_years": sorted(review_years),
        "training_feedback_years": ["2012", "2013", "2014"],
        "internal_validation_years": ["2016"],
        "holdout_years": ["2017", "2018", "2019", "2020"],
        "execution_protocol": meta.get("exec_protocol"),
        "parity": parity,
        "n_labels": int(len(labels)),
        "n_bc_rows": int(include.sum()),
        "n_expert_edits": int(len(edit_rows)),
        "base_model_sha256": _sha256(base_model),
        "human_bc_model_sha256": _sha256(human_model),
        "source_labels_sha256": _sha256(labels_path),
        "bc_config": cfg.to_dict(),
        "ppo_label_control": {
            "model": str(control_model),
            "model_sha256": _sha256(control_model),
            "history": history,
            "evaluation": eval_result,
            "definition": control_sidecar["control_definition"],
        },
        "action_probability_summary": str(output_dir / "action_probability_summary.json"),
        "action_probability_rows": str(prob_path),
        "grouped_year_metrics": str(output_dir / "grouped_year_metrics.csv"),
        "grouped_metric_records": grouped_metrics.to_dict(orient="records"),
    }
    _write_json(output_dir / "experiment_summary.json", summary)
    (output_dir / "README.md").write_text(
        "# 真人专家反馈 BC 后续对照\n\n"
        f"本目录分析 `{session.name}` 的真人专家审阅。\n"
        "`ppo_label_bc_control` 是无专家修改标签对照：保留同一批状态、BC 权重、配置、2016 验证年和随机种子；"
        "仅把专家改过的 BC 目标替换回同状态下记录的 PPO 提案。它是 BC 训练对照，不是模拟专家反馈。\n\n"
        "原始 HITL 会话目录未被修改。该对照是在看到当前真人 BC 结果后补充的机制分析，属于探索性结果，不是事前预注册验证。"
        "2017—2020年虽未提供反馈标签，但先前已被用于探索性配置比较，不属于完全独立盲测。\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path, help="Frozen genuine-human HITL session")
    parser.add_argument("--output-dir", type=Path, help="New output folder; must not already exist")
    args = parser.parse_args()
    output = run(args.session, args.output_dir)
    print(f"Follow-up experiment complete: {output}")
    print(f"Summary: {output / 'experiment_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
