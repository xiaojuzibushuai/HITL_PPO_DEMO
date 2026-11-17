"""Sensitivity of a fixed human-feedback BC configuration to random seed.

This is a post-hoc robustness analysis. It does not tune hyperparameters or
modify the human review session. The same 2016 validation season is used for
checkpoint selection in every seed, and 2017-2020 remain unreviewed evaluation
years rather than a pristine blind test.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import evaluate
from common.model_io import load_ppo_checkpoint, sidecar_config_path
from hitl.bc_finetune import finetune_bc
from hitl.config import BCConfig
from hitl.human_followup_experiment import _probability_rows


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 43, 44])
    args = ap.parse_args()
    session = args.session.resolve()
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)

    labels_path = session / "04_bc_export" / "labels.json"
    labels = pd.read_json(labels_path)
    base_model = session / "01_train" / "ppo_rice_best.pth"
    human_model = session / "05_bc" / "ppo_hitl.pth"
    side = json.loads(sidecar_config_path(human_model).read_text(encoding="utf-8"))
    frozen_cfg = dict(side["bc"])
    assert int(frozen_cfg["seed"]) == 42
    assert list(map(str, frozen_cfg["val_years"])) == ["2016"]
    assert set(args.seeds).isdisjoint({42})
    records = []

    for seed in args.seeds:
        arm = out / f"seed_{seed}"
        arm.mkdir()
        agent, base_cfg, _ = load_ppo_checkpoint(base_model)
        cfg = BCConfig(**{**frozen_cfg, "seed": seed})
        model = arm / "ppo_hitl.pth"
        with (arm / "training.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            history = finetune_bc(agent, labels, cfg=cfg, save_path=model)
        sidecar_config_path(model).write_text(
            json.dumps({**base_cfg, "kind": "bc_finetune", "bc": cfg.to_dict(),
                        "base_model": str(base_model), "base_model_sha256": digest(base_model),
                        "labels_sha256": digest(labels_path), "best_epoch": history["best_epoch"]},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (arm / "evaluation.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            evaluate(model, out_dir=arm / "eval", only="ppo", scenario=None)
        trained, _, _ = load_ppo_checkpoint(model)
        base, _, _ = load_ppo_checkpoint(base_model)
        _, probs = _probability_rows(
            labels, {"ppo": (base, list(base.action_space)),
                     "human_feedback_bc": (trained, list(trained.action_space))}
        )
        yearly = pd.read_csv(arm / "eval" / "yearly_metrics_PPO.csv")
        for split, frame in [("all_8", yearly), ("unreviewed_4", yearly[yearly.year >= 2017])]:
            records.append({
                "seed": seed, "split": split, "best_epoch": history["best_epoch"],
                "managed_water": float(frame.managed_water.mean()),
                "operation_days": float((frame.irrigation_times + frame.drainage_times).mean()),
                "rain_utilization": float(frame.rain_utilization.mean()),
                "stress_days": float(frame.stress_days.mean()),
                "yield_factor": float(frame.yield_factor.mean()),
                "expert_prob": probs["all"]["human_feedback_bc"]["mean_expert_action_probability"],
                "expert_top1": probs["all"]["human_feedback_bc"]["top1_expert_matches"],
                "model_sha256": digest(model),
            })
        print(f"seed {seed} complete", flush=True)

    # Existing, frozen human run is seed 42. Include its unchanged published
    # checkpoint and yearly evaluation as the fifth seed.
    base, _, _ = load_ppo_checkpoint(base_model)
    trained, _, _ = load_ppo_checkpoint(human_model)
    _, probs = _probability_rows(labels, {
        "ppo": (base, list(base.action_space)),
        "human_feedback_bc": (trained, list(trained.action_space)),
    })
    yearly = pd.read_csv(session / "06_compare" / "hitl_ppo" / "yearly_metrics_PPO.csv")
    for split, frame in [("all_8", yearly), ("unreviewed_4", yearly[yearly.year >= 2017])]:
        records.append({
            "seed": 42, "split": split, "best_epoch": side.get("best_epoch"),
            "managed_water": float(frame.managed_water.mean()),
            "operation_days": float((frame.irrigation_times + frame.drainage_times).mean()),
            "rain_utilization": float(frame.rain_utilization.mean()),
            "stress_days": float(frame.stress_days.mean()),
            "yield_factor": float(frame.yield_factor.mean()),
            "expert_prob": probs["all"]["human_feedback_bc"]["mean_expert_action_probability"],
            "expert_top1": probs["all"]["human_feedback_bc"]["top1_expert_matches"],
            "model_sha256": digest(human_model),
        })
    df = pd.DataFrame(records).sort_values(["split", "seed"])
    df.to_csv(out / "per_seed_metrics.csv", index=False, encoding="utf-8-sig")
    summary = df.groupby("split").agg({
        key: ["mean", "std", "min", "max"] for key in
        ["managed_water", "operation_days", "rain_utilization", "stress_days",
         "yield_factor", "expert_prob", "expert_top1"]
    })
    summary.to_csv(out / "summary.csv", encoding="utf-8-sig")
    (out / "protocol.json").write_text(json.dumps({
        "status": "post-hoc fixed-configuration seed sensitivity",
        "source_session": str(session), "source_labels_sha256": digest(labels_path),
        "base_model_sha256": digest(base_model), "seeds": sorted([*args.seeds, 42]),
        "bc_config_except_seed": {k: v for k, v in frozen_cfg.items() if k != "seed"},
        "unreviewed_2017_2020_not_blind": True,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out / "summary.csv")


if __name__ == "__main__":
    main()
