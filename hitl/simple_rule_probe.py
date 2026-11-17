"""Post-hoc simple-rule probe for the human-feedback study.

This deliberately broad rule is derived after inspecting expert edits: when
the frozen PPO asks for -10 mm drainage and zero is legal, request zero.
It is an exploratory challenge baseline, not a prespecified comparator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.action_mask import ACTION_SPACE, valid_action_mask
from evaluation import run_eval as evaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    model = args.model.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(out_dir)
    original = evaluator._greedy_ppo_mm
    counts = {"ppo_minus10": 0, "replaced": 0}
    zero_index = list(ACTION_SPACE).index(0)

    def choose(agent, state_vec: np.ndarray, state_row, cfg: dict):
        raw, executed = original(agent, state_vec, state_row, cfg)
        if abs(raw + 10.0) > 1e-6:
            return raw, executed
        counts["ppo_minus10"] += 1
        state = torch.as_tensor(state_vec, dtype=torch.float32).reshape(1, -1)
        if bool(valid_action_mask(state, ACTION_SPACE)[0, zero_index].item()):
            counts["replaced"] += 1
            return 0.0, 0.0
        return raw, executed

    evaluator._greedy_ppo_mm = choose
    result = evaluator.run_eval(model, out_dir, only="ppo", scenario=None)
    (out_dir / "probe_definition.json").write_text(
        json.dumps(
            {
                "definition": "PPO -10 mm drainage request -> 0 if 0 is legal; otherwise unchanged",
                "status": "post-hoc exploratory challenge baseline",
                "counts": counts,
                "evaluation": result,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
