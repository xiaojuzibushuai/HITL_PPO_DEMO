#!/usr/bin/env python
"""PPO_Rice_Env 统一 CLI 入口。

直接运行本文件（无参数）= 正式一键 pipeline::

  python run.py
  # 等同 python run.py pipeline

分步::

  python run.py show-config
  python run.py train --episodes 1500
  python run.py eval --model output/runs/run_*/01_train/ppo_rice_best.pth --out-dir ...
  python run.py hitl board           # 二期看板（与 CLI 共用 hitl_sessions）
  python run.py hitl seed|review|resync|bc|compare

可编程：``python main.py``（改 METHOD）或 ``from api import dispatch``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from common.model_io import load_model_config, load_ppo_checkpoint  # noqa: E402
from common.paths import ROOT, ensure_repo_path, mkdir  # noqa: E402
from common.scenario import set_active_scenario, validate_scenario_data  # noqa: E402

ensure_repo_path()


def cmd_show_config(args: argparse.Namespace) -> int:
    """打印情景 + 模型协议，并校验训练/评估表（每年季内日期集合精确对齐）。"""
    sc = set_active_scenario(args.scenario)
    model = load_model_config(args.model_config)
    data_report = validate_scenario_data(sc, need_train=True, need_eval=True)
    payload = {
        "scenario": sc.as_dict(),
        "data_ok": data_report,
        "model": {
            "policy_mode": model.get("policy_mode"),
            "n_states": model.get("n_states"),
            "n_actions": model.get("n_actions"),
            "hidden_dim": model.get("hidden_dim"),
            "decision_defaults": model.get("decision_defaults"),
            "evaluation_defaults": model.get("evaluation_defaults"),
        },
        "paths": {
            "root": str(ROOT),
            "scenario_file": str(Path(args.scenario) if args.scenario else ROOT / "configs" / "scenario.json"),
            "model_config_file": str(
                Path(args.model_config) if args.model_config else ROOT / "configs" / "model_config.json"
            ),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """训练 PPO：只落盘 best / last。``--save-dir`` 省略则新建 ``runs/run_*/01_train``。"""
    import os

    from common.run_store import RunPaths, create_run

    sc = set_active_scenario(args.scenario)
    from common.data_prep import prepare_train_workdir
    from train.ppo_model import save_training_results, train_ppo

    train_cfg = load_model_config(args.model_config)
    save_arg = getattr(args, "save_dir", None)
    run_root = None
    if save_arg:
        save_dir = str(mkdir(save_arg))
    else:
        run_root = create_run(note="run.py train")
        save_dir = str(RunPaths(run_root).ensure().train)
    prev_cwd = Path.cwd()
    prepare_train_workdir(sc, chdir=True, validate=True)
    try:
        result = train_ppo(
            total_episodes=args.episodes,
            trajectory_num=args.trajectory_num,
            save_dir=save_dir,
            model_name=args.model_name,
            config=train_cfg,
            verbose=True,
        )
    finally:
        os.chdir(prev_cwd)
    if run_root is not None:
        print("run:", run_root)
    print("best:", result.get("model_path"))
    print("last:", result.get("last_model_path"))
    print("best_ep:", result.get("best_episode"), "best_score:", result.get("best_mean_return"))
    try:
        arts = save_training_results(result, save_dir=save_dir)
        print("plots:", arts.get("plot"))
        print("log_csv:", arts.get("log"))
    except Exception as e:
        print("save_training_results skipped:", e)
    return 0


def cmd_self_check(args: argparse.Namespace) -> int:
    """无 CSV 轻量自检：包路径锁定 + 动作掩码 / Hc 守卫 / 决策管线 / 年指标列。"""
    import pandas as pd
    import torch

    from common.action_mask import ACTION_SPACE, valid_action_mask
    from common.paths import ROOT, assert_modules_local, ensure_repo_path
    from decision.infer import apply_decision_pipeline, fixed_rule_action
    from decision.rules import apply_safety_drainage_guard, correct_policy_action
    from evaluation.metrics import compute_yearly_metrics
    import importlib

    import train.PPO_RL_brain as brain
    import train.PPO_Rice_env as env_mod
    import hitl

    # hitl/__init__ 把 pipeline 函数导出到包属性，须按子模块路径加载
    hitl_pipe_mod = importlib.import_module("hitl.pipeline")
    api_mod = importlib.import_module("api")

    ensure_repo_path()
    rels = assert_modules_local(brain, env_mod, hitl, hitl_pipe_mod, api_mod)
    print("IMPORT_ROOT", ROOT)
    for r in rels:
        print("IMPORT_OK", r)
    assert "use_action_mask" in brain.PPO.__init__.__code__.co_varnames
    # 确认不是工作区外层 hitl/
    assert Path(hitl.__file__).resolve().is_relative_to(ROOT)

    st = torch.tensor([[120.0, 20.0, 50.0, 60.0, 100.0, 0.2, 1.0, 10.0]])
    m = valid_action_mask(st, ACTION_SPACE)
    assert int(m[0].sum().item()) == 1
    assert ACTION_SPACE[int(m[0].nonzero()[0])] == -100

    g = apply_safety_drainage_guard(0.0, h=120, hp=60, hc=100)
    assert g == -100.0
    assert correct_policy_action(g, h=120, hmax=50, hp=60, hc=100) == -60.0

    r = apply_decision_pipeline(40, h=100, hmin=20, hmax=50, hp=60, hc=100)
    assert r["action"] < 0
    assert fixed_rule_action(10, 20, 50, 60, 100) > 0

    df = pd.DataFrame(
        [
            {
                "date": "2012/6/15",
                "m": 20.0,
                "d": 0.0,
                "P_1_a": 5.0,
                "et0_a": 3.0,
                "lamda": 0.2,
                "Hc": 100.0,
                "h_before": 25.0,
                "h_after": 40.0,
                "a_exec": 20.0,
            },
            {
                "date": "2012/6/16",
                "m": 0.0,
                "d": 10.0,
                "P_1_a": 0.0,
                "et0_a": 3.0,
                "lamda": 0.2,
                "Hc": 100.0,
                "h_before": 110.0,
                "h_after": 95.0,
                "a_exec": -10.0,
            },
        ]
    )
    y = compute_yearly_metrics(df)
    assert len(y) == 1 and "rain_utilization" in y.columns
    print("SELF_CHECK_OK")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """多年整季闭环评估。``--out-dir`` 省略则新建 ``runs/run_*/02_eval``。"""
    from common.run_store import RunPaths, create_run
    from evaluation.run_eval import run_eval

    if args.only != "rule" and not args.model:
        raise SystemExit("eval: only=ppo/both 时必须 --model")

    out_arg = getattr(args, "out_dir", None)
    if out_arg:
        out_dir = str(mkdir(out_arg))
    else:
        run_root = create_run(note="run.py eval")
        out_dir = str(RunPaths(run_root).ensure().eval)
        print("run:", run_root)

    set_active_scenario(args.scenario)
    info = run_eval(
        args.model,
        out_dir,
        only=args.only,
        max_years=args.max_years,
        scenario=args.scenario,
    )
    summary = info.get("summary")
    if summary is not None:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({k: v for k, v in info.items() if k != "summary"}, ensure_ascii=False, indent=2))
    return 0


def _run_steps(tag: str, steps: list[tuple[str, list[str]]]) -> int:
    """子进程串跑子命令（隔离 train 的 chdir）。成功返回 0。"""
    import subprocess

    for name, cmd in steps:
        print(f"\n===== {tag}: {name} =====", flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            print(f"{tag}_FAIL at {name} (exit={r.returncode})")
            return int(r.returncode)
    return 0


def _cfg_flags(args: argparse.Namespace) -> list[str]:
    """子命令后的 --scenario / --model-config（须在子命令名之后，避免被 default 冲掉）。"""
    flags: list[str] = []
    if getattr(args, "scenario", None):
        flags += ["--scenario", str(args.scenario)]
    if getattr(args, "model_config", None):
        flags += ["--model-config", str(args.model_config)]
    return flags


def cmd_pipeline(args: argparse.Namespace) -> int:
    """正式一键：show-config → train → eval → ``output/runs/run_<时间戳>/``。"""
    from common.run_store import ensure_pipeline_dirs

    py = sys.executable
    run = str(ROOT / "run.py")
    train_p, eval_p, run_root = ensure_pipeline_dirs(note="run.py pipeline")
    save_dir, eval_dir = str(train_p), str(eval_p)
    model = str(Path(save_dir) / f"{args.model_name}_best.pth")
    cfg = _cfg_flags(args)

    train_cmd = [
        py, run, "train", *cfg,
        "--episodes", str(args.episodes),
        "--trajectory-num", str(args.trajectory_num),
        "--save-dir", save_dir,
        "--model-name", args.model_name,
    ]
    eval_cmd = [py, run, "eval", *cfg, "--model", model, "--out-dir", eval_dir]
    if args.max_years is not None:
        eval_cmd += ["--max-years", str(args.max_years)]

    code = _run_steps(
        "PIPELINE",
        [
            ("show-config", [py, run, "show-config", *cfg]),
            ("train", train_cmd),
            ("eval", eval_cmd),
        ],
    )
    if code != 0:
        return code
    print("\nPIPELINE_OK")
    print("run:", run_root)
    print("best:", model)
    print("eval:", eval_dir)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """单日状态 → 可执行灌排 mm（无季内递推）。"""
    import numpy as np
    import torch

    from common.scenario import ks_factor
    from decision.infer import decide_from_logits

    set_active_scenario(args.scenario)
    agent, cfg, device = load_ppo_checkpoint(args.model)
    ks = float(ks_factor(args.h)) if args.ks is None else float(args.ks)
    state_vec = np.array(
        [args.h, args.hmin, args.hmax, args.hp, args.hc, args.lamda, ks, args.p_cumu7],
        dtype=np.float32,
    )
    x = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = agent.policy_net(x)
    dec = cfg.get("decision_defaults") or load_model_config(args.model_config).get("decision_defaults", {})
    post = decide_from_logits(
        logits,
        x,
        action_space=agent.action_space,
        use_action_mask=bool(agent.use_action_mask),
        hmin=args.hmin,
        hmax=args.hmax,
        hp=args.hp,
        hc=args.hc,
        p_cumu7=args.p_cumu7,
        hc_guard=bool(dec.get("hc_guard", True)),
        physical_clip=bool(dec.get("physical_clip", True)),
        drying_rule=bool(dec.get("drying_rule", False)),
        low_water_refill=bool(dec.get("low_water_refill", True)),
    )
    out = {
        "raw_action_mm": post["raw_action"],
        "exec_action_mm": post["action"],
        "dtype": post["dtype"],
        "desc": post["desc"],
        "policy_mode": cfg.get("policy_mode", "masked"),
        "safety_reason": post.get("safety_reason", ""),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def _add_cfg_flags(sp: argparse.ArgumentParser) -> None:
    """子命令上也挂配置开关；default=SUPPRESS，避免覆盖「子命令前」已解析的值。"""
    sp.add_argument(
        "--scenario",
        default=argparse.SUPPRESS,
        help="覆盖 configs/scenario.json",
    )
    sp.add_argument(
        "--model-config",
        default=argparse.SUPPRESS,
        help="覆盖 configs/model_config.json",
    )


def build_parser() -> argparse.ArgumentParser:
    """组装子命令。

    ``--scenario`` / ``--model-config`` 可写在子命令前或后::

      python run.py --scenario X train
      python run.py train --scenario X
    """
    p = argparse.ArgumentParser(prog="run.py", description="PPO_Rice_Env 统一入口")
    p.add_argument("--scenario", default=None, help="覆盖 configs/scenario.json")
    p.add_argument("--model-config", default=None, help="覆盖 configs/model_config.json")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("show-config", help="打印情景 + 模型协议 + 数据校验")
    _add_cfg_flags(s)
    s.set_defaults(func=cmd_show_config)

    pipe = sub.add_parser(
        "pipeline",
        help="★ 正式一键：show-config → train → eval → output/runs/run_<时间戳>/",
    )
    _add_cfg_flags(pipe)
    pipe.add_argument("--episodes", type=int, default=1500, help="正式建议 1000～2000")
    pipe.add_argument("--trajectory-num", type=int, default=10)
    pipe.add_argument("--model-name", default="ppo_rice")
    pipe.add_argument("--max-years", type=int, default=None, help="评估年数；默认全部验证年")
    pipe.set_defaults(func=cmd_pipeline)

    t = sub.add_parser("train", help="仅训练（只存 best + last）→ 默认新建 runs/run_*/01_train")
    _add_cfg_flags(t)
    t.add_argument("--episodes", type=int, default=1500, help="建议 1000～2000")
    t.add_argument("--trajectory-num", type=int, default=10, help="每 N 局做一次 PPO update")
    t.add_argument("--save-dir", default=None, help="省略则新建 runs/run_*/01_train")
    t.add_argument("--model-name", default="ppo_rice")
    t.set_defaults(func=cmd_train)

    sc = sub.add_parser("self-check", help="无数据断言；不能替代 eval")
    _add_cfg_flags(sc)
    sc.set_defaults(func=cmd_self_check)

    e = sub.add_parser("eval", help="整季闭环评估：PPO vs 固定规则 → 默认新建 runs/run_*/02_eval")
    _add_cfg_flags(e)
    e.add_argument("--model", default=None, help="*_best.pth；--only rule 时可省略")
    e.add_argument("--out-dir", default=None, help="省略则新建 runs/run_*/02_eval")
    e.add_argument("--only", choices=["ppo", "rule", "both"], default="both")
    e.add_argument("--max-years", type=int, default=None, help="最多评估年数；默认全部验证年")
    e.set_defaults(func=cmd_eval)

    pr = sub.add_parser("predict", help="单日状态→灌排mm（无季内递推）")
    _add_cfg_flags(pr)
    pr.add_argument("--model", required=True)
    pr.add_argument("--h", type=float, required=True)
    pr.add_argument("--hmin", type=float, required=True)
    pr.add_argument("--hmax", type=float, required=True)
    pr.add_argument("--hp", type=float, required=True)
    pr.add_argument("--hc", type=float, required=True)
    pr.add_argument("--lamda", type=float, default=0.2)
    pr.add_argument("--ks", type=float, default=None)
    pr.add_argument("--p-cumu7", type=float, default=0.0)
    pr.set_defaults(func=cmd_predict)

    # hitl 在 main() 里整段转发，避免父 argparse 吃掉 --max-years 等
    sub.add_parser("hitl", help="★ 二期 HITL：seed→review→bc（见 hitl/README）")

    return p


_SUBCOMMANDS = frozenset({
    "show-config", "pipeline", "train", "self-check", "eval", "predict", "hitl",
})


def _dispatch_hitl(argv: list[str]) -> int:
    """``run.py hitl ...`` → ``hitl.cli``（与 ``python -m hitl`` 同源）。"""
    from hitl.cli import main as hitl_main

    i = argv.index("hitl")
    pre, post = argv[:i], argv[i + 1 :]
    # 子命令前的全局 --scenario 并入 HITL argv
    j = 0
    while j < len(pre):
        if pre[j] == "--scenario" and j + 1 < len(pre):
            post = ["--scenario", pre[j + 1], *post]
            j += 2
            continue
        j += 1
    return int(hitl_main(post))


def main(argv: list[str] | None = None) -> int:
    """入口：无子命令时默认 ``pipeline``；``hitl`` 整段转交。"""
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    if "hitl" in argv:
        return _dispatch_hitl(argv)
    if not any(a in _SUBCOMMANDS for a in argv):
        argv.append("pipeline")
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
