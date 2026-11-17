"""HITL CLI 实现：``run.py hitl`` 与 ``python -m hitl`` 共用。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py hitl",
        description="二期 HITL：seed → review → resync → bc",
    )
    p.add_argument("--scenario", default=None, help="覆盖 configs/scenario.json")
    sub = p.add_subparsers(dest="hitl_cmd")

    pipe = sub.add_parser("pipeline", help="★ 导入并准备审阅；未定稿返回 awaiting_review")
    pipe.add_argument("--work", default=None)
    pipe.add_argument("--from-run", default=None, help="一期 output/runs/run_<ts>/；默认最近一次")
    pipe.add_argument("--model", default=None)
    pipe.add_argument("--max-years", type=int, default=None)
    pipe.add_argument("--epochs", type=int, default=None)
    pipe.add_argument("--skip-bc", action="store_true")

    s = sub.add_parser("seed", help="导入一期评估表 + best")
    s.add_argument("--work", default=None)
    s.add_argument("--from-run", default=None, help="一期 output/runs/run_<ts>/；默认最近一次")

    r = sub.add_parser("review", help="rollout → review.csv")
    r.add_argument("--work", default=None)
    r.add_argument("--model", default=None)
    r.add_argument("--max-years", type=int, default=None)

    rs = sub.add_parser("resync", help="审阅期：改动作后重推 + 冻结 PPO 刷新建议")
    rs.add_argument("--work", default=None)
    rs.add_argument("--model", default=None)

    fin = sub.add_parser("finalize", help="连续重放校验后绑定定稿指纹")
    fin.add_argument("--work", required=True)
    fin.add_argument("--year", action="append", default=None, help="可重复指定；默认全部审阅年")
    fin.add_argument("--source", choices=["human", "simulated_rule_reviewer"], default="human")

    b = sub.add_parser("bc", help="定稿重放 + BC → ppo_hitl.pth")
    b.add_argument("--work", default=None)
    b.add_argument("--review", default=None)
    b.add_argument("--model", default=None)
    b.add_argument("--epochs", type=int, default=None)

    c = sub.add_parser("compare", help="BC 后三方对比：规则 / 原PPO / HITL")
    c.add_argument("--work", default=None)
    c.add_argument("--max-years", type=int, default=None)

    ra = sub.add_parser(
        "rationale-audit",
        help="结构化改因事后核查 → 07_metrics/expert_rationale_audit.json",
    )
    ra.add_argument("--work", default=None)

    board = sub.add_parser("board", help="★ 打开审阅看板 http://127.0.0.1:5050")
    board.add_argument("--host", default="127.0.0.1")
    board.add_argument("--port", type=int, default=5050)

    return p


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    # 无子命令 → pipeline（对齐 run.py 无参默认 pipeline）
    known = {
        "pipeline",
        "seed",
        "review",
        "resync",
        "finalize",
        "bc",
        "compare",
        "rationale-audit",
        "board",
    }
    if not any(a in known for a in argv):
        argv = ["pipeline", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = args.hitl_cmd or "pipeline"
    scenario = getattr(args, "scenario", None)

    if cmd == "seed":
        from hitl import seed

        info = seed(args.work, scenario=scenario, from_run=getattr(args, "from_run", None))
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0 if info.get("ok") else 1

    if cmd == "review":
        from hitl import review

        df = review(
            model_path=args.model,
            work=args.work,
            max_years=args.max_years,
            scenario=scenario,
        )
        n_y = int(df["year"].nunique()) if len(df) and "year" in df.columns else 0
        print(f"review_days={len(df)} years={n_y}")
        return 0

    if cmd == "resync":
        from hitl import resync

        df = resync(model_path=args.model, work=args.work)
        n_c = int(df["confirmed"].sum()) if "confirmed" in df.columns else 0
        n_n = int(df["needs_reconfirm"].sum()) if "needs_reconfirm" in df.columns else 0
        print(f"resync_days={len(df)} confirmed={n_c} needs_reconfirm={n_n}")
        return 0

    if cmd == "finalize":
        from hitl.pipeline import finalize_review, _load_review_df
        from hitl.session import SessionPaths, resolve_session
        sp = SessionPaths(resolve_session(args.work, create=False))
        df = finalize_review(sp.root, _load_review_df(sp), years=args.year, source=args.source)
        print(json.dumps({"session": str(sp.root), "days": len(df), "status": "finalized"}, ensure_ascii=False))
        return 0

    if cmd == "bc":
        from hitl import bc

        out = bc(
            Path(args.review) if args.review else None,
            model_path=args.model,
            work=args.work,
            epochs=args.epochs,
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    if cmd == "pipeline":
        from hitl import pipeline

        out = pipeline(
            work=args.work,
            model_path=args.model,
            scenario=scenario,
            max_years=args.max_years,
            epochs=args.epochs,
            skip_bc=bool(args.skip_bc),
            from_run=getattr(args, "from_run", None),
        )
        # 摘要打印，避免把整段 ce 曲线刷屏
        summary = {
            "review_days": out.get("review_days"),
            "review_years": out.get("review_years"),
            "review_csv": out.get("review_csv"),
            "hitl_model": (out.get("bc") or {}).get("hitl_model"),
            "n_edited": (out.get("bc") or {}).get("n_edited"),
            "ok": True,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("HITL_PIPELINE_OK")
        return 0

    if cmd == "compare":
        from hitl.compare_eval import compare_three
        from hitl.expert_rationale_analysis import analyze_expert_rationales

        out = compare_three(args.work, scenario=scenario, max_years=args.max_years)
        try:
            audit = analyze_expert_rationales(out.get("session") or args.work)
        except Exception as e:
            audit = {"ok": False, "error": str(e)}
        print(
            json.dumps(
                {
                    "session": out.get("session"),
                    "metrics": out.get("metrics"),
                    "expert_rationale_audit": audit.get("path") or audit,
                    "ok": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if cmd == "rationale-audit":
        from hitl.expert_rationale_analysis import analyze_expert_rationales

        out = analyze_expert_rationales(args.work)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if out.get("ok") else 1

    if cmd == "board":
        from .web.app import main as board_main

        board_main(host=args.host, port=args.port)
        return 0

    return 1
