"""二期 HITL 可编程入口（CLI / Web / ``api.hitl_*`` 共用）。

流程::

    seed → review →（确认/修改）→ resync → … → 各年审完 → bc → compare_three

落盘：``output/hitl_sessions/<sess_id>/``（见 ``session.py``）。
须先有一期 ``output/runs/run_*/``。
"""
from __future__ import annotations

import json
import os
import shutil
import hashlib
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from common.data_prep import prepare_eval_workdir
from common.model_io import load_ppo_checkpoint, sidecar_config_path
from common.scenario import set_active_scenario

from .bc_finetune import finetune_bc
from .config import BC_DEFAULT, BCConfig
from .recompute import (
    EXEC_PROTOCOL_VERSION,
    recompute_consistent_trajectory,
    save_labels,
    trajectory_fingerprint,
    verify_review_vs_labels,
)
from .resync import resync_review_with_policy
from .metrics_report import compute_absorption, timed_stage
from .rollout import list_policy_years, rollout_all_years, save_review_table
from .session import SessionPaths, resolve_session


def _session(work: str | Path | None = None, *, create: bool = True) -> SessionPaths:
    return SessionPaths(resolve_session(work, create=create)).ensure()


def _resolve_model(sp: SessionPaths, model_path: str | Path | None = None) -> Path:
    if model_path:
        p = Path(model_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"指定模型不存在: {p}")
    if sp.model.exists():
        return sp.model
    raise FileNotFoundError(
        f"会话内无权重 {sp.model}：请先 run.py pipeline，再 hitl seed"
    )


def _load_review_df(sp: SessionPaths, review: pd.DataFrame | str | Path | None = None) -> pd.DataFrame:
    if isinstance(review, pd.DataFrame):
        return review
    if review is not None:
        p = Path(review)
        return pd.read_json(p) if p.suffix.lower() == ".json" else pd.read_csv(p)
    for p in (sp.review_json, sp.review_csv):
        if p.exists():
            return pd.read_json(p) if p.suffix.lower() == ".json" else pd.read_csv(p)
    raise FileNotFoundError(f"找不到审阅表（先 hitl review）: {sp.review_json}")


def _save_current_review(sp: SessionPaths, df: pd.DataFrame) -> None:
    from .exec_layer import date_ymd

    out = df.copy()
    if "date" in out.columns:
        out["_ord"] = out["date"].map(date_ymd)
        cols = ["year", "_ord"] if "year" in out.columns else ["_ord"]
        out = out.sort_values(cols, kind="mergesort").drop(columns=["_ord"]).reset_index(drop=True)
    save_review_table(out, sp.review_csv)
    out.to_json(sp.review_json, orient="records", force_ascii=False)


def seed(
    work: str | Path | None = None,
    *,
    scenario: str | None = None,
    new_session: bool = False,
    from_run: str | Path | None = None,
) -> dict[str, Any]:
    """从一期 run 整拷 ``01_train``+``02_eval`` → 会话，并准备 ``_workdir``。

    Parameters
    ----------
    work
        会话根；``None`` 或 ``new_session`` 则新建 ``sess_*``。
    from_run
        一期 ``runs/run_*`` 路径；省略则用最近一次 run。
    scenario
        覆盖 ``configs/scenario.json``。

    Returns
    -------
    dict
        ``session, run_id, from_run, workdir, copied, missing, ok``
    """
    if new_session or work is None:
        sp = _session(None, create=True)
    else:
        sp = _session(work, create=True)

    with timed_stage(sp, "seed"):
        sc = set_active_scenario(scenario)
        wd = sp.workdir
        prepare_eval_workdir(work=wd, scenario=sc, validate=True)

        from common.run_store import latest_run, resolve_run

        if from_run is not None:
            run_root = resolve_run(from_run, create=False)
        else:
            run_root = latest_run()

        report: dict[str, Any] = {
            "session": str(sp.root),
            "workdir": str(wd),
            "from_run": str(run_root) if run_root else None,
            "run_id": run_root.name if run_root else None,
            "scenario": sc.as_dict(),
            "copied": ["forecast_eval", "actual_eval", "growth_para"],
            "missing": [],
        }

        if run_root is None or not (run_root / "01_train").exists():
            report["missing"].append(
                "no_phase1_run：请先 python run.py pipeline（或指定 --from-run 指向含 best 的 run）"
            )
            report["ok"] = False
            report["work"] = str(sp.root)
            (sp.train / "seed_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return report

        best = run_root / "01_train" / "ppo_rice_best.pth"
        if not best.is_file():
            report["missing"].append(
                f"no_best_pth：{best} 不存在（该 run 未训完）。换 --from-run 或先 train"
            )
            report["ok"] = False
            report["work"] = str(sp.root)
            (sp.train / "seed_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return report

        ingest = sp.ingest_phase1_run(run_root)
        report["copied"].extend(ingest.get("copied_dirs") or [])
        report["missing"].extend(ingest.get("missing") or [])
        # 原始训练/评估表：run.uploads 已整拷；再补 scenario 快照（缺啥补啥）
        snap = sp.snapshot_scenario_tables(sc)
        if snap:
            report["copied"].append(f"uploads_snapshot:{','.join(snap)}")
        if sp.model.exists():
            shutil.copy2(sp.model, wd / "model.pth")
            report["copied"].append("workdir_model")

        (sp.train / "seed_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        sp.update_meta(seed=report, run_id=report.get("run_id"), from_run=report.get("from_run"))
        report["ok"] = bool(sp.model.exists()) and "01_train" in (ingest.get("copied_dirs") or [])
        if not report["ok"] and not report["missing"]:
            report["missing"].append("seed 后会话内仍无 ppo_rice_best.pth")
        report["work"] = str(sp.root)
        return report


def review(
    *,
    model_path: str | Path | None = None,
    work: str | Path | None = None,
    years: Optional[list[str]] = None,
    max_years: Optional[int] = None,
    scenario: str | None = None,
    do_seed: bool = True,
    from_run: str | Path | None = None,
) -> pd.DataFrame:
    """冻结原 PPO 整季 rollout → ``03_review/current``；锁定 ``original_trajectories``。

    Returns
    -------
    DataFrame
        逐日审阅表（含 ``a_model/a_expert/confirmed/...``）。
    """
    sp = _session(work, create=True)
    if do_seed:
        info = seed(sp.root, scenario=scenario, from_run=from_run)
        if not info.get("ok"):
            raise FileNotFoundError(
                "seed 失败：" + "; ".join(info.get("missing") or ["未知"])
            )

    model_p = _resolve_model(sp, model_path)
    prev = Path.cwd()
    os.chdir(sp.workdir)
    try:
        with timed_stage(sp, "review", max_years=max_years):
            agent, _, _ = load_ppo_checkpoint(model_p)
            import train.PPO_Rice_env as PPO_Rice_env

            env = PPO_Rice_env.Irrigation()
            from .config import resolve_year_split

            avail = list_policy_years(env)
            if years is None:
                if max_years is not None:
                    years = avail[: max_years]
                else:
                    # 默认只滚「审阅年」（专家改写 + 行为克隆样本）；留出验证年不进审阅表
                    years = resolve_year_split(avail)["review_years"] or avail
            df = rollout_all_years(agent, env, years=years)

            if sp.compare_daily.exists() and len(df) and "date" in df.columns:
                disagree = pd.read_csv(sp.compare_daily)
                if "disagree" in disagree.columns:
                    bad = set(disagree.loc[disagree["disagree"] == 1, "date"].astype(str))
                    df["prefer_review"] = df["date"].astype(str).isin(bad).astype(int)

            _save_current_review(sp, df)
            sp.save_original_trajectories(df, force=False)
            split = resolve_year_split(avail)
            sp.update_meta(
                review_days=int(len(df)),
                year_split={
                    **split,
                    "review_used": [str(y) for y in years],
                    "note": (
                        "审阅表默认仅含审阅年；一期二方=全评估年；"
                        "三方默认=全评估年，并另汇总留出验证年"
                    ),
                },
            )
            return df
    finally:
        os.chdir(prev)


def resync(
    review: pd.DataFrame | str | Path | None = None,
    *,
    model_path: str | Path | None = None,
    work: str | Path | None = None,
    fresh_indices: set[int] | None = None,
    override_indices: set[int] | None = None,
) -> pd.DataFrame:
    """审阅期：保留已确认动作 → 重推环境 → 冻结 PPO 刷新后续建议。

    更新 ``current/``，不碰 ``original_trajectories/``（不再写 history）。
    ``override_indices``：本轮改了动作的行（作废起点）；``fresh_indices``：确认/改动行。
    """
    sp = _session(work, create=False)
    review_df = _load_review_df(sp, review)
    # 与看板一致：按日历序重置行号，保证行号与 /api/rows 的 i 对齐
    from .exec_layer import date_ymd

    if "date" in review_df.columns and len(review_df):
        review_df = review_df.copy()
        review_df["_ord"] = review_df["date"].map(date_ymd)
        cols = ["year", "_ord"] if "year" in review_df.columns else ["_ord"]
        review_df = (
            review_df.sort_values(cols, kind="mergesort")
            .drop(columns=["_ord"])
            .reset_index(drop=True)
        )
    model_p = _resolve_model(sp, model_path)
    prev = Path.cwd()
    os.chdir(sp.workdir)
    try:
        with timed_stage(sp, "resync"):
            agent, _, _ = load_ppo_checkpoint(model_p)
            import train.PPO_Rice_env as PPO_Rice_env

            df = resync_review_with_policy(
                review_df,
                agent,
                PPO_Rice_env.Irrigation(),
                fresh_indices=fresh_indices,
                override_indices=override_indices,
            )
            _save_current_review(sp, df)
            sp.update_meta(
                last_resync_needs=int(df["needs_reconfirm"].sum())
                if "needs_reconfirm" in df.columns
                else 0
            )
            return df
    finally:
        os.chdir(prev)


def _count_expert_edits(review_df: pd.DataFrame) -> int:
    """可编辑日上专家请求 ≠ a_model 的天数（全表可编辑日，不限已定稿年）。"""
    from .exec_layer import expert_action_float

    if review_df.empty or "a_model" not in review_df.columns:
        return 0
    df = review_df
    if "editable" in df.columns:
        df = df[df["editable"].fillna(True).astype(bool)]
    if df.empty or "a_expert" not in df.columns:
        return 0
    n = 0
    for _, r in df.iterrows():
        ex = expert_action_float(r.get("a_expert"))
        if ex is None:
            continue
        try:
            if abs(float(ex) - float(r["a_model"])) > 1e-6:
                n += 1
        except (TypeError, ValueError):
            continue
    return n


def assess_bc_readiness(
    review_df: pd.DataFrame,
    meta: Optional[dict[str, Any]] = None,
    *,
    workdir: str | Path | None = None,
    require_edits: bool = True,
    require_finalized_years: bool = True,
    check_fingerprints: bool = True,
    check_parity: bool = True,
) -> dict[str, Any]:
    """看板 ``bc_ready`` 与 CLI ``bc()`` 共用的预检（不含真正开训）。

    ``check_parity=True`` 时需可读评估工作目录（``workdir``），会做季初连续重放。
    """
    meta = dict(meta or {})
    blockers: list[str] = []
    feedback_source = str(meta.get("feedback_source") or "").strip()
    feedback_status = str(meta.get("feedback_status") or "").strip()
    accepted_human_sources = {"human", "human_whole_season_acceptance"}
    acceptance_check_only = feedback_source == "acceptance_only" and not require_edits
    if feedback_source not in accepted_human_sources and not acceptance_check_only:
        pending_note = (
            "（会话仍为 pending，须定稿后才写入 feedback_source）"
            if feedback_status == "pending" or meta.get("review_mode") == "human_dashboard"
            else ""
        )
        blockers.append(
            "BC 正式流程仅接受真人审阅反馈；"
            f"当前 feedback_source={feedback_source or '未记录'}{pending_note}"
        )
    empty = review_df is None or review_df.empty
    if empty:
        return {
            "ok": False,
            "bc_ready": False,
            "blockers": ["无审阅表"],
            "n_edit": 0,
            "n_needs_reconfirm": 0,
            "n_unconfirmed_editable": 0,
            "missing_review_years": [],
            "fingerprint_bad_years": [],
            "parity": None,
            "review_years_needed": [],
            "finalized_years": [str(y) for y in (meta.get("finalized_years") or [])],
        }

    df = review_df
    n_need = (
        int(df["needs_reconfirm"].fillna(False).astype(bool).sum())
        if "needs_reconfirm" in df.columns
        else 0
    )
    if n_need:
        blockers.append(f"仍有 {n_need} 天 needs_reconfirm")

    editable = df.get("editable", pd.Series(True, index=df.index)).fillna(False).astype(bool)
    confirmed = df.get("confirmed", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    n_unconfirmed = int((editable & ~confirmed).sum())
    if n_unconfirmed:
        blockers.append(f"存在 {n_unconfirmed} 个未确认的可编辑日")

    n_edit = _count_expert_edits(df)
    if require_edits and n_edit <= 0:
        blockers.append("改动日为 0")

    from .config import resolve_year_split

    years_in_table = (
        sorted(df["year"].astype(str).unique()) if "year" in df.columns else []
    )
    split = meta.get("year_split") if isinstance(meta.get("year_split"), dict) else None
    expected = list((split or {}).get("review_years") or (split or {}).get("review_used") or [])
    if not expected:
        expected = list(resolve_year_split(years_in_table).get("review_years") or [])
    review_years_needed = [y for y in expected if y in set(years_in_table)]
    if not review_years_needed and years_in_table and expected:
        review_years_needed = [y for y in years_in_table if y in set(expected)] or list(
            years_in_table
        )
    sources_by_year = meta.get("feedback_sources_by_year")
    if isinstance(sources_by_year, dict) and sources_by_year:
        bad_sources = {
            y: str(sources_by_year.get(y) or "未记录")
            for y in review_years_needed
            if str(sources_by_year.get(y) or "") not in accepted_human_sources
            and not (not require_edits and str(sources_by_year.get(y) or "") == "acceptance_only")
        }
        if bad_sources:
            blockers.append("规范审阅年含非真人反馈来源: " + ", ".join(
                f"{y}={source}" for y, source in sorted(bad_sources.items())
            ))
    finalized = [str(y) for y in (meta.get("finalized_years") or [])]
    missing = [y for y in review_years_needed if y not in set(finalized)]
    if require_finalized_years:
        if not review_years_needed:
            blockers.append("表内无规范审阅年")
        if missing:
            blockers.append("审阅年未全部定稿: " + ", ".join(missing))

    fingerprint_bad: list[str] = []
    if check_fingerprints:
        if "year" not in df.columns:
            blockers.append("审阅表缺 year 字段")
        else:
            stored_fp = meta.get("trajectory_fingerprints") or {}
            for y, g in df.groupby(df["year"].astype(str)):
                cur = trajectory_fingerprint(g, year=str(y))
                stored = stored_fp.get(str(y))
                if not stored or stored != cur:
                    fingerprint_bad.append(str(y))
            if fingerprint_bad:
                blockers.append("轨迹指纹缺失或失效: " + ", ".join(fingerprint_bad))

    parity: dict[str, Any] | None = None
    # 轻门槛未过时不做重放，避免看板轮询浪费；指纹已坏时同样跳过
    run_parity = (
        check_parity
        and workdir is not None
        and not any(
            b.startswith("无审阅表")
            or b.startswith("仍有")
            or b.startswith("存在")
            or b.startswith("改动日")
            or b.startswith("审阅年")
            or b.startswith("表内无")
            or b.startswith("审阅表缺")
            or b.startswith("轨迹指纹")
            for b in blockers
        )
    )
    if check_parity and workdir is None:
        blockers.append("parity 检查需要 workdir")
    elif run_parity:
        assert workdir is not None  # 与上方 run_parity 条件一致，供类型收窄
        work_path = Path(workdir)
        prev = Path.cwd()
        try:
            os.chdir(work_path)
            consistent = recompute_consistent_trajectory(df)
            parity = verify_review_vs_labels(df, consistent)
            if not parity.get("ok"):
                blockers.append(
                    "定稿与连续重放不一致"
                    f"（mismatch={parity.get('n_mismatch')}）"
                )
        except Exception as e:
            parity = {"ok": False, "error": str(e)}
            blockers.append(f"连续重放失败: {e}")
        finally:
            os.chdir(prev)

    ok = len(blockers) == 0
    return {
        "ok": ok,
        "bc_ready": ok,
        "blockers": blockers,
        "n_edit": n_edit,
        "n_needs_reconfirm": n_need,
        "n_unconfirmed_editable": n_unconfirmed,
        "missing_review_years": missing,
        "fingerprint_bad_years": fingerprint_bad,
        "parity": parity,
        "review_years_needed": review_years_needed,
        "finalized_years": finalized,
        "feedback_source": feedback_source,
    }


def bc(
    review: pd.DataFrame | str | Path | None = None,
    *,
    model_path: str | Path | None = None,
    work: str | Path | None = None,
    epochs: Optional[int] = None,
    bc_cfg: Optional[BCConfig] = None,
    save_name: str = "ppo_hitl.pth",
    require_no_pending: bool = True,
    require_edits: bool = True,
    require_finalized_years: bool = True,
) -> dict[str, Any]:
    """定稿：固定动作重放（不调 PPO）→ ``04_bc_export/labels.json`` → BC → ``05_bc/``。

    门禁与看板 ``bc_ready`` 共用 :func:`assess_bc_readiness`（含指纹与 parity）。

    ``require_no_pending`` 保留兼容：待复核始终拦截（参数忽略）。
    """
    sp = _session(work, create=False)
    review_df = _load_review_df(sp, review)
    _ = require_no_pending

    meta: dict[str, Any] = {}
    meta_path = sp.root / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    gate = assess_bc_readiness(
        review_df,
        meta,
        workdir=sp.workdir,
        require_edits=require_edits,
        require_finalized_years=require_finalized_years,
        check_fingerprints=True,
        check_parity=True,
    )
    if not gate["ok"]:
        raise ValueError("BC 预检未通过: " + "; ".join(gate["blockers"]))

    model_p = _resolve_model(sp, model_path)
    cfg = BCConfig(**{**(bc_cfg or BC_DEFAULT).to_dict()})
    if epochs is not None:
        cfg.epochs = epochs

    sp.bc_export.mkdir(parents=True, exist_ok=True)
    sp.bc.mkdir(parents=True, exist_ok=True)

    previous_cwd = Path.cwd()
    os.chdir(sp.workdir)
    try:
        with timed_stage(sp, "bc", epochs=cfg.epochs):
            consistent = recompute_consistent_trajectory(review_df)
            labels_path = save_labels(consistent, sp.labels)
            parity = verify_review_vs_labels(review_df, consistent)
            (sp.bc_export / "review_labels_parity.json").write_text(
                json.dumps(parity, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if not parity.get("ok"):
                raise ValueError(
                    "定稿轨迹与连续重放不一致，已阻止 BC。"
                    f" compared={parity.get('n_compared')}"
                    f" mismatch={parity.get('n_mismatch')}。"
                    "请回到审阅 resync 并重新确认。"
                    f" 详见 {sp.bc_export / 'review_labels_parity.json'}"
                )
            agent, model_cfg, _ = load_ppo_checkpoint(model_p)
            save_path = sp.bc / save_name
            hist = finetune_bc(agent, consistent, cfg=cfg, save_path=save_path)
            side = sidecar_config_path(save_path)
            # BC 实际配置（勿与原 PPO sidecar 的 lr 混淆）
            bc_side = {
                **model_cfg,
                "kind": "bc_finetune",
                "exec_protocol": EXEC_PROTOCOL_VERSION,
                "bc": cfg.to_dict(),
                "base_model": str(model_p),
                "base_model_sha256": hashlib.sha256(model_p.read_bytes()).hexdigest(),
                "base_model_cfg": model_cfg,
                "best_epoch": hist.get("best_epoch"),
                "n_samples": hist.get("n_samples"),
                "train_indices": hist.get("train_indices"),
                "val_indices": hist.get("val_indices"),
                "validation_years": hist.get("validation_years"),
                "split_unit": hist.get("split_unit"),
                "n_train": hist.get("n_train"),
                "n_validation": hist.get("n_validation"),
                "trajectory_fingerprint_all": trajectory_fingerprint(review_df),
                "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            }
            side.write_text(
                json.dumps(bc_side, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            src_side = sidecar_config_path(model_p)
            if src_side.exists():
                (sp.bc / "base_ppo_config.json").write_text(
                    src_side.read_text(encoding="utf-8"), encoding="utf-8"
                )
            out = {
                "session": str(sp.root),
                "work": str(sp.root),
                "labels": str(labels_path),
                "hitl_model": str(save_path),
                "exec_protocol": EXEC_PROTOCOL_VERSION,
                "n_labels": int(len(consistent)),
                "n_include_bc": int(consistent["include_in_bc"].sum())
                if "include_in_bc" in consistent.columns
                else int(len(consistent)),
                "n_edited": int((consistent["flag"] == "edited").sum())
                if "flag" in consistent.columns
                else 0,
                "review_labels_parity": {
                    "ok": bool(parity.get("ok")),
                    "n_compared": int(parity.get("n_compared") or 0),
                    "n_mismatch": int(parity.get("n_mismatch") or 0),
                    "path": str(sp.bc_export / "review_labels_parity.json"),
                },
                "bc": {
                    "best_epoch": hist.get("best_epoch"),
                    "ce": hist.get("ce"),
                    "val_ce": hist.get("val_ce"),
                    "kl": hist.get("kl"),
                    "loss": hist.get("loss"),
                    "lr": hist.get("lr"),
                    "config": hist.get("config"),
                    "n_samples": hist.get("n_samples"),
                    "train_indices": hist.get("train_indices"),
                    "val_indices": hist.get("val_indices"),
                    "validation_years": hist.get("validation_years"),
                    "split_unit": hist.get("split_unit"),
                    "n_train": hist.get("n_train"),
                    "n_validation": hist.get("n_validation"),
                },
            }
            (sp.bc / "hitl_result.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            sp.update_meta(bc=out)
            out["absorption"] = compute_absorption(sp, hitl_model=save_path)
            return out
    finally:
        os.chdir(previous_cwd)


def finalize_review(work, review_df, *, source="human", years=None):
    """只定稿已完成审阅的轨迹；whole-season finalize 明确接纳非待复核日。"""
    sp = _session(work, create=False)
    df = review_df.copy()
    if df.empty or "year" not in df.columns:
        raise ValueError("审阅表为空或缺 year 字段，不能定稿")
    selected = set(map(str, years or df["year"].astype(str).unique()))
    available = set(df["year"].astype(str))
    missing = sorted(selected - available)
    if missing:
        raise ValueError("审阅表不存在所选年份: " + ", ".join(missing))
    take = df["year"].astype(str).isin(selected)
    editable = df.get("editable", pd.Series(True, index=df.index)).fillna(False).astype(bool)
    pending = df.get("needs_reconfirm", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    pending_selected = take & editable & pending
    if bool(pending_selected.any()):
        dates = df.loc[pending_selected, "date"].astype(str).tolist()
        raise ValueError(
            f"所选年份仍有 {len(dates)} 天待专家复核，不能定稿；"
            f"请逐日认可当前 PPO 建议或指定专家动作。日期示例: {dates[:8]}"
        )
    # 此调用本身代表专家明确接纳整季；只确认非待复核日，绝不清除 pending 标记。
    if "confirmed" not in df.columns:
        df["confirmed"] = False
    df.loc[take & editable & ~pending, "confirmed"] = True
    previous_cwd = Path.cwd()
    try:
        os.chdir(sp.workdir)
        labels = recompute_consistent_trajectory(df.loc[take])
        parity = verify_review_vs_labels(df.loc[take], labels)
        if not parity["ok"]:
            raise ValueError(f"定稿轨迹不一致: {parity}")
    finally:
        os.chdir(previous_cwd)
    _save_current_review(sp, df)
    meta = json.loads((sp.root / "meta.json").read_text(encoding="utf-8"))
    fps = dict(meta.get("trajectory_fingerprints") or {})
    sources_by_year = dict(meta.get("feedback_sources_by_year") or {})
    for year in selected:
        fps[year] = trajectory_fingerprint(df, year=year)
        sources_by_year[year] = source
    sp.update_meta(
        trajectory_fingerprints=fps,
        finalized_years=sorted(set(meta.get("finalized_years", [])) | selected),
        feedback_source=source,
        feedback_sources_by_year=sources_by_year,
        feedback_status="recorded",
        exec_protocol=EXEC_PROTOCOL_VERSION,
    )
    sp.append_edit({"action": "finalize", "years": sorted(selected), "source": source, "fingerprints": fps})
    return df


def pipeline(
    *,
    work: str | Path | None = None,
    model_path: str | Path | None = None,
    scenario: str | None = None,
    max_years: Optional[int] = None,
    years: Optional[list[str]] = None,
    epochs: Optional[int] = None,
    skip_bc: bool = False,
    from_run: str | Path | None = None,
    new_session: bool = True,
) -> dict[str, Any]:
    """一键：新建会话 → seed → review →（可选 bc）。不含 compare（需人工确认后另调）。"""
    sp = _session(None if new_session and work is None else work, create=True)
    seed_info = seed(sp.root, scenario=scenario, from_run=from_run)
    if not seed_info.get("ok"):
        raise FileNotFoundError(
            "seed 失败：" + "; ".join(seed_info.get("missing") or ["未知"])
        )
    df = review(
        model_path=model_path,
        work=sp.root,
        years=years,
        max_years=max_years,
        scenario=scenario,
        do_seed=False,
    )
    out: dict[str, Any] = {
        "session": str(sp.root),
        "seed": seed_info,
        "review_days": int(len(df)),
        "review_years": int(df["year"].nunique()) if len(df) and "year" in df.columns else 0,
        "review_csv": str(sp.review_csv),
    }
    if not skip_bc:
        editable = df.get("editable", pd.Series(True, index=df.index)).fillna(False).astype(bool)
        confirmed = df.get("confirmed", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        if bool((editable & ~confirmed).any()):
            out["bc"] = {"status": "awaiting_review", "next": "审阅 → finalize → bc"}
        else:
            out["bc"] = bc(model_path=model_path, work=sp.root, epochs=epochs)
    return out
