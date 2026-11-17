"""可编程对外 API（与 ``python run.py ...`` / ``run.py hitl ...`` / ``main.py`` 等价）。

一期产物：``output/runs/run_<时间戳>/{uploads,01_train,02_eval}``  
二期产物：``output/hitl_sessions/sess_<时间戳>/``（与 Web 看板共用）

推荐入口（改 METHOD 即可）::

    python main.py          # 见 main.py 顶部 METHOD / KW

或代码::

    from api import dispatch, list_methods
    dispatch("pipeline", episodes=1500)
    dispatch("hitl_seed", from_run="output/runs/run_…")
    dispatch("hitl_review", max_years=1)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from common.paths import ROOT, ensure_repo_path, mkdir
from common.scenario import set_active_scenario, validate_scenario_data

ensure_repo_path()


def show_config(scenario: str | None = None, model_config: str | None = None) -> dict[str, Any]:
    """校验并返回情景 + 模型协议。

    Returns
    -------
    dict
        ``scenario, model, data_ok, root``
    """
    from common.model_io import load_model_config

    sc = set_active_scenario(scenario)
    model = load_model_config(model_config)
    data_ok = validate_scenario_data(sc, need_train=True, need_eval=True)
    return {"scenario": sc.as_dict(), "model": model, "data_ok": data_ok, "root": str(ROOT)}


def pipeline(
    *,
    episodes: int = 1500,
    trajectory_num: int = 10,
    model_name: str = "ppo_rice",
    scenario: str | None = None,
    model_config: str | None = None,
    max_years: int | None = None,
    verbose: bool = True,
    run_id: str | None = None,
    note: str = "api.pipeline",
) -> dict[str, Any]:
    """一期一键：``show_config`` → train → evaluate。

    每次新建 ``output/runs/run_<时间戳>/``（或指定 ``run_id``）。

    Returns
    -------
    dict
        ``config, train, eval, run, run_id``
    """
    from common.run_store import RunPaths, create_run

    from common.scenario import set_active_scenario as _set_sc

    cfg = show_config(scenario=scenario, model_config=model_config)
    if verbose:
        _print_config_gate(cfg)
    run_root = create_run(run_id=run_id, note=note, scenario=cfg.get("scenario"))
    rp = RunPaths(run_root).ensure()
    rp.snapshot_scenario_tables(_set_sc(scenario))
    if verbose:
        print("【一期 pipeline】新建 run →", run_root.name)
        print("【一期 pipeline】开始训练 episodes=%s …" % episodes)
    train_out = train(
        episodes=episodes,
        trajectory_num=trajectory_num,
        save_dir=rp.train,
        model_name=model_name,
        scenario=scenario,
        model_config=model_config,
        verbose=verbose,
    )
    if verbose:
        print("【一期 pipeline】训练结束 →", train_out.get("model_path"))
        print("【一期 pipeline】开始评估 →", rp.eval)
    eval_out = evaluate(
        train_out["model_path"],
        out_dir=rp.eval,
        max_years=max_years,
        scenario=scenario,
    )
    rp.update_meta(
        train=train_out.get("model_path"),
        eval_dir=str(rp.eval),
        best=str(rp.best_model) if rp.best_model.exists() else train_out.get("model_path"),
    )
    if verbose:
        print("【一期 pipeline】完成 run_id=%s" % run_root.name)
    return {
        "config": cfg,
        "train": train_out,
        "eval": eval_out,
        "run": str(run_root),
        "run_id": run_root.name,
    }


def _print_config_gate(cfg: dict[str, Any]) -> None:
    """pipeline / 开训前打印 1.1 校验摘要（失败时 show_config 已抛错，能走到这里=已通过）。"""
    data = cfg.get("data_ok") or {}
    sc = cfg.get("scenario") or {}
    tr = data.get("train") or {}
    ev = data.get("eval") or {}
    print()
    print("=" * 64)
    print("【1.1 校验】已通过（失败会直接报错中断，不会进入训练）")
    print("  情景:", sc.get("name") or data.get("name"))
    print("  日历: start=%s terminal=%s duration_days=%s (日历展开=%s)" % (
        data.get("start") or sc.get("start"),
        data.get("terminal") or sc.get("terminal"),
        data.get("duration_days") or sc.get("duration_days"),
        data.get("duration_from_calendar"),
    ))
    print("  训练集: ok=%s  n_years=%s  bad_years=%s" % (
        tr.get("ok"), tr.get("n_years"), tr.get("bad_years"),
    ))
    print("         forecast=%s" % (tr.get("forecast"),))
    print("         actual  =%s" % (tr.get("actual"),))
    print("  评估集: ok=%s  n_years=%s  bad_years=%s" % (
        ev.get("ok"), ev.get("n_years"), ev.get("bad_years"),
    ))
    print("         forecast=%s" % (ev.get("forecast"),))
    print("         actual  =%s" % (ev.get("actual"),))
    print("  policy_mode:", (cfg.get("model") or {}).get("policy_mode"))
    print("=" * 64)
    print()


def train(
    *,
    episodes: int = 1500,
    trajectory_num: int = 10,
    save_dir: str | Path | None = None,
    model_name: str = "ppo_rice",
    scenario: str | None = None,
    model_config: str | None = None,
    verbose: bool = True,
    progress_callback=None,
    keep_active_scenario: bool = False,
) -> dict[str, Any]:
    """训练 PPO；写出 best + last + 训练曲线。

    ``save_dir`` 省略时新建 ``runs/run_*/01_train``。  
    评估请用返回的 ``model_path``（best）。
    ``progress_callback``：每局回调 dict（看板实时进度）。
    ``keep_active_scenario``：True 时沿用已 ``activate_scenario_obj`` 的情景（看板 uploads）。

    Returns
    -------
    dict
        含 ``model_path, artifacts``；若新建 run 另含 ``run, run_id``。
    """
    from common.data_prep import prepare_train_workdir
    from common.model_io import load_model_config
    from common.run_store import RunPaths, create_run
    from common.scenario import get_scenario
    from train.ppo_model import save_training_results, train_ppo

    import os

    sc = get_scenario() if keep_active_scenario else set_active_scenario(scenario)
    train_cfg = load_model_config(model_config)
    run_root = None
    if save_dir is None:
        run_root = create_run(note="api.train")
        save_dir = RunPaths(run_root).ensure().train
    save_dir = mkdir(save_dir)
    prev_cwd = Path.cwd()
    prepare_train_workdir(sc, chdir=True, validate=True)
    try:
        result = train_ppo(
            total_episodes=episodes,
            trajectory_num=trajectory_num,
            save_dir=save_dir,
            model_name=model_name,
            config=train_cfg,
            verbose=verbose,
            progress_callback=progress_callback,
        )
    finally:
        os.chdir(prev_cwd)
    try:
        result["artifacts"] = save_training_results(result, save_dir=save_dir)
    except Exception as e:
        result["artifacts_error"] = str(e)
    if run_root is not None:
        result["run"] = str(run_root)
        result["run_id"] = run_root.name
        RunPaths(run_root).update_meta(train=result.get("model_path"))
    return result


def evaluate(
    model: str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
    only: str = "both",
    max_years: int | None = None,
    scenario: str | None = None,
    run: str | Path | None = None,
    keep_active_scenario: bool = False,
) -> dict[str, Any]:
    """整季闭环评估（PPO / 规则 / both）。

    - ``run``：写入该一期运行的 ``02_eval/``  
    - ``out_dir``：显式输出目录  
    - 二者皆空：新建 ``runs/run_*/02_eval``（须已有 ``model``）

    ``only='rule'`` 时可省略 model。
    ``keep_active_scenario``：沿用看板已激活的 uploads 情景。
    """
    from common.run_store import RunPaths, create_run, resolve_run
    from evaluation.run_eval import run_eval

    if only != "rule" and model is None:
        raise ValueError("only=ppo/both 时必须提供 model")
    run_root = None
    if run is not None:
        run_root = resolve_run(run, create=False)
        if run_root is None:
            raise FileNotFoundError(f"找不到一期 run: {run}")
        out_dir = RunPaths(run_root).ensure().eval
    elif out_dir is None:
        run_root = create_run(note="api.evaluate")
        out_dir = RunPaths(run_root).ensure().eval
    out = mkdir(out_dir)
    result = run_eval(
        model,
        out,
        only=only,
        max_years=max_years,
        scenario=scenario,
        keep_active_scenario=keep_active_scenario,
    )
    if run_root is not None:
        result["run"] = str(run_root)
        result["run_id"] = run_root.name
        RunPaths(run_root).update_meta(eval_dir=str(out))
    return result


def predict(
    model: str | Path,
    *,
    h: float,
    hmin: float,
    hmax: float,
    hp: float,
    hc: float,
    lamda: float = 0.2,
    ks: float | None = None,
    p_cumu7: float = 0.0,
    scenario: str | None = None,
) -> dict[str, Any]:
    """单日决策：掩码 → Hc 守卫 → 限幅。返回 ``decide_from_logits`` 结果字典。"""
    import numpy as np
    import torch

    from common.model_io import load_model_config, load_ppo_checkpoint
    from common.scenario import ks_factor
    from decision.infer import decide_from_logits

    set_active_scenario(scenario)
    agent, cfg, device = load_ppo_checkpoint(model)
    ks_v = float(ks_factor(h)) if ks is None else float(ks)
    state_vec = np.array([h, hmin, hmax, hp, hc, lamda, ks_v, p_cumu7], dtype=np.float32)
    x = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = agent.policy_net(x)
    dec = cfg.get("decision_defaults") or load_model_config().get("decision_defaults", {})
    return decide_from_logits(
        logits,
        x,
        action_space=agent.action_space,
        use_action_mask=bool(agent.use_action_mask),
        hmin=hmin,
        hmax=hmax,
        hp=hp,
        hc=hc,
        p_cumu7=p_cumu7,
        hc_guard=bool(dec.get("hc_guard", True)),
        physical_clip=bool(dec.get("physical_clip", True)),
        drying_rule=bool(dec.get("drying_rule", False)),
        low_water_refill=bool(dec.get("low_water_refill", True)),
    )


# ── 二期 HITL（转发 hitl.pipeline / compare_eval）──────────────────────────


def hitl_seed(
    work: str | Path | None = None,
    *,
    scenario: str | None = None,
    from_run: str | Path | None = None,
) -> dict[str, Any]:
    """新建或指定会话，整拷一期 ``01_train``+``02_eval``。见 ``hitl.pipeline.seed``。"""
    from hitl import seed

    return seed(work, scenario=scenario, from_run=from_run)


def hitl_review(
    *,
    model_path: str | Path | None = None,
    work: str | Path | None = None,
    max_years: int | None = None,
    years: list[str] | None = None,
    scenario: str | None = None,
    from_run: str | Path | None = None,
) -> Any:
    """冻结 PPO 生成审阅轨迹并锁定 original。返回 DataFrame。"""
    from hitl import review

    return review(
        model_path=model_path,
        work=work,
        max_years=max_years,
        years=years,
        scenario=scenario,
        from_run=from_run,
    )


def hitl_resync(
    *,
    model_path: str | Path | None = None,
    work: str | Path | None = None,
) -> Any:
    """审阅期重推 + 冻结 PPO 刷新后续建议。返回更新后的 DataFrame。"""
    from hitl import resync

    return resync(model_path=model_path, work=work)


def hitl_bc(
    review: Any = None,
    *,
    model_path: str | Path | None = None,
    work: str | Path | None = None,
    epochs: int | None = None,
    require_no_pending: bool = True,
) -> dict[str, Any]:
    """定稿 → ``04_bc_export/labels.json`` → BC → ``05_bc/ppo_hitl.pth``。"""
    from hitl import bc

    return bc(
        review,
        model_path=model_path,
        work=work,
        epochs=epochs,
        require_no_pending=require_no_pending,
    )


def hitl_compare(
    work: str | Path | None = None,
    *,
    scenario: str | None = None,
    max_years: int | None = None,
) -> dict[str, Any]:
    """BC 后三方对比 → ``06_compare`` + ``07_metrics/task_compare.json``。"""
    from hitl.compare_eval import compare_three

    return compare_three(work, scenario=scenario, max_years=max_years)


def hitl_pipeline(
    *,
    work: str | Path | None = None,
    model_path: str | Path | None = None,
    scenario: str | None = None,
    max_years: int | None = None,
    epochs: int | None = None,
    skip_bc: bool = False,
    from_run: str | Path | None = None,
) -> dict[str, Any]:
    """二期一键 seed→review→bc（不含人工确认与 compare）。"""
    from hitl import pipeline as hitl_pipe

    return hitl_pipe(
        work=work,
        model_path=model_path,
        scenario=scenario,
        max_years=max_years,
        epochs=epochs,
        skip_bc=skip_bc,
        from_run=from_run,
    )


def hitl_board(
    *,
    host: str = "127.0.0.1",
    port: int = 5050,
    debug: bool = False,
) -> dict[str, Any]:
    """启动双流程 Web 看板（阻塞至 Ctrl+C）。浏览器打开 ``http://host:port``。

    与 ``python run.py hitl board`` 相同：
    智能体训练（runs）+ 专家微调（hitl_sessions）。
    """
    from hitl.web.app import main as board_main

    url = f"http://{host}:{port}"
    print(f"HITL 看板 → {url}")
    board_main(host=host, port=port, debug=debug)
    return {"ok": True, "url": url}


# ── 统一分发（main.py / 外部脚本换方法名即可）────────────────────────────

METHODS: dict[str, Callable[..., Any]] = {
    "show_config": show_config,
    "pipeline": pipeline,
    "train": train,
    "evaluate": evaluate,
    "predict": predict,
    "hitl_seed": hitl_seed,
    "hitl_review": hitl_review,
    "hitl_resync": hitl_resync,
    "hitl_bc": hitl_bc,
    "hitl_compare": hitl_compare,
    "hitl_pipeline": hitl_pipeline,
    "hitl_board": hitl_board,
}


def list_methods() -> list[str]:
    """对外可调用方法名（与 ``main.py`` 的 METHOD 一致）。"""
    return list(METHODS.keys())


def dispatch(method: str, **kwargs: Any) -> Any:
    """按方法名调用统一对外接口。

    Parameters
    ----------
    method :
        ``list_methods()`` 中的名字，如 ``"pipeline"`` / ``"hitl_review"``。
    **kwargs :
        透传给对应函数（见各函数签名）。
    """
    if method not in METHODS:
        raise ValueError(
            f"未知 method={method!r}；可选: {', '.join(list_methods())}"
        )
    return METHODS[method](**kwargs)
