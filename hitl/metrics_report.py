"""论文用指标落盘：``07_metrics/absorption.json`` · ``effort.json`` · BC 看板摘要。"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd

from .config import action_to_idx
from .exec_layer import expert_action_filled, expert_action_float
from .session import SessionPaths


def _load_json_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path)


def _write(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _label_mm(row: pd.Series) -> float | None:
    """BC/吸收率标签 mm：优先 ``a_label_mm``（掩码合法），否则专家请求，再否则 ``a_model``。"""
    lab = row.get("a_label_mm")
    try:
        if lab is not None and not (isinstance(lab, float) and np.isnan(lab)):
            return float(lab)
    except (TypeError, ValueError):
        pass
    if expert_action_filled(row.get("a_expert")):
        return expert_action_float(row.get("a_expert"))
    am = row.get("a_model")
    try:
        if am is None or (isinstance(am, float) and np.isnan(am)):
            return None
        return float(am)
    except (TypeError, ValueError):
        return None


def _is_edit_vs_model(row: pd.Series) -> bool:
    lab = _label_mm(row)
    if lab is None:
        return False
    try:
        am = float(row.get("a_model", lab))
    except (TypeError, ValueError):
        return False
    return abs(lab - am) > 1e-6


def compute_absorption(
    sp: SessionPaths,
    *,
    hitl_model: Optional[Path] = None,
) -> dict[str, Any]:
    """专家改动吸收率：微调后策略是否学会「改动日」并保持「未改日」。

    - ``edit_reproduce_rate``：改动日上 HITL 策略 argmax = 专家标签的比例（学没学会改）
    - ``confirm_hold_rate_policy``：未改日上 HITL 仍输出原标签（跟模型）的比例（有没有忘掉）
    - 样本侧：改动占比等（不依赖权重）
    """
    orig_path = sp.original_traj / "all_years.json"
    # 优先用 BC 导出的一致轨迹（含 s_vec）；否则用 current review
    labels_df = _load_json_df(sp.labels) if sp.labels.exists() else pd.DataFrame()
    cur = labels_df if not labels_df.empty else _load_json_df(sp.review_json)
    if cur.empty and sp.review_csv.exists():
        cur = pd.read_csv(sp.review_csv)
    orig = _load_json_df(orig_path)
    if cur.empty:
        out = {"ok": False, "error": "无 current review / labels"}
        _write(sp.metrics / "absorption.json", out)
        return out

    if "confirmed" not in cur.columns:
        cur["confirmed"] = True  # labels 导出日多为定稿后样本
    if "include_in_bc" in cur.columns:
        conf = cur[cur["include_in_bc"].fillna(False).astype(bool)].copy()
    else:
        conf = cur[cur["confirmed"].fillna(False).astype(bool)].copy()
        if "editable" in conf.columns:
            conf = conf[conf["editable"].fillna(True).astype(bool)]

    n_conf = int(len(conf))
    n_edit = n_keep = 0
    for _, r in conf.iterrows():
        if _is_edit_vs_model(r):
            n_edit += 1
        else:
            n_keep += 1

    n_edit_vs_orig = None
    if not orig.empty and n_conf and "date" in conf.columns and "date" in orig.columns:
        omap = {
            str(r["date"]): float(r.get("a_model", r.get("a_expert", 0)) or 0)
            for _, r in orig.iterrows()
            if pd.notna(r.get("a_model", r.get("a_expert")))
        }
        n_edit_vs_orig = 0
        for _, r in conf.iterrows():
            lab = _label_mm(r)
            if lab is None:
                continue
            d = str(r["date"])
            if d in omap and abs(lab - omap[d]) > 1e-6:
                n_edit_vs_orig += 1

    reproduce = None
    keep_rate_policy = None
    policy_err = None
    by_year: dict[str, Any] = {}
    tot_edit = tot_keep = match_edit = match_keep = 0
    _forced_stats: dict[str, Any] = {
        "n_forced_mask_only": 0,
        "forced_mask_match_rate": None,
        "note_forced_mask": "掩码仅剩一档的日子不计入 edit_reproduce / confirm_hold 主率",
    }
    model_path = Path(hitl_model) if hitl_model else sp.hitl_model
    # labels.json 用 s_vec；审阅表/resync 用 state_vec
    sv_col = "s_vec" if "s_vec" in conf.columns else ("state_vec" if "state_vec" in conf.columns else None)
    has_sv = bool(sv_col) and n_conf > 0
    if model_path.exists() and has_sv:
        try:
            from common.action_mask import ACTION_SPACE, mask_policy_logits
            from common.model_io import load_ppo_checkpoint
            import torch

            from common.action_mask import valid_action_mask

            agent, _, device = load_ppo_checkpoint(model_path)
            agent.policy_net.eval()
            action_space = list(getattr(agent, "action_space", None) or ACTION_SPACE)
            by_year_acc: dict[str, dict[str, int]] = {}
            n_forced_mask = 0
            match_forced = 0
            for _, r in conf.iterrows():
                lab = _label_mm(r)
                if lab is None:
                    continue
                sv = r.get(sv_col)
                if sv is None:
                    continue
                if not isinstance(sv, (list, tuple)) and not hasattr(sv, "__len__"):
                    continue
                try:
                    arr = np.asarray(sv, dtype=np.float32).reshape(-1)
                    if arr.size == 0 or np.isnan(arr).any():
                        continue
                except Exception:
                    continue
                x = torch.FloatTensor(arr).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _ = agent.policy_net(x)
                    # 与 PPO/BC 一致：掩码后再 argmax
                    masked = mask_policy_logits(logits, x, action_space)
                    n_valid = int(valid_action_mask(x, action_space).sum().item())
                pred_idx = int(masked.argmax(dim=-1).item())
                try:
                    target_idx = action_to_idx(lab)
                except Exception:
                    continue
                hit = pred_idx == target_idx
                # 掩码仅剩一档：匹配来自约束，不计入经验吸收率主指标
                if n_valid <= 1:
                    n_forced_mask += 1
                    if hit:
                        match_forced += 1
                    continue
                is_edit = _is_edit_vs_model(r)
                if is_edit:
                    tot_edit += 1
                    if hit:
                        match_edit += 1
                else:
                    tot_keep += 1
                    if hit:
                        match_keep += 1

                y = str(r.get("year") or "").strip()
                if not y and "date" in r.index:
                    y = str(r["date"]).replace("-", "/").split("/")[0]
                if y:
                    bucket = by_year_acc.setdefault(
                        y, {"tot_edit": 0, "match_edit": 0, "tot_keep": 0, "match_keep": 0}
                    )
                    if is_edit:
                        bucket["tot_edit"] += 1
                        if hit:
                            bucket["match_edit"] += 1
                    else:
                        bucket["tot_keep"] += 1
                        if hit:
                            bucket["match_keep"] += 1

            reproduce = (match_edit / tot_edit) if tot_edit else None
            keep_rate_policy = (match_keep / tot_keep) if tot_keep else None
            if tot_edit == 0 and tot_keep == 0 and n_forced_mask == 0:
                policy_err = "有 s_vec/state_vec 但无一有效样本可评"
            by_year = {
                y: {
                    "edit_reproduce_rate": (
                        (b["match_edit"] / b["tot_edit"]) if b["tot_edit"] else None
                    ),
                    "confirm_hold_rate_policy": (
                        (b["match_keep"] / b["tot_keep"]) if b["tot_keep"] else None
                    ),
                    "n_edited": b["tot_edit"],
                    "n_keep": b["tot_keep"],
                }
                for y, b in sorted(by_year_acc.items())
            }
            # 暂存强制掩码日统计，写入 out 时合并
            _forced_stats = {
                "n_forced_mask_only": n_forced_mask,
                "forced_mask_match_rate": (
                    (match_forced / n_forced_mask) if n_forced_mask else None
                ),
                "note_forced_mask": (
                    "掩码仅剩一档的日子不计入 edit_reproduce / confirm_hold 主率"
                ),
            }
        except Exception as e:
            reproduce = None
            keep_rate_policy = None
            policy_err = str(e)
            by_year = {}
    elif not model_path.exists():
        policy_err = "无 ppo_hitl.pth，仅样本统计"
    elif not has_sv:
        policy_err = "无 s_vec/state_vec（请先 BC 导出 labels.json）"

    out = {
        "ok": True,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "n_days": int(len(cur)),
        "n_confirmed": n_conf,
        "n_edited_among_confirmed": n_edit,
        "n_keep_among_confirmed": n_keep,
        "confirm_keep_rate": (n_keep / n_conf) if n_conf else None,
        "edit_share_among_confirmed": (n_edit / n_conf) if n_conf else None,
        "n_edit_vs_original_traj": n_edit_vs_orig,
        "edit_reproduce_rate": reproduce,
        "confirm_hold_rate_policy": keep_rate_policy,
        "n_eval_edit": tot_edit,
        "n_eval_keep": tot_keep,
        "n_match_edit": match_edit,
        "n_match_keep": match_keep,
        "n_forced_mask_only": _forced_stats.get("n_forced_mask_only"),
        "forced_mask_match_rate": _forced_stats.get("forced_mask_match_rate"),
        "by_year": by_year,
        "hitl_model": str(model_path) if model_path.exists() else None,
        "source": "labels.json" if not labels_df.empty else "review",
        "note": policy_err,
        "defs": {
            "edit_reproduce_rate": (
                "改动日且掩码可选档>1：掩码后 argmax == 专家请求标签（强制单档日另计）"
            ),
            "confirm_hold_rate_policy": (
                "未改日且掩码可选档>1：掩码后 argmax == 标签（强制单档日另计）"
            ),
            "n_forced_mask_only": "掩码仅剩一档的确认日数（匹配不计入主吸收率）",
            "include_in_bc": "定稿可编辑日进 BC；晒田/落干默认不进；改动日权重 w_edit，其余权重 1",
            "by_year": "按审阅年拆分的吸收率；留出验证年无专家改动时改动复现为 null",
        },
    }
    _write(sp.metrics / "absorption.json", out)
    # 概率配对图是懒缓存；BC 后权重已变，必须作废以免前端读到旧 absorption_probs.json
    stale = sp.metrics / "absorption_probs.json"
    if stale.exists():
        try:
            stale.unlink()
        except OSError:
            pass
    return out


def build_bc_dashboard(sp: SessionPaths) -> dict[str, Any]:
    """看板用：学什么 / 训成没 / 学完好不好。"""
    hitl = {}
    p = sp.bc / "hitl_result.json"
    if p.exists():
        try:
            hitl = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            hitl = {}
    abs_ = {}
    ap = sp.metrics / "absorption.json"
    if ap.exists():
        try:
            abs_ = json.loads(ap.read_text(encoding="utf-8"))
        except Exception:
            abs_ = {}
    task = {}
    tp = sp.metrics / "task_compare.json"
    if tp.exists():
        try:
            task = json.loads(tp.read_text(encoding="utf-8"))
        except Exception:
            task = {}

    bc_hist = hitl.get("bc") or {}
    ce = bc_hist.get("ce") or []
    val_ce = bc_hist.get("val_ce") or []
    trained_ok = bool(sp.hitl_model.exists()) and bool(hitl)
    n_include = hitl.get("n_include_bc")
    n_labels = hitl.get("n_labels")
    n_edited = hitl.get("n_edited")

    means = {}
    arms = (task.get("arms") or {}) if isinstance(task, dict) else {}
    for arm in ("rule", "original_ppo", "hitl_ppo"):
        m = (arms.get(arm) or {}).get("means") or {}
        if m:
            means[arm] = m

    def _delta_pct(hitl_v, rule_v, higher_better: bool) -> float | None:
        try:
            hv, rv = float(hitl_v), float(rule_v)
            if abs(rv) < 1e-12:
                return None
            raw = (hv - rv) / abs(rv) * 100.0
            # 灌/排/管理水量：越少越好 → 相对基线「改善%」取反号展示在看板文案里由前端处理
            return raw if higher_better else -raw
        except Exception:
            return None

    quality = None
    if means.get("hitl_ppo") and means.get("rule") and means.get("original_ppo"):
        h, r, o = means["hitl_ppo"], means["rule"], means["original_ppo"]
        quality = {
            "vs_rule": {
                "rain_utilization_delta_pct": _delta_pct(
                    h.get("rain_utilization"), r.get("rain_utilization"), True
                ),
                "managed_water_improve_pct": _delta_pct(
                    h.get("managed_water"), r.get("managed_water"), False
                ),
                "yield_factor_delta_pct": _delta_pct(
                    h.get("yield_factor"), r.get("yield_factor"), True
                ),
            },
            "vs_original_ppo": {
                "rain_utilization_delta_pct": _delta_pct(
                    h.get("rain_utilization"), o.get("rain_utilization"), True
                ),
                "managed_water_improve_pct": _delta_pct(
                    h.get("managed_water"), o.get("managed_water"), False
                ),
                "yield_factor_delta_pct": _delta_pct(
                    h.get("yield_factor"), o.get("yield_factor"), True
                ),
            },
        }

    return {
        "has_model": bool(sp.hitl_model.exists()),
        "trained_ok": trained_ok,
        "learn": {
            "n_labels_rows": n_labels,
            "n_include_bc": n_include,
            "n_edited": n_edited,
            "n_keep": (int(n_include) - int(n_edited))
            if n_include is not None and n_edited is not None
            else None,
            "explain": (
                "labels.json 含整季重放行；进 BC 的是定稿可编辑日（排除晒田/落干）。"
                "其中「改动日」权重 w_edit，未改确认日权重 1——不是只训 20 天。"
            ),
        },
        "train": {
            "best_epoch": bc_hist.get("best_epoch"),
            "ce_start": float(ce[0]) if ce else None,
            "ce_end": float(ce[-1]) if ce else None,
            "val_ce_start": float(val_ce[0]) if val_ce else None,
            "val_ce_end": float(val_ce[-1]) if val_ce else None,
            "ce_dropped": (
                float(ce[0]) - float(ce[-1]) if len(ce) >= 2 else None
            ),
        },
        "absorb": {
            "edit_reproduce_rate": abs_.get("edit_reproduce_rate"),
            "confirm_hold_rate_policy": abs_.get("confirm_hold_rate_policy"),
            "edit_share_among_confirmed": abs_.get("edit_share_among_confirmed"),
            "n_edited_among_confirmed": abs_.get("n_edited_among_confirmed"),
            "n_keep_among_confirmed": abs_.get("n_keep_among_confirmed"),
            "by_year": abs_.get("by_year") or {},
            "note": abs_.get("note"),
            "ok": abs_.get("ok"),
        },
        "quality": quality,
        "quality_scope": "mean",
        "paths": {
            "labels": str(sp.labels) if sp.labels.exists() else None,
            "hitl_model": str(sp.hitl_model) if sp.hitl_model.exists() else None,
            "hitl_result": str(p) if p.exists() else None,
            "absorption": str(ap) if ap.exists() else None,
        },
    }


def load_effort(sp: SessionPaths) -> dict[str, Any]:
    p = sp.metrics / "effort.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"stages": [], "total_seconds": 0.0}


def record_effort(sp: SessionPaths, stage: str, seconds: float, **extra: Any) -> dict[str, Any]:
    data = load_effort(sp)
    entry = {
        "stage": stage,
        "seconds": round(float(seconds), 3),
        "at": datetime.now().isoformat(timespec="seconds"),
        **extra,
    }
    data.setdefault("stages", []).append(entry)
    data["total_seconds"] = round(
        sum(float(s.get("seconds", 0)) for s in data["stages"]), 3
    )
    data["updated_at"] = entry["at"]
    _write(sp.metrics / "effort.json", data)
    return data


@contextmanager
def timed_stage(sp: SessionPaths, stage: str, **extra: Any) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record_effort(sp, stage, time.perf_counter() - t0, **extra)
