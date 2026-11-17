"""行为克隆微调：CE + 旧策略 KL；不调用 PPO.update。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from common.action_mask import ACTION_SPACE, mask_policy_logits

from .config import BC_DEFAULT, BCConfig, describe_bc_rationale
from .recompute import to_bc_arrays

PathLike = Union[str, Path]


def _masked_log_probs(net, bx: torch.Tensor, action_space=None) -> tuple[torch.Tensor, torch.Tensor]:
    """与 PPO 推理一致：先动作掩码再 log_softmax。"""
    action_space = action_space or ACTION_SPACE
    logits, _ = net(bx)
    masked = mask_policy_logits(logits, bx, action_space)
    return masked, F.log_softmax(masked, dim=1)


def _set_trainable(net, cfg: BCConfig) -> None:
    for p in net.parameters():
        p.requires_grad = True
    if cfg.freeze_value_head:
        for p in net.value_head.parameters():
            p.requires_grad = False
    if cfg.train_policy_head_only:
        for p in net.features.parameters():
            p.requires_grad = False
        for p in net.value_head.parameters():
            p.requires_grad = False
        for p in net.policy_head.parameters():
            p.requires_grad = True


def _split_indices(n: int, val_ratio: float, min_val: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(min_val, int(round(n * val_ratio)))
    n_val = min(n_val, max(1, n // 5)) if n >= 10 else max(1, n // 5)
    if n < 10:
        return idx, idx
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if len(train_idx) == 0:
        train_idx = idx
    return train_idx, val_idx


def finetune_bc(
    agent,
    labels: pd.DataFrame,
    *,
    cfg: Optional[BCConfig] = None,
    save_path: Optional[PathLike] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    w_edit: Optional[float] = None,
    freeze_value_head: Optional[bool] = None,
    kl_beta: Optional[float] = None,
) -> dict:
    cfg = BCConfig(**{**BC_DEFAULT.to_dict(), **(cfg.to_dict() if cfg else {})})
    if epochs is not None:
        cfg.epochs = epochs
    if batch_size is not None:
        cfg.batch_size = batch_size
    if lr is not None:
        cfg.lr = lr
    if w_edit is not None:
        cfg.w_edit = w_edit
    if freeze_value_head is not None:
        cfg.freeze_value_head = freeze_value_head
    if kl_beta is not None:
        cfg.kl_beta = kl_beta

    if cfg.label_smoothing != 0:
        raise ValueError("掩码 BC 当前仅支持 label_smoothing=0")
    print(f"[BC] 实际配置: {cfg.to_dict()}")
    device = agent.device
    net = agent.policy_net
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    states, actions, flags = to_bc_arrays(labels)
    if not np.isfinite(states).all():
        raise ValueError("BC 状态包含非有限数值")
    sample_w = np.where(flags == "edited", float(cfg.w_edit), 1.0).astype(np.float32)
    x_all = torch.FloatTensor(states).to(device)
    y_all = torch.LongTensor(actions).to(device)
    w_all = torch.FloatTensor(sample_w).to(device)

    action_space = list(getattr(agent, "action_space", None) or ACTION_SPACE)
    # 训练前：全部目标须在合法掩码内
    with torch.no_grad():
        from common.action_mask import valid_action_mask

        mask = valid_action_mask(x_all, action_space)
        for i in range(y_all.size(0)):
            yi = int(y_all[i].item())
            if yi < 0 or yi >= mask.size(1) or not bool(mask[i, yi].item()):
                raise ValueError(
                    f"BC 目标动作不在掩码内（样本 {i}, a_idx={yi}）。"
                    "请回到审阅改合法请求档后再导出。"
                )
        _, old_log_probs = _masked_log_probs(net, x_all, action_space)

    included = labels.loc[labels["include_in_bc"].astype(bool)].reset_index(drop=True)
    available_val_years = (
        sorted(included["year"].astype(str).unique().tolist())
        if "year" in included.columns
        else []
    )
    effective_val_years = list(map(str, cfg.val_years))
    split_unit = "row_internal_only_single_season"
    if not effective_val_years and len(available_val_years) >= 2:
        # 按整季留出，避免相邻日期同时落入训练集与验证集；正式四季审阅默认留最新一季。
        effective_val_years = [available_val_years[-1]]
    if effective_val_years:
        if "year" not in included.columns:
            raise ValueError("按季拆分 BC 需要 labels.year")
        is_val = included["year"].astype(str).isin(set(effective_val_years)).to_numpy()
        train_idx, val_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
        if not len(train_idx) or not len(val_idx):
            raise ValueError(
                "按年拆分 BC 训练/验证集为空；"
                f"available={available_val_years}, requested={effective_val_years}"
            )
        cfg.val_years = effective_val_years
        split_unit = "season"
    else:
        train_idx, val_idx = _split_indices(len(states), cfg.val_ratio, cfg.min_val_size, cfg.seed)
    train_idx_t = torch.as_tensor(train_idx, device=device)
    val_idx_t = torch.as_tensor(val_idx, device=device)
    _set_trainable(net, cfg)
    net.train()
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.1)
        if cfg.scheduler == "cosine"
        else None
    )

    def _ce_kl(bx, by, bw, bold):
        # 与部署/评估一致：非法档掩码后再算 CE / KL（默认 label_smoothing=0）
        masked, log_probs = _masked_log_probs(net, bx, action_space)
        ce = (
            F.cross_entropy(
                masked, by, label_smoothing=float(cfg.label_smoothing), reduction="none"
            )
            * bw
        ).mean()
        kl = (
            (bold.exp() * (bold - log_probs)).sum(dim=1).mean()
            if cfg.kl_beta > 0
            else masked.new_zeros(())
        )
        loss = ce + cfg.kl_beta * kl
        if not torch.isfinite(loss):
            raise RuntimeError("BC 损失非有限（NaN/Inf），已中止训练")
        return ce, kl, loss

    ds = TensorDataset(
        x_all[train_idx_t], y_all[train_idx_t], w_all[train_idx_t], old_log_probs[train_idx_t],
    )
    loader = DataLoader(ds, batch_size=min(cfg.batch_size, max(1, len(ds))), shuffle=True)
    history: dict = {
        "ce": [], "kl": [], "loss": [], "val_ce": [], "lr": [],
        "best_epoch": 0,
        "config": cfg.to_dict(),
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "n_samples": int(len(states)),
        "split_unit": split_unit,
        "validation_years": effective_val_years,
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
    }
    best_val = float("inf")
    best_state = None
    patience_left = cfg.early_stop_patience

    for ep in range(cfg.epochs):
        net.train()
        ep_ce = ep_kl = ep_loss = 0.0
        n = 0
        for bx, by, bw, bold in loader:
            ce, kl, loss = _ce_kl(bx, by, bw, bold)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm, error_if_nonfinite=True)
            opt.step()
            bs = bx.size(0)
            ep_ce += float(ce.item()) * bs
            ep_kl += float(kl.item()) * bs
            ep_loss += float(loss.item()) * bs
            n += bs
        if sched is not None:
            sched.step()

        net.eval()
        with torch.no_grad():
            v_ce, _, _ = _ce_kl(
                x_all[val_idx_t], y_all[val_idx_t], w_all[val_idx_t], old_log_probs[val_idx_t],
            )
        train_ce = ep_ce / max(n, 1)
        val_ce = float(v_ce.item())
        history["ce"].append(train_ce)
        history["kl"].append(ep_kl / max(n, 1))
        history["loss"].append(ep_loss / max(n, 1))
        history["val_ce"].append(val_ce)
        history["lr"].append(float(opt.param_groups[0]["lr"]))
        print(
            f"[BC] epoch {ep+1}/{cfg.epochs}  "
            f"train_CE={train_ce:.4f}  val_CE={val_ce:.4f}  "
            f"kl={history['kl'][-1]:.4f}"
        )
        if val_ce < best_val - cfg.min_delta:
            best_val = val_ce
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            history["best_epoch"] = ep + 1
            patience_left = cfg.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[BC] 早停于 epoch {ep+1}（best={history['best_epoch']}）")
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    for p in net.parameters():
        p.requires_grad = True

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Windows 中文路径：经文件句柄写（与一期 save_model 一致）
        with open(save_path, "wb") as f:
            torch.save(net.state_dict(), f)
        print(f"HITL 策略已保存: {save_path}")

    net.eval()
    return history
