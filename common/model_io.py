"""模型配置 + PPO 权重加载（合并原 config_load）。

- load_model_config：读 configs/model_config.json
- normalize_policy_mode：policy_mode → use_action_mask
- load_ppo_checkpoint：pth + 旁路 *_config.json → (agent, cfg, device)

run.py / evaluation / ppo_model 推理统一走这里。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .action_mask import ACTION_SPACE
from .paths import ROOT, ensure_repo_path


def normalize_policy_mode(config: dict[str, Any]) -> dict[str, Any]:
    """统一 policy_mode → use_action_mask，并补齐 action_space。"""
    cfg = dict(config)
    mode = str(cfg.get("policy_mode", "masked")).strip().lower() or "masked"
    cfg["policy_mode"] = mode
    cfg["use_action_mask"] = mode not in ("unmasked", "raw", "off", "none", "false", "0")
    if "action_space" not in cfg and "action_space_mm" in cfg:
        cfg["action_space"] = list(cfg["action_space_mm"])
    if "action_space" not in cfg:
        cfg["action_space"] = list(ACTION_SPACE)
    return cfg


def load_model_config(path: str | Path | None = None) -> dict[str, Any]:
    """加载仓库级 model_config.json（策略协议，不含农艺情景）。"""
    cfg_path = Path(path) if path else ROOT / "configs" / "model_config.json"
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return normalize_policy_mode(data)


def sidecar_config_path(model_path: str | Path) -> Path:
    p = Path(model_path)
    return p.with_name(p.stem + "_config.json")


def load_ppo_checkpoint(
    model_path: str | Path,
    *,
    device: torch.device | None = None,
):
    """加载 (agent, config, device)。优先旁路 *_config.json，否则用仓库 model_config。"""
    ensure_repo_path()
    from train.PPO_RL_brain import PPO  # 延迟 import，避免 common 启动绑死 train/

    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    cfg_path = sidecar_config_path(model_path)
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        # 兼容曾将基础网络配置包在 BC 元数据内的 sidecar。
        if cfg.get("kind") == "bc_finetune" and isinstance(cfg.get("base_model_cfg"), dict):
            cfg = {**cfg["base_model_cfg"], **cfg}
    else:
        cfg = load_model_config()
        cfg["note"] = f"无旁路配置 {cfg_path.name}，使用仓库 model_config"

    cfg = normalize_policy_mode(cfg)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_space = list(cfg.get("action_space") or ACTION_SPACE)

    agent = PPO(
        n_states=int(cfg.get("n_states", 8)),
        n_actions=len(action_space),
        action_space=action_space,
        device=device,
        hidden_dim=int(cfg.get("hidden_dim", 512)),
        use_action_mask=bool(cfg.get("use_action_mask", True)),
        lr=float(cfg.get("learning_rate", 2e-4)),
        gamma=float(cfg.get("gamma", 0.98)),
        clip_param=float(cfg.get("clip_param", 0.15)),
        value_loss_coef=float(cfg.get("value_loss_coef", 0.7)),
        entropy_coef=float(cfg.get("entropy_coef", 0.03)),
        batch_size=int(cfg.get("batch_size", 1024)),
        ppo_epochs=int(cfg.get("ppo_epochs", 15)),
        lam=float(cfg.get("lam", 0.90)),
    )

    # 用文件句柄读，避免个别 torch/Windows 对中文路径字符串的兼容问题
    with open(model_path, "rb") as f:
        try:
            sd = torch.load(f, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(f, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    agent.policy_net.load_state_dict(sd)
    agent.policy_net.eval()
    return agent, cfg, device
