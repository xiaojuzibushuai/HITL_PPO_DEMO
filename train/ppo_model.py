"""PPO 训练与结果落盘（一期正式版）。

本文件只负责：
  - ``train_ppo``：与 env 交互 + PPO update；只写 best / last
  - ``save_model`` / ``save_training_results``：权重与曲线

推荐经 ``api.train`` / ``run.py train`` 调用（会写入 ``output/runs/run_*/01_train``）。
训练逻辑：掩码采样 → step → 外层动作罚 → 定期 update → 近窗均回报刷 best。
"""

import json
import pandas as pd
import torch
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

# Windows 下尽量用系统中文字体，避免曲线图缺字警告（无字体则静默回退）
try:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.action_mask import ACTION_SPACE
from common.model_io import normalize_policy_mode
from common.paths import ROOT, mkdir

from .PPO_Rice_env import Irrigation
from .PPO_RL_brain import PPO

# 默认超参（与 ACTION_SPACE / configs/model_config.json 对齐）
DEFAULT_CONFIG = {
    'n_states': 8,
    'n_actions': len(ACTION_SPACE),
    'action_space': list(ACTION_SPACE),
    'hidden_dim': 512,
    'policy_mode': 'masked',
    'learning_rate': 2e-4,
    'gamma': 0.98,
    'clip_param': 0.15,
    'value_loss_coef': 0.7,
    'entropy_coef': 0.03,
    'batch_size': 1024,
    'ppo_epochs': 15,
    'lam': 0.90,
}


def save_model(
    model: PPO,
    model_config: Dict,
    model_path: str
) -> Tuple[str, str]:
    """
    保存PPO模型权重和配置
    
    Args:
        model: PPO模型实例
        model_config: 模型配置字典，包含：
            {
                'n_states': int,
                'n_actions': int,
                'action_space': list,
                'learning_rate': float,
                'gamma': float,
                'clip_param': float,
                'value_loss_coef': float,
                'entropy_coef': float,
                'batch_size': int,
                'ppo_epochs': int,
                'lam': float,
                'hidden_dim': int,
                'training_episodes': int,
                'training_time': float,
                'final_mean_return': float,
                'final_mean_actions': float
            }
        model_path: 模型保存路径
            示例: /path/to/ppo_irrigation_best.pth
    
    Returns:
        Tuple: (model_path, config_path)
    
    会自动生成：
    - /path/to/ppo_irrigation_best.pth          （模型权重）
    - /path/to/ppo_irrigation_best_config.json  （模型配置）
    """
    mp = Path(model_path)
    if not mp.is_absolute():
        mp = ROOT / mp
    model_path = os.path.abspath(str(mp))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    base = os.path.splitext(model_path)[0]
    config_path = base + "_config.json"

    # torch 2.0.x 在 Windows 中文路径下对「路径字符串」torch.save 会误报
    # Parent directory does not exist；写到已打开的二进制文件即可绕过。
    state = model.policy_net.state_dict()
    with open(model_path, "wb") as f:
        torch.save(state, f)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=4)

    return model_path, config_path


# ==================== 核心方法1：训练 ====================

def train_ppo(
    total_episodes: int = 1500,
    trajectory_num: int = 10,
    save_dir: str | Path | None = None,
    model_name: str = "ppo_rice",
    config: Optional[Dict] = None,
    verbose: bool = True,
    debug_mode: bool = False,
    progress_callback: Optional[Callable] = None,
) -> Dict:
    """训练 PPO，并落盘两种权重（定稿，不再存中间 ckpt）。

    存盘策略（本场景推荐）：
      - ``{name}_best.pth``：近窗口（默认约 100 轮）回报均值的**历史最优**；
        训练中每次刷新立刻写盘；**评估/上线只用 best**。
      - ``{name}_last.pth``：设定轮数全部跑完后的**末轮**权重，便于对照末态是否变差。

    不按「每 N 百轮存一份」：灌溉季回报方差大，中间切片很少被真正拿来部署，徒增文件。

    轮数建议（当前 data 约 80+ 年 × 约 112 日/季）：
      1000～2000 通常够用；看近 100 轮回报是否走平再决定是否加到 3000。

    Args:
        total_episodes: 总轮数（默认 1500）
        trajectory_num: 每多少局做一次 PPO update（默认 10）
        save_dir / model_name: 权重目录与文件名前缀
        config / verbose / debug_mode / progress_callback: 可选
    """
    import time
    
    # 默认配置
    default_config = {
        'learning_rate': 2e-4,
        'gamma': 0.98,
        'clip_param': 0.15,
        'value_loss_coef': 0.7,
        'entropy_coef': 0.03,
        'batch_size': 1024,
        'ppo_epochs': 15,
        'lam': 0.90,
        'hidden_dim': 512,
        'max_grad_norm': 0.5,
    }
    
    if config is None:
        config = default_config.copy()
    else:
        config = {**default_config, **config}
    config = normalize_policy_mode({**DEFAULT_CONFIG, **config})
    
    # 只存两种权重：best（近窗口均回报历史最优）/ last（末轮）
    # 正式入口 api.train 会传入 runs/.../01_train；裸调时落盘到临时 scratch
    if not save_dir:
        save_dir = ROOT / "output" / "_scratch" / "train"
    save_dir = mkdir(save_dir)
    best_path = os.path.join(save_dir, f"{model_name}_best.pth")
    last_path = os.path.join(save_dir, f"{model_name}_last.pth")
    model_path = best_path

    # 训练 workdir 里 forecast/actual 已是 scenario 训练表拷贝；
    # 字段名 *_policy 是历史 API，内容与 state_space 相同（各一份独立 DataFrame）。
    env = Irrigation()
    state_space = env.state_space_policy
    actual_state_space = env.actual_state_space_policy

    action_space = list(ACTION_SPACE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPO(
        n_states=8,
        n_actions=len(action_space),
        action_space=action_space,
        lr=config["learning_rate"],
        gamma=config["gamma"],
        clip_param=config["clip_param"],
        value_loss_coef=config["value_loss_coef"],
        entropy_coef=config["entropy_coef"],
        batch_size=config["batch_size"],
        ppo_epochs=config["ppo_epochs"],
        lam=config["lam"],
        device=device,
        hidden_dim=config["hidden_dim"],
        use_action_mask=config.get("use_action_mask", True),
    )

    return_list = []
    mean_return_list = []
    action_count_log = []
    loss_log = []

    # 全局最佳：近 window 轮回报均值最高时的权重（window=min(100, 已跑轮数)）
    best_mean_return = float("-inf")
    best_model_state = None
    best_episode = -1
    best_window = min(100, max(10, total_episodes // 10))  # 短训也追踪；长训用 100

    start_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print("开始PPO训练")
        print(f"{'='*60}")
        print(f"总轮数: {total_episodes}  (建议正式训 1000～2000)")
        print(f"动作空间: {len(action_space)} 档 | policy_mode={config.get('policy_mode')} mask={config.get('use_action_mask')}")
        print(f"存盘: BEST=近{best_window}轮均值最优 → {os.path.basename(best_path)}")
        print(f"      LAST=训完末轮           → {os.path.basename(last_path)}")
        print(f"      评估/上线请用 BEST")
        print(f"学习率: {config['learning_rate']} | 设备: {device}")
        print(f"{'='*60}\n")
    
    last_update_losses: Optional[Dict] = None

    # 训练循环
    for episode in range(total_episodes):
        state, actual_state_pd = env.reset(state_space)
        ep_year = str(state.date).split("/")[0]
        raw_state = np.array(env.process_series(state)).astype(np.float32)
        done = False
        episode_return = 0.0
        step_count = 0
        episode_experience = []
        nonzero_action_count = 0
        irrig_n = drain_n = 0

        while not done and step_count < 200:
            action, action_idx, log_prob, action_probs = agent.choose_action(
                raw_state,
                episode=episode,
                total_episodes=total_episodes,
            )

            if action > 0:
                irrig_n += 1
                nonzero_action_count += 1
            elif action < 0:
                drain_n += 1
                nonzero_action_count += 1

            next_state, reward, done, actual_action = env.step(
                actual_state_pd, actual_state_space, state_space, state, action
            )

            step_penalty = 0.5 if action != 0 else 0
            base_reward = reward[0]
            final_penalty = 5 * nonzero_action_count if done else 0
            total_reward = base_reward - step_penalty - final_penalty
            scaled_reward = total_reward * 0.1

            next_raw_state = np.array(env.process_series(next_state)).astype(np.float32)

            experience = {
                "state": raw_state,
                "action": action_idx,
                "reward": scaled_reward,
                "next_state": next_raw_state,
                "done": done,
                "log_prob": log_prob.item(),
                "action_agent": action,
                "nonzero_action": 1 if action != 0 else 0,
            }

            if debug_mode:
                experience.update({
                    "state.h_before": state.h_before,
                    "actual_action": actual_action,
                    "r0": reward[1],
                    "r1": reward[2],
                    "r2": reward[3],
                    "r4": reward[4],
                    "next_state.h_before": next_state.h_before,
                    "step": step_count,
                    "episode": episode,
                    "date": state.date,
                    "action_probs": action_probs.tolist(),
                })

            episode_experience.append(experience)
            raw_state = next_raw_state
            state = next_state
            step_count += 1
            episode_return += scaled_reward

        agent.store_trajectory(episode_experience)

        did_update = False
        if (episode + 1) % trajectory_num == 0:
            loss_info = agent.update(episode)
            if loss_info:
                loss_log.extend(loss_info)
                # 本轮 update 末批均值，便于看曲线
                last_update_losses = {
                    "policy": float(np.mean([x["policy_loss"] for x in loss_info])),
                    "value": float(np.mean([x["value_loss"] for x in loss_info])),
                    "entropy": float(np.mean([x["entropy_loss"] for x in loss_info])),
                }
                did_update = True

        return_list.append(episode_return)
        mean_return = np.mean(return_list[-10:]) if len(return_list) >= 10 else np.mean(return_list)
        mean_w = (
            float(np.mean(return_list[-best_window:]))
            if len(return_list) >= best_window
            else float(mean_return)
        )
        mean_return_list.append(mean_return)
        action_count_log.append(nonzero_action_count)

        if progress_callback is not None:
            try:
                progress_callback({
                    "episode": episode + 1,
                    "total_episodes": total_episodes,
                    "return": float(episode_return),
                    "mean_return": float(mean_return),
                    "actions": int(nonzero_action_count),
                    "best_mean_return": (
                        float(best_mean_return) if best_mean_return != float("-inf") else None
                    ),
                })
            except Exception:
                pass

        # 全局最佳：近 best_window 轮回报均值刷新时立刻落盘
        if len(return_list) >= best_window:
            score = float(np.mean(return_list[-best_window:]))
            if score > best_mean_return:
                best_mean_return = score
                best_model_state = {
                    k: v.detach().clone() for k, v in agent.policy_net.state_dict().items()
                }
                best_episode = episode + 1
                snap_cfg = normalize_policy_mode({
                    "n_states": 8,
                    "n_actions": len(action_space),
                    "action_space": action_space,
                    "policy_mode": config.get("policy_mode", "masked"),
                    "learning_rate": config["learning_rate"],
                    "gamma": config["gamma"],
                    "clip_param": config["clip_param"],
                    "value_loss_coef": config["value_loss_coef"],
                    "entropy_coef": config["entropy_coef"],
                    "batch_size": config["batch_size"],
                    "ppo_epochs": config["ppo_epochs"],
                    "lam": config["lam"],
                    "hidden_dim": config["hidden_dim"],
                    "training_episodes": episode + 1,
                    "best_mean_return": best_mean_return,
                    "best_episode": best_episode,
                    "best_window": best_window,
                    "checkpoint_role": "best",
                    "note": f"训练中刷新的全局最佳（ep {best_episode}）",
                })
                save_model(agent, snap_cfg, best_path)
                if verbose:
                    print(
                        f"  ★ Best ep={best_episode}  "
                        f"近{best_window}轮均值={best_mean_return:.3f} → 写入 best"
                    )

        if verbose:
            best_s = (
                f"{best_mean_return:.2f}@ep{best_episode}"
                if best_episode > 0
                else "-"
            )
            line = (
                f"[{episode+1:4d}/{total_episodes}] "
                f"R={episode_return:6.2f} mean10={mean_return:6.2f} mean{best_window}={mean_w:6.2f} "
                f"days={step_count:3d} year={ep_year} "
                f"irr={irrig_n:2d} drn={drain_n:2d} nz={nonzero_action_count:2d} "
                f"T={agent.temperature:.2f} best={best_s}"
            )
            if did_update and last_update_losses:
                line += (
                    f" | Lπ={last_update_losses['policy']:.3f} "
                    f"Lv={last_update_losses['value']:.3f} "
                    f"H={last_update_losses['entropy']:.3f}"
                )
            print(line)

            if (episode + 1) % 100 == 0:
                recent = return_list[-100:]
                recent_a = action_count_log[-100:]
                improvement = (
                    np.mean(recent) - np.mean(return_list[-200:-100]) if len(return_list) >= 200 else 0
                )
                elapsed = time.time() - start_time
                eta = elapsed / (episode + 1) * (total_episodes - episode - 1)
                print(f"\n{'='*72}")
                print(f"  Summary @{episode+1}  已用 {elapsed/60:.1f}min  ETA {eta/60:.1f}min")
                print(
                    f"  近100轮 R: 均={np.mean(recent):.3f} "
                    f"max={np.max(recent):.3f} min={np.min(recent):.3f}  "
                    f"Δvs前100={improvement:+.3f}"
                )
                print(
                    f"  近100轮动作数均={np.mean(recent_a):.1f}  "
                    f"best={best_s}  window={best_window}"
                )
                print(f"{'='*72}\n")
    end_time = time.time()
    training_time = end_time - start_time
    final_mean_return = (
        np.mean(return_list[-100:]) if len(return_list) >= 100 else np.mean(return_list)
    )
    final_mean_actions = (
        np.mean(action_count_log[-100:]) if len(action_count_log) >= 100 else np.mean(action_count_log)
    )

    # 1) 先存末轮（训练结束时的网络）
    last_cfg = normalize_policy_mode({
        "n_states": 8,
        "n_actions": len(action_space),
        "action_space": action_space,
        "policy_mode": config.get("policy_mode", "masked"),
        "learning_rate": config["learning_rate"],
        "gamma": config["gamma"],
        "clip_param": config["clip_param"],
        "value_loss_coef": config["value_loss_coef"],
        "entropy_coef": config["entropy_coef"],
        "batch_size": config["batch_size"],
        "ppo_epochs": config["ppo_epochs"],
        "lam": config["lam"],
        "hidden_dim": config["hidden_dim"],
        "training_episodes": total_episodes,
        "training_time": training_time,
        "final_mean_return": float(final_mean_return),
        "final_mean_actions": float(final_mean_actions),
        "checkpoint_role": "last",
        # last = 从第 1 轮一直训到第 N 轮后的「终点网络」（完整跑完）；不是中间切片拼盘
        # best = 过程中近窗均回报历史最优（评估/上线用）；终点未必更好
        "note": "完整训完所有 episodes 后的终点权重（从第1轮连续训到末轮）；评估请用 best",
    })
    last_path, last_cfg_path = save_model(agent, last_cfg, last_path)

    # 2) 再写全局最佳（若有）；否则 best = last
    if best_model_state is not None:
        agent.policy_net.load_state_dict(best_model_state)
        best_note = f"全局最佳：近{best_window}轮均值={best_mean_return:.4f}（刷新于 ep {best_episode}）"
    else:
        best_mean_return = float(final_mean_return)
        best_episode = total_episodes
        best_note = "轮数不足窗口，best 与 last 相同"

    best_cfg = normalize_policy_mode({
        **{k: last_cfg[k] for k in (
            "n_states", "n_actions", "action_space", "policy_mode",
            "learning_rate", "gamma", "clip_param", "value_loss_coef",
            "entropy_coef", "batch_size", "ppo_epochs", "lam", "hidden_dim",
            "training_episodes", "training_time", "final_mean_return", "final_mean_actions",
        )},
        "best_mean_return": float(best_mean_return) if best_mean_return != float("-inf") else float(final_mean_return),
        "best_episode": int(best_episode),
        "best_window": int(best_window),
        "checkpoint_role": "best",
        "note": best_note,
    })
    model_path, config_path = save_model(agent, best_cfg, best_path)

    if verbose:
        print(f"\n{'='*60}")
        print("训练完成")
        print(f"{'='*60}")
        print(f"总时间: {training_time/3600:.2f} 小时")
        print(f"末 100 轮均回报: {final_mean_return:.4f} | 均动作数: {final_mean_actions:.1f}")
        print(f"BEST → {model_path}")
        print(f"       {best_note}")
        print(f"LAST → {last_path}")
        print(f"{'='*60}\n")

    return {
        "model_path": model_path,
        "config_path": config_path,
        "last_model_path": last_path,
        "last_config_path": last_cfg_path,
        "return_list": return_list,
        "mean_return_list": mean_return_list,
        "action_count_log": action_count_log,
        "loss_log": loss_log,
        "training_time": training_time,
        "final_mean_return": float(final_mean_return),
        "final_mean_actions": float(final_mean_actions),
        "best_mean_return": float(best_cfg["best_mean_return"]),
        "best_episode": int(best_episode),
        "config": best_cfg,
    }


def _analyze_convergence(return_list: list, action_count_log: list) -> Dict:
    """短报告用：前后 100 轮对比；不足 100 轮则跳过。"""
    if len(return_list) < 100:
        return {"status": "训练轮数不足，无法评估收敛性"}
    final_100_return = return_list[-100:]
    final_100_actions = action_count_log[-100:]
    early_100_return = return_list[:100]
    early_100_actions = action_count_log[:100]
    return_improvement = float(np.mean(final_100_return) - np.mean(early_100_return))
    action_improvement = float(np.mean(early_100_actions) - np.mean(final_100_actions))
    return_stability = float(1 - (np.std(final_100_return) / (np.mean(final_100_return) + 1e-6)))
    return {
        "return_improvement": return_improvement,
        "action_improvement": action_improvement,
        "return_stability": return_stability,
        "convergence_status": (
            "已收敛" if return_improvement > 0.5 and action_improvement > 2 and return_stability > 0.7
            else "训练中"
        ),
    }


# ==================== 训练结果保存和可视化 ====================

def save_training_results(
    results: Dict,
    save_dir: str | Path | None = None,
) -> Dict[str, str]:
    """
    保存训练结果（日志、报告、图表）
    
    Args:
        results: train_ppo()返回的结果字典
        save_dir: 保存目录（默认 output/_scratch/train；正式走 runs/.../01_train）
    
    Returns:
        Dict: 保存的文件路径
    
    Raises:
        ValueError: 训练结果为空或无效
    """
    # 验证输入数据
    if not results.get('return_list') or len(results['return_list']) == 0:
        raise ValueError("错误：训练结果为空，无法保存")

    if not save_dir:
        save_dir = ROOT / "output" / "_scratch" / "train"
    save_dir = mkdir(save_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    artifacts = {}

    # 1. 保存训练日志（CSV）
    log_df = pd.DataFrame({
        'episode': range(1, len(results['return_list']) + 1),
        'return': results['return_list'],
        'mean_return': results['mean_return_list'],
        'action_count': results['action_count_log']
    })
    log_path = os.path.join(save_dir, f'training_log_{timestamp}.csv')
    log_df.to_csv(log_path, index=False)
    artifacts['log'] = log_path
    print(f"✓ 训练日志已保存: {log_path}")

    # 1b. PPO update 损失落盘（面板曲线；旧跑次可能无此文件）
    loss_log = results.get("loss_log") or []
    if loss_log:
        loss_df = pd.DataFrame(loss_log)
        # 按 episode 聚合，便于与回报曲线对齐
        if "episode" in loss_df.columns:
            agg = (
                loss_df.groupby("episode", as_index=False)[
                    ["policy_loss", "value_loss", "entropy_loss", "total_loss"]
                ]
                .mean()
                .sort_values("episode")
            )
            # 保留逐 batch 明细 + 按局均值两份：明细给诊断，均值给看板
            loss_detail_path = os.path.join(save_dir, f"training_loss_detail_{timestamp}.csv")
            loss_df.to_csv(loss_detail_path, index=False)
            loss_path = os.path.join(save_dir, f"training_loss_{timestamp}.csv")
            agg.to_csv(loss_path, index=False)
            artifacts["loss"] = loss_path
            artifacts["loss_detail"] = loss_detail_path
            print(f"✓ 训练损失已保存: {loss_path}（明细 {loss_detail_path}）")
        else:
            loss_path = os.path.join(save_dir, f"training_loss_{timestamp}.csv")
            loss_df.to_csv(loss_path, index=False)
            artifacts["loss"] = loss_path
            print(f"✓ 训练损失已保存: {loss_path}")
    
    # 2. 生成训练报告（JSON）
    # 计算统计数据
    return_list = results['return_list']
    action_count_log = results['action_count_log']
    
    # 最后100轮的平均值
    final_100_return = return_list[-100:] if len(return_list) >= 100 else return_list
    final_100_actions = action_count_log[-100:] if len(action_count_log) >= 100 else action_count_log
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'training_summary': {
            'total_episodes': len(return_list),
            'total_time_hours': round(results['training_time'] / 3600, 2),
            'final_mean_return': round(results['final_mean_return'], 4),
            'final_mean_actions': round(results['final_mean_actions'], 2),
        },
        'return_statistics': {
            'min': float(np.min(return_list)),
            'max': float(np.max(return_list)),
            'mean': float(np.mean(return_list)),
            'std': float(np.std(return_list)),
            'final_100_mean': float(np.mean(final_100_return)),
        },
        'action_statistics': {
            'min': float(np.min(action_count_log)),
            'max': float(np.max(action_count_log)),
            'mean': float(np.mean(action_count_log)),
            'std': float(np.std(action_count_log)),
            'final_100_mean': float(np.mean(final_100_actions)),
        },
        'convergence_analysis': _analyze_convergence(return_list, action_count_log),
        'hyperparameters': results['config'],
    }
    report_path = os.path.join(save_dir, f'training_report_{timestamp}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    artifacts['report'] = report_path
    print(f"✓ 训练报告已保存: {report_path}")
    
    # 3. 训练曲线（英文标签，避免无中文字体时刷屏警告）
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("PPO training", fontsize=16, fontweight="bold")

    episodes = range(1, len(results["return_list"]) + 1)

    ax1 = axes[0, 0]
    ax1.plot(episodes, results["return_list"], alpha=0.3, label="episode return", color="blue")
    ax1.plot(episodes, results["mean_return_list"], label="mean return", color="red", linewidth=2)
    ax1.set_xlabel("episode")
    ax1.set_ylabel("return")
    ax1.set_title("Return")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(episodes, results["action_count_log"], color="green", linewidth=1.5)
    ax2.fill_between(episodes, results["action_count_log"], alpha=0.3, color="green")
    ax2.set_xlabel("episode")
    ax2.set_ylabel("nonzero actions")
    ax2.set_title("Actions / episode")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    loss_log = results.get("loss_log", [])
    if loss_log:
        update_episodes = np.linspace(0, len(return_list), len(loss_log))
        policy_losses = [l.get("policy_loss", 0) for l in loss_log]
        value_losses = [l.get("value_loss", 0) for l in loss_log]
        ax3.plot(update_episodes, policy_losses, label="policy loss", marker="o", markersize=3)
        ax3.plot(update_episodes, value_losses, label="value loss", marker="s", markersize=3)
        ax3.set_xlabel("episode")
        ax3.set_ylabel("loss")
        ax3.set_title("Loss")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "no loss log", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Loss")

    ax4 = axes[1, 1]
    ax4.axis("off")
    stats_text = (
        f"episodes: {len(results['return_list'])}\n"
        f"hours: {results['training_time']/3600:.2f}\n"
        f"final mean return: {results['final_mean_return']:.4f}\n"
        f"return max/min/std: {np.max(results['return_list']):.2f} / "
        f"{np.min(results['return_list']):.2f} / {np.std(results['return_list']):.2f}\n"
        f"final mean actions: {results['final_mean_actions']:.1f}\n"
        f"actions max/min/std: {np.max(results['action_count_log']):.0f} / "
        f"{np.min(results['action_count_log']):.0f} / {np.std(results['action_count_log']):.1f}\n"
    )
    ax4.text(
        0.05,
        0.95,
        stats_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'training_results_{timestamp}.png')
    # 与权重保存一致：经文件句柄写入，避开中文路径 + 旧 torch 的路径字符串坑
    with open(plot_path, "wb") as f:
        plt.savefig(f, dpi=150, bbox_inches="tight", format="png")
    plt.close()
    artifacts['plot'] = plot_path
    print(f"✓ 训练结果图已保存: {plot_path}")
    
    return artifacts
