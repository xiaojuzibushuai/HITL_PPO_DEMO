# PPO_Rice_Env

2026-10-25操作交接：先读[文件与操作总索引](D:/Desktop/人在回路修正的强化学习模型/文件与操作总索引_20261025.md)与[修复后代码操作手册](D:/Desktop/人在回路修正的强化学习模型/交付说明_20261025/02_修复后代码操作手册与差异.md)。本轮论文使用`run_20261024_170858`；看板案例为`sess_v3_candidate_feedback_20261025`。只读结果时不要重新训练或覆盖正式会话。

**一律在本目录运行**（`cd PPO_Rice_Env`）。业务包只认：`common` / `train` / `decision` / `evaluation` / `hitl` / `api`。

| 文件 | 角色 |
|------|------|
| **`main.py`** | **正式分步入口**：改 `STEP`（1.1→2.5 或 `web`） |
| **`api.py`** | 对外调度层；具体业务实现位于下列各包 |
| **`run.py`** | 等价命令行 |

---

## 正式流水线（推荐用 main.py）

打开 `main.py`：

1. 改 **`STEP`**（当前文件为`"web"`；先确认实际值）
2. 跑 `python main.py`
3. 按终端提示回填 **`RUN_REL` / `SESSION_REL`**，再改下一步

| STEP | 含义 |
|------|------|
| `1.1` | 校验配置与数据 |
| `1.2` | 正式训练（默认 1500 轮）→ `output/runs/…/01_train/` |
| `1.3` | 全验证年评估 → `…/02_eval/` |
| `1.0` | 一期一键 = 1.1→1.2→1.3 |
| `2.1`～`2.5` | seed → review → resync → **逐年定稿** → bc → compare；定稿使用看板或CLI，main没有对应STEP |
| `web` | 开看板（与上表同一套 api 模块） |

明细见 **`docs/模块验证.md`**。

导入自检：

```bash
python run.py self-check
```

---

## Web 完整方案

```text
main.py → STEP = "web"
# 浏览器 http://127.0.0.1:5050
```

面板顺序：已有训练会话 → 训评 → 人工审阅（改动作 ↔ 一致性重算）→ 各审阅年定稿 → 行为克隆(BC) → 三方对比。

已有PPO时可直接从新建审阅会话开始。注意：`run.py`不带子命令会启动训练流水线；仅看结果请用`python run.py hitl board`。看板/通用CLI的BC默认值与论文选定配置不同，完整参数见交接手册。

---

## 程序与论文产物

- `output/runs/run_<时间戳>/`：一期 `uploads` + `01_train` + `02_eval`
- `output/hitl_sessions/sess_<时间戳>/`：二期（与看板共用）
- `output/paper_experiments/gaoyou_v3_20261025/`：本轮正式论文的冻结结果与候选选择；不是单个会话的三方均值。

---

## 文档

| 文档 | 内容 |
|------|------|
| `docs/模块验证.md` | 与 main.py 同步的正式步骤 |
| `docs/DESIGN.md` | 设计总览 |
| `docs/场景与输出.md` | 目录与字段 |
| `docs/RL与训练.md` | 状态 / 奖励 / PPO |
| `docs/WEB_HITL.md` | 看板说明 |
| `hitl/README.md` | HITL CLI |
