# 二期 HITL

当前完整操作见[修复后代码手册](D:/Desktop/人在回路修正的强化学习模型/交付说明_20261025/02_修复后代码操作手册与差异.md)。审阅修改后重算、整季接受后逐年定稿，才能BC；`main.py`没有finalize STEP，须使用看板或`run.py hitl finalize`。不要用`confirmed=True`或改指纹代替实际审阅。

与 Web 看板**共用** `output/hitl_sessions/<sess>/`。  
`seed` 会把一期 `01_train`+`02_eval` **整拷进会话**，单个文件夹即可分析全流程。

正式分步请用仓库根 **`main.py`**（`STEP = "2.1"` … `"2.5"` 或 `"web"`），与 `api.hitl_*` 同源。

```bash
# 等价 CLI（正式，非冒烟）
python run.py pipeline                 # 一期 → runs/run_<ts>/
python run.py hitl seed --from-run …   # 二期会话
python run.py hitl board               # = main.py STEP web
```

---

## 目录

```
hitl_sessions/sess_<ts>/
  01_train/   02_eval/          ← 来自一期（完整拷贝）
  03_review/  original + current
  04_bc_export/  05_bc/     ← labels（BC 监督样本）+ 微调权重
  06_compare/ 07_metrics/
  _workdir/
```

| 原则 | 做法 |
|---|---|
| 与 Web 共用 | 同一 sess 路径 |
| 原始 vs 当前 | `original_trajectories/` 锁定；改写只动 `current/` |
| BC 时机 | 各审阅年审完并定稿，再统一微调；审阅期冻结原 PPO |

---

## 模块 ↔ 代码

| 设计 | 代码 |
|---|---|
| 基准（含一期拷贝） | `session` + `pipeline.seed` |
| 审阅交互 | `web/`（结构化 `reason_code`、一键认可需重看） |
| 后续重算 | `resync.py` |
| 定稿导出 | `recompute.py` + `pipeline.finalize_review` |
| BC | `bc_finetune.py` |
| 三方评价 | `compare_eval.py` |
| 连续小额灌/排 | `evaluation/operation_runs.py` → `yearly_metrics` |
| 等雨/偏干事后核查 | `expert_rationale_analysis.py` → `07_metrics/expert_rationale_audit.json` |
| 真人会话机制对照 | `human_followup_experiment.py`（只读；不作主结果） |
| 模拟专家探索 | `gaoyou_experiment.py`（**非**论文真人主线） |

对外：`seed` · `review` · `resync` · `finalize_review` · `bc` · `compare_three`。CLI 定稿命令为 `finalize`。看板默认 BC 配置未必是论文最终超参；正式结果以定稿会话 + 锁定配置为准。

### 审阅操作要点

1. 从冻结 PPO run 新建看板会话（勿用模拟会话）。  
2. 改「专家」动作时选结构化改因（等雨 ≠ 允许偏干）；清「需重看」后逐年定稿。  
3. BC → 三方；看全评估年 + 留出年；小额段与改因核查见 `yearly_metrics` / `expert_rationale_audit.json`。  
4. 细节与 API：`docs/WEB_HITL.md`、`docs/场景与输出.md`。
