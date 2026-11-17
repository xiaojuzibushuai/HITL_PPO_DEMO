# PPO_Rice_Env 设计总览（一期定稿）

## 1. 目录与「谁是真相源」

```text
PPO_Rice_Env/
├── main.py / run.py / api.py   # 对外入口（METHOD / CLI / 可编程）
├── configs/                    # scenario.json + model_config.json
├── common/                     # paths · scenario · action_mask · model_io · data_prep · run_store
├── train/                      # ★ 一期训练三件套
├── decision/                   # 推理/评估后处理（Hc 守卫 / 限幅）
├── evaluation/                 # 多年闭环评估 + 年指标
├── hitl/                       # ★ 二期：seed→review→resync→bc→compare + Web 看板
├── data/                       # forecast·actual（训）/ *_policy（评）
├── output/runs/                # 一期 run_<ts>/{uploads,01_train,02_eval}
├── output/hitl_sessions/       # 二期 sess_<ts>/01…07
└── docs/
```

show-config → train → eval →（可选）hitl board / pipeline

| 路径 | 角色 |
|---|---|
| `train/PPO_Rice_env.py` | 环境：水量平衡、奖励、reset/step |
| `train/PPO_RL_brain.py` | PPO 网络 + 掩码采样/更新 |
| `train/ppo_model.py` | `train_ppo` / 存盘（best+last） |
| `hitl/*` | 人在回路审阅与 BC 微调 |

**运行只认 `train/` + `hitl/` + `main.py` / `run.py` / `api.py`。**


```bash
cd PPO_Rice_Env
python run.py                       # 无参 = pipeline（IDE 点运行同效）
# 或：python run.py pipeline
# 出问题再分步：show-config / train / eval / predict
```

评估默认**覆盖全部验证年**；只有显式 `--max-years N` 才截断。

---

## 2. CLI

| 命令 | 做什么 |
|---|---|
| `pipeline` | 正式一键：校验配置/数据 → **训练** → **训完后才评估** |
| `show-config` / `train` / `eval` / `predict` | 分步 |
| `self-check` | 改掩码/规则后可选断言（无 CSV） |

`eval` 与 `predict` 的差别：

- `predict`：你给一天的 `h,hmin,…`，出一次动作；不滚下一天。
- `eval`：从季初日滚到末日，每天用 PPO 或固定规则选动作，再经**同一套** `Irrigation.step`（限幅+降雨−ET−渗漏）得到次日水深；最后按年汇总指标再对比。

---

## 3. 评估：固定规则是什么？怎么和 PPO 比？

规则：`<hmin` 灌到 `hmax`；`>Hp` 或 `≥Hc` 排到目标；否则不动。  
两边共用 `env.step` 水量递推；只换决策器。详见下文指标与产物。

`delta_pct`：越低越好 → `(rule-ppo)/|rule|*100`（正=PPO 更省）；越高越好反之。

---

## 4. 指标（一期够用，二期 HITL 直接复用）

**五核心（年）：** 雨总量 / 灌总量 / 排总量 / 雨利用率 `(P-d)/P` / 产量水分系数 `yield_factor`  

**扩展（年）：** 灌排次数、managed_water、胁迫天、超 Hc 天/暴露量/事件数、最长连续超淹、`no_drain_high_days`  

**对比：** `compare_summary.json`（多年均值 + Δ%，含超 Hc / 胁迫等扩展键）+ **`disagree`**（逐日动作分歧率）  

一期**不**再堆新农艺指标；看板用现有列即可。缺的是「专家改动作后重算」——属二期 HITL，不是缺年指标。

**分年图 / KPI 卡片的数据来源：** 一律来自 **评估集**（`forecast_eval` / `actual_eval`）上跑出的 `02_eval/yearly_metrics_*.csv` 与 `compare_summary.json`，**不是**训练集上的训练曲线。训练曲线只解释「学得怎样」；分年柱图解释「在评估年上相对固定规则怎样」。

**评估集年份划分（HITL，见 `hitl/config.py`）：**

| 用途 | 默认年份 | 说明 |
|------|----------|------|
| **审阅年** | 2012, 2013, 2014, 2016 | 生成审阅表 → 专家改写 → **行为克隆(BC)** 监督样本 |
| **留出验证年** | 2017, 2018, 2019, 2020 | 专家默认不改；衡量 HITL 在「未见年」的泛化 |
| **一期二方对比** | 评估集**全部**年 | 无审阅/验证划分 |
| **三方对比主表** | 评估集**全部**年 | 与二方可对齐；另写 `holdout_compare` 仅留出年 |

**行为克隆(BC)**：用已确认的专家动作做监督微调（交叉熵 + 相对原策略 KL），**不是**再跑 PPO.update。面板与文档请写全称「行为克隆(BC)」。

---

## 5. 产物是否够可视化 / 支撑「HITL 重要」

| 阶段 | 文件 | 用途 |
|---|---|---|
| 训练 | `*_best/last.pth`、`*_config.json` | 权重与旁路超参 |
| 训练 | `training_log_*.csv` | 逐局回报 / mean / 非零动作数 → 学习曲线 |
| 训练 | `training_loss_*.csv` | PPO update 的 policy / value / entropy_loss（可解释收敛；旧跑次可能无此文件） |
| 训练 | `training_report_*.json`、`training_results_*.png` | 汇总、粗收敛判断、静态总览图 |
| 评估（评估集） | `policy_output_{PPO,RULE}.csv` | 逐日 h/a/雨/ET |
| 评估（评估集） | `yearly_metrics_*.csv`、`compare_summary.json` | **KPI 卡片 + 分年柱图** |
| 评估（评估集） | **`policy_compare_daily.csv`** | PPO≠规则的日 → **HITL 优先审阅点** |
| 评估 | `run_manifest.json` | 路径清单，看板/HITL 免扫盘 |

**看板「训练结果」应对齐旧结果页主视觉（不新增农艺指标）：**

1. **必要**  
   - 落盘 `training_loss_*.csv`（避免面板只能靠 png）  
   - 用现有 log / yearly / compare 做：回报+动作曲线、KPI 卡片、**评估集分年柱图**  
   - 流程门禁：数据集校验通过才训评；或上传已训 `.pth` + 评估集校验通过后仅整季推理  
2. **建议**  
   - 滑动窗口平滑回报 + 收敛文案（log + report）  
   - `delta_pct` 覆盖超 Hc / 胁迫等扩展键  
   - best / final 切换再评估（可选）  
3. **暂缓**  
   - 独立「原始动作熵」采样序列；现有 `entropy_loss` 足够  

**结论：** 一期农艺「五核心」不必再扩作物种类。HITL 论文机制侧需额外核算：连续小额灌溉/续灌/排水段（执行量）、结构化改因（等雨 vs 允许偏干分开）及事后预报兑现与水层/`ks` 响应——见 `evaluation/operation_runs.py`、`hitl/expert_rationale_analysis.py`、`docs/WEB_HITL.md`。

---

## 6. 一期 / 二期边界 + 统一面板

正式产物**只认** `output/runs/` 与 `output/hitl_sessions/`。Web 看板与二期 CLI **共用**后者。

| 一期 | 二期 |
|---|---|
| `runs/run_<ts>/{uploads,01_train,02_eval}` | `hitl_sessions/sess_<ts>/`（**含**拷贝进来的 `01_train`+`02_eval`） |
| 可单独 `run.py pipeline` | 可单独 `hitl seed/review/…`；看板同一目录 |

**面板只编排 API（不重写算法）：**

```text
上传表 → uploads/（或 pipeline 快照 scenario）
  → api.pipeline → runs/.../{uploads,01_train,02_eval}
  → hitl.seed(from_run)（整拷 uploads+训评进 sess）→ review
  → 确认/修改 → resync（original 锁定，current 更新）
  → 各年审完 → bc → compare_three（规则/原PPO/HITL）
```

**审阅 vs 定稿：** `resync` 冻结 PPO 刷建议；`bc` 固定动作重放、仅 `confirmed`、各年审完再微调。

---

## 7. 二期会话 = 完整分析包

| 目录 | 内容 |
|---|---|
| `uploads` | 原始训练/评估 CSV（seed 整拷 + scenario 补全） |
| `01_train` · `02_eval` | 一期整拷，离线可分析训练与基准对比 |
| `03_review` | `original_trajectories/`（锁）+ `current/`（可变） |
| `04_bc_export` · `05_bc` | BC 监督 labels、`ppo_hitl.pth` |
| `06_compare` · `07_metrics` | 三方对比 + 吸收率/耗时 |

对外：`api.pipeline`；`seed` · `review` · `resync` · `bc` · `compare_three`。详见 `场景与输出.md`。

---

## 8. 测试入口（对外只认 `run.py`）

```bash
python run.py pipeline              # → output/runs/run_<ts>/
python run.py hitl board            # 读写 hitl_sessions（与 CLI 共用）
python run.py hitl seed --from-run output/runs/run_…
python run.py hitl review | resync | bc | compare
```

编程：改 `main.py` 的 `METHOD`，或 `from api import dispatch; dispatch("pipeline")`。  
Web 调用链：`docs/WEB_HITL.md`。
