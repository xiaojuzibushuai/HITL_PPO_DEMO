# Web HITL 看板

## 启动

```text
main.py → STEP = "web"
# 或: python run.py hitl board
# 浏览器 → http://127.0.0.1:5050
```

仅从 `PPO_Rice_Env` 目录启动。硬刷新以加载最新静态资源（`asset_v`）。

**会话切换：** 新建/加载/更换训练会话会清空训评曲线、校验框、文件选择控件与评估摘要，再按新 run 重绘；新建/加载/更换微调会话会清空审阅本地改动、水层/灌排/降雨图与三方区。同一会话内 resync / BC / 对比只刷新状态。

**新建 run 与校验：** 默认**不**自动拷贝情景表（`snapshot_scenario=false`）。须「保存上传文件」或点「使用默认情景表」写入本 run 的 `uploads/`，再点「② 数据集校验」。校验读的是磁盘 `uploads/`，不是浏览器 file input；下方灰色标签 = 磁盘已有文件。空上传会失败；每项单独检查扩展名（`.csv` / 可选 `.pth`）。

---

## 流程（规范）

```text
⓪ 选训练会话（A 新建 / B 加载历史训练会话 · 互斥）
→ ① 上传数据集（仅新建路径）
→ ② 数据集校验（仅新建路径；通过才允许训评）
→ ③ 训练 / 整季推理 · 结果回放
→ ④ 人工审阅（改专家动作 → **自动**一致性重算 → 本审阅年定稿 → 换年，闭环）
→ ⑤ 行为克隆(BC)（须审阅就绪）
→ ⑥ 三方对比
```

| 会话类型 | 磁盘 | 面板展示 |
|----------|------|----------|
| **训练会话** | `output/runs/run_*` | `01_train` 曲线/损失 + **评估集** `02_eval` KPI / 分年图 |
| **微调会话** | `output/hitl_sessions/sess_*` | `03_review`～`07_metrics` 摘要与人工审阅表 |

**一致性重算**嵌在④人工审阅内：改「专家」动作后约 0.4s **自动**重算水深与后续日。改第 T 天后，T 日执行专家动作，T 后重推 PPO 建议并标「需重看」；专家逐日认可当前建议或指定专家动作。若同一批次同时修改多个日期，只保留最早修改，后续选择因状态已改变而作废并要求复核。「本审阅年定稿」不能清除待复核标记；有待复核时硬拒绝。步进条不单列「重算」。

两条路径：

- **A**：上传训/评表 → 校验通过 → 训练或整季推理  
- **B**：加载历史训练会话（或上传已训 `.pth` + 评估集）→ 训评回放 → 人工审阅后续  

**行为克隆(BC)**：用专家确认动作监督微调（CE+KL），不是再跑 PPO。  

**年份：** 人工审阅默认 `2012/2013/2014/2016`；留出 `2017–2020`。下拉**只列规范审阅年且强制单年**（不含留出年、无「全部」；三图始终对应当前年）。二方与三方主表均为**全评估年**；三方另汇总留出年 `holdout_compare`。  

面板文案不使用「一期 / 二期」称呼。分年柱图 / KPI **只来自评估集**（见 `docs/DESIGN.md` §4–5）。

**训练过程曲线：** 指标单位为「局」（Episode）。回报图旁「展示平滑」为**滑动平均窗**（50/100/200/500 局或关），只改绿线展示，不是训练超参。默认 100 局。

看板 KPI 默认展示：雨利用率、灌/排总量、管理水量、产量水分系数、灌排次数。  
**Δ%** = 相对**固定规则基线**（绿/正 = 相对基线更好；越低越好指标已取反展示）。  
`stress_days` / `overpond_*` 仍写入 `yearly_metrics_*.csv`，面板主表不展示（属扩展诊断，非论文五核心）。

**文件夹摘要：** 每个文件下列用途 +「看指标→文件」速查；支持 **zip 一键打包**。说明与 `docs/场景与输出.md` §2 对齐（后端 `_FILE_ROLE_EXACT` / 前端 `FILE_ROLES`）。  

评估集与三方可在「多年均值」与「单年整季」之间切换。  
- **多年均值**：`02_eval/compare_summary.json`（KPI）+ `yearly_metrics_*.csv` 分年柱图（固定顺序：雨利用率→灌→排→管理水量→产量系数）。  
- **单年**：同年 KPI + `policy_output_*.csv` 整季三图（**仅 PPO 水深** + 灌排 + 实测雨；阈值用色带 hmin–hmax）。进入人工审阅且 BC 尚未生成时，训评/事后结果面板隐藏。

人工审阅「与规则分歧」= **模型执行 ≠ 经验规则**（浮点容差 1e-6；不是专家改动）。默认表展示**全部日**（含晒田/落干锁定行，可取消「显示固定规则日」只看可编辑日）；可再勾选仅看分歧。表列：**模型提案** / **模型执行** / **经验规则** / **专家**。整季三图始终含固定规则日。  

降雨：专家审阅仅显示决策时可得的 `P_cumu7` 预报；当日实测 `P_1_a` 继续供环境水量平衡与事后评价使用，不进入审阅表/API，也不在审阅期间显示训评回放。旧会话加载时会剔除曾存入审阅表的实测降雨列。  

### `03_review/` 目录分工

| 路径 | 作用 |
|------|------|
| `original_trajectories/` | **只读基线**：生成审阅表时锁定的原 PPO 整季轨迹。可「重置本审阅年」从这里恢复。 |
| `current/` | **唯一实时审阅表**：改动作/重算都写这里。 |
| `edits_log.jsonl` | 专家确认/改动的追加日志（审计用）。 |

不再维护 `history/`（与 current 重复）；回退用 original。

**审阅完成后：** 不必另存「完成表」——定稿结果就在 `current/`。点 BC 时从 `current` 读表 → 按专家动作重放 → 写出：

| 路径 | 作用 |
|------|------|
| `04_bc_export/labels.json` | 定稿后整季重放导出行；**进 BC** 的是 `include_in_bc=true`（定稿可编辑日；晒田/落干默认排除） |
| `05_bc/ppo_hitl.pth` | 微调后的策略权重 |
| `05_bc/hitl_result.json` | BC 摘要：`n_labels` / `n_include_bc` / `n_edited` / `review_labels_parity` / CE 曲线 |
| `04_bc_export/review_labels_parity.json` | 审阅表 vs 导出样本逐日校验（`h_before` / `a_exec↔a_actual` / `h_after`） |

**导出 / BC 门槛（硬）：** 定稿请求序列 → 从季初 ``hmax`` 连续重放（不调 PPO）→ 与定稿轨迹逐日校验（晨水 / 状态 / 请求 / 执行 / 日末）全部通过才允许 BC。非法请求档须重新审阅，不能用执行量冒充专家意见。``label_smoothing=0``；非有限损失中止。定稿写入轨迹指纹；改动作后该年须重新定稿。正式 BC 还会检查 `feedback_source`，拒绝模拟审阅/规则生成标签。

**BC 内部验证/开发季：** 默认用审阅年中的 `2016` 做 CE 早停（整季留出，避免同季相邻日同时进训练/验证）。**论文勿称 2016 为独立测试季**；真正未参与审阅与调参的留出评价季是 `2017–2020`。若调试会话只有一个审阅年，才兼容行级随机切分——该结果不作正式实验配置。

**会话溯源：** 看板新建只写 `review_mode=human_dashboard`、`feedback_status=pending`；定稿后才写入 `feedback_source` / `feedback_sources_by_year` 并置 `feedback_status=recorded`。正式 BC 仍只认真人来源（`human` / `human_whole_season_acceptance`）。

**结构化改因：** 改动作时选 `reason_code`：`merge_small_irrigation` / `wait_forecast_rain` / `allow_low_water` / `other`（等雨与允许偏干必须分开）。自由文本仅补充。三方/指标接口会写 `07_metrics/expert_rationale_audit.json`：分项报告减灌、预报兑现（`P_cumu7` 时窗=决策日+随后6日，对齐 `P_1_a…P_7_a`）、偏干后水层/`ks`；**不合成等雨成功率**。

**连续小额操作（执行量 `m`/`d`，非提案）：**

| 指标 | 判据 | 输出 |
|------|------|------|
| 连续小额灌溉（主） | 日历连续、全日 `0<m≤20`、长度≥2；另拆 ≥3 d | 段数 / 天数 / 水量 |
| 连续小额续灌（补） | 首日 `m>20`，后续连续 `0<m≤20`（例 60,20,20） | 段数 / 天数 / 水量 |
| 连续小额排水 | 与主指标同形，但用 `d`；**不与灌水合并** | 段数 / 天数 / 水量 |
| 操作日数 / 连续段数 | 有灌或排的日数；任意正量日历连续段 | 与旧 `irrigation_times`（日数）区分 |

零操作、日期不连续、年界、固定规则日打断。实现：`evaluation/operation_runs.py` → 写入 `yearly_metrics_*.csv`；审阅高亮 `is_small_run` 只标**灌溉**主段+续灌后续小额日。阈值 20 mm 为研究判据（10 mm 档的两档），论文中说明；敏感性可用 10/30 mm。

**升级后：** 对本审阅年 **resync → 重新定稿** 再 BC（旧执行量 / 渗漏口径可能已变）。吸收率主指标排除「掩码仅剩一档」强制日。

**进训样本语义（易混）：**

- `n_labels`：导出行总数（含固定规则日等，整季轨迹行）。
- `n_include_bc`：**真正进行为克隆**的天数 ≈ 定稿可编辑日（排除晒田/落干）。不是「只训改动的 20 天」。
- `n_edited`：其中专家动作 ≠ 模型提案的天数；训练时权重 = `w_edit`，其余进训日权重 = 1。
- 为何这样：只训改动日易过拟合、忘掉原 PPO；整季确认日 + 加重改动日，才是「保住经验 + 拉向专家」。

**吸收率（`07_metrics/absorption.json`）：**

- `edit_reproduce_rate`：改动日上 HITL 策略 argmax 是否等于专家标签（学没学会改）。
- `confirm_hold_rate_policy`：未改日是否仍跟标签（有没有忘掉原策略）。
- `by_year`：按审阅年拆分的同上两率；看板「对比视图」切年时 BC 摘要跟年走，多年均值用全体。
- 复现率低 ≠ 训练失败：CE 下降只说明拟合标签分布；argmax 复现专家改动另计。KL/`w_edit` 偏保守时常见「任务 KPI 略变、改动复现仍低」。

**为何 HITL 相对原 PPO 看起来差不多：**

1. 专家只改了少量日（如 20/356），信号弱。  
2. BC 带 KL，刻意不远离原策略。  
3. **留出验证年**审阅未改，但参数更新仍可能改变未标注状态；留出评价正是检查外溢，**不能预设 HITL≈原 PPO**。  
4. 看「多年均值」或**审阅年**与留出年时，分别解释；勿把「未改」说成「应当不变」。

改第 T 天 → 重算写 **`current/`**（T 前不动，T 后旧改动作废）；`original` 不动。  

### 看板⑥ BC 结果摘要与对比视图

- 视图 = **多年均值**：摘要用全体吸收率 + 多年 KPI 均值相对改善。  
- 视图 = **某年**：摘要用该年 `by_year` 吸收率 + 该年三方 KPI；无专家改动的年勿预设 HITL≈PPO。  

### 人工审阅 → BC 门禁

1. 生成审阅表时「年数上限」须**留空**，否则会只出一年。  
2. **专家**列默认 **—（未填）**；选档即指定专家请求。改第 **T** 天 → **仅该审阅年**：**T 前整行不动**；**T 日**晨水深与模型提案不变；专家请求与 PPO 走同一执行链；T 后按新水层刷新 PPO 建议并标「需重看」。专家可逐日点「认可当前建议」或选择专家动作；认可当前 PPO 档不改变该日执行轨迹，但会记录为明确反馈。  
3. **本审阅年定稿** = 显式接纳已完成复核的整季 + 写入轨迹指纹。任何 `needs_reconfirm` 未清零时，接口硬拒绝定稿；不得用定稿按钮代替逐日复核。定稿前本地尚未提交的动作/确认会先提交。改动作 / 确认后该年指纹与定稿失效，须再定稿。  
4. **`bc_ready` / CLI `bc()`** 共用 `assess_bc_readiness`：定稿年齐全、有改动日、可编辑日已确认、无待复核、**轨迹指纹匹配**、**连续重放 parity**。看板会返回 `bc_blockers` / `parity_ok`。验收可用 `require_edits=False`。  
5. 预检失败则看板不显示可 BC、CLI 亦拒绝。看板提交 BC **成功后自动跑三方对比**。重置审阅年会同时清定稿与该年指纹。  

**经验规则列：** `a_rule` = 原规则整季轨迹同日动作（基线对照）；`a_rule_current_state` = 按**当前审阅水层**现算的规则建议。勿把前者当成当前状态下的提示。

### 审阅信息边界

人工审阅表、`GET /api/rows` 与审阅整季图仅显示决策时可用的 `P_cumu7` 预报，不返回 `P_1_a`。开始人工审阅后，训评/实测雨回放面板默认隐藏；BC 生成后才恢复结果分析。底层实测气象仍参与环境递推与事后绩效计算，不作为当日动作的可见输入。

---

## 智能体训练 API

| HTTP | 作用 |
|------|------|
| `POST /api/run/new` | 新建 `run_*`（可快照默认表） |
| `POST /api/run/switch` | 绑定历史训练会话（回放） |
| `POST /api/run/upload` | 上传 csv → `uploads/`；可选 `model` → `01_train/ppo_rice_best.pth` |
| `POST /api/run/validate` | 校验；`mode=full\|eval_only` |
| `POST /api/run/start_train` | 后台训+评；**须校验通过**；`eval_only` 须已有 best |
| `GET /api/run/progress` | 轮询进度 |
| `GET /api/run/train_log` | 训练曲线（回报/动作；有则含损失） |
| `GET /api/run/summary` | 摘要（report + compare + 评估集 yearly） |
| `GET /api/run/season?year=` | 评估集单年整季轨迹（水深/阈值/动作）；人工审阅未完成前不返回 `P_1_a` |
| `GET /api/run/download_zip` | 打包当前训练会话 |

## 专家微调 API

| HTTP | 作用 |
|------|------|
| `POST /api/start_session` | 选训练会话 → 新建微调会话 + seed + 人工审阅表 |
| `POST /api/switch_session` | 加载历史微调会话 |
| `GET /api/rows` | 人工审阅行（含 `forecast_rain_7d`/`P_cumu7`/`reason_code`，不含实测 `P_1_a`；`prefer=1` 仅分歧日；`year=` 筛年） |
| `POST /api/resync` | 改动作/确认/结构化改因；body 可含 `overrides`/`confirms`/`reason_codes`/`reasons` |
| `POST /api/review/accept_pending_year` | 一键认可本审阅年全部需重看 |
| `POST /api/review/finalize_year` | 本审阅年定稿（`year`） |
| `POST /api/review/reset_year` | 本审阅年恢复 original（`year`） |
| `POST /api/bc` | 行为克隆；**须 `bc_ready`**（支持 `w_edit`） |
| `POST /api/compare` | 三方对比（全评估年 + 留出年）；并写 `expert_rationale_audit.json` |
| `GET /api/compare/yearly?year=` | 三方多年均值 / 单年 KPI（含各臂 `means` 与小额段字段） |
| `POST /api/metrics` | 吸收率 + 专家改因事后核查 |
| CLI `hitl rationale-audit` | 仅跑改因事后核查 → `07_metrics/expert_rationale_audit.json` |
| `GET /api/session/download_zip` | 打包当前微调会话 |

会话状态字段（`/api/boot` · `session`）：`expected_review_years`、`confirmed_years`、`edited_years`、`missing_review_years`、`n_edit`、`bc_ready`。
