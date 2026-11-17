原始数据表（输入，不含 KPI）：
- forecast.csv / actual.csv           训练集气象
- forecast_eval.csv / actual_eval.csv 评估集气象
- growth_para.csv                    生育期与水深约束
可由面板上传，或 pipeline 时从 scenario 自动快照。

训评产物与指标请看同级目录：
- 01_train/training_log_*.csv        学习曲线（episode/return/action_count）
- 01_train/ppo_rice_best.pth         正式评估/审阅用权重
- 02_eval/compare_summary.json       多年均值 KPI 与 Δ%
- 02_eval/yearly_metrics_*.csv       分年 KPI
- 02_eval/policy_output_*.csv        整季逐日水深/动作
详见 docs/场景与输出.md §2。
