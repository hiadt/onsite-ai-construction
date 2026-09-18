# PathGuard 多构型数据清单

本目录保存五轴、六轴工程车辆的统一数据口径。原始审计记录仍保留在 `analysis/data_recovery/` 和 `reports/data_recovery/`，本目录只保存规范清单、统计和可复现实验入口。

## 数据口径

- 现有主基线：359 条去重候选路线。
- ZYJisBoss：55 条候选，其中 14 条重复、32 条新增无标签、9 条未确认、0 条新增有标签。
- SHANYU000：702 条本机唯一有标签路线、3061 条本机唯一无标签路线、815 条同路线新报告、4089 条重复、92 条冲突、20 条未解析。
- 五轴、六轴均纳入统一数据池；车型结构证据和结果标签分别记录。

## 字段原则

- `vehicle_structure`：five_axis、six_axis、unknown_or_conflict。
- `label_status`：labeled、unlabeled、aggregate_only、unknown。
- `comparison_status`：baseline_existing、confirmed_new_labeled、confirmed_new_unlabeled、same_route_new_report、duplicate、conflict、unresolved。
- 结构推断可以用于车型分层和路线泛化实验，但不得伪造逐样本结果标签。

禁止在仓库提交原始 NPZ、模型、压缩备份、凭据、个人绝对路径和大型运行缓存。
