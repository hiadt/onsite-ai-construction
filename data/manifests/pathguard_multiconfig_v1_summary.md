# PathGuard-MultiConfig-v1 数据整理摘要

状态：已合并两位成员的只读审计结果，待统一规范清单脚本生成最终逐条 manifest。

## 已合并来源

| 来源 | 结果 |
|---|---:|
| 主数据基线 | 359 条去重候选 |
| ZYJisBoss 候选 | 55 条：14 重复、32 新增无标签、9 未确认、0 新增有标签 |
| SHANYU000 候选 | 702 有标签、3061 无标签、815 同路线新报告、4089 重复、92 冲突、20 未解析 |

## 车型结构

SHANYU000 扫描结果包含六轴显式 9 条、六轴结构推断 661 条、五轴显式 2853 条、未知或冲突 240 条；ZYJisBoss 包含六轴显式 8 条、六轴结构推断 19 条、五轴显式 27 条。五轴和六轴均进入统一数据池。

## 使用规则

有逐样本标签的记录用于监督指标；无标签记录用于路线覆盖、风险排序和拒判验证；同路线多报告用于稳定性和版本变化分析；重复记录保留来源但不重复计入样本量。

## 后续生成

运行仓库中的整理脚本后，生成：

- `pathguard_multiconfig_v1_manifest.csv`
- `labeled_samples.csv`
- `unlabeled_routes.csv`
- `same_route_new_reports.csv`
- `duplicates.csv`
- `conflicts.csv`
- `unresolved.csv`
- `vehicle_structure_evidence.csv`
