# ZYJisBoss 32 条结果回查报告

生成日期：2026-09-19

## 结论

- 回查对象：32 条 `confirmed_new_unlabeled` 和 9 条 `unresolved`。
- 找回明确通过标签：0 条。
- 找回明确失败标签：0 条。
- 只有路线级指标：1 条，来自 9 条未确认路线。
- 只有部署聚合摘要：12 条，均来自 32 条新增无标签路线。
- 完全没有结果：28 条，其中新增无标签路线 20 条、未确认路线 8 条。
- 多结果冲突：0 条。
- 9 条未确认路线中，恢复最终结果标签 0 条；1 条补到路线级指标，但仍不能认定通过或失败，也不能解决其主数据去重状态。
- 新增真实标签训练样本：五轴 0 条，六轴 0 条。

本轮没有把部署汇总数字、路线净空指标或模型输出当作逐样本真实标签。

## 32 条新增无标签路线

### 只有聚合摘要：12 条

`active_deployment.json` 的 `routes` 字段按 `route_file_sha256` 明确关联以下路线，但 `offline_replay` 仅给出批次 `completed=9`、`hard_certificate_passed=9`，不能判断 12 条中具体哪 9 条通过。

- `462f2eccac127d5d`：cliff_1（five_axis）
- `f8c1bcb1b1b4309f`：cliff_2（five_axis）
- `473342c86fc55ee2`：cross_1（five_axis）
- `bb2dcffec0dc2d79`：cross_2（five_axis）
- `8f44c9d01c3b5a47`：mounarea_1（five_axis）
- `c314263dc039c205`：mounarea_2（five_axis）
- `953a8b9da5119b94`：narrow_1（five_axis）
- `eee60842afb61a39`：overpass_1（five_axis）
- `45670c7cab1abadb`：overpass_2（five_axis）
- `e5e2574a68612721`：ramp_1（five_axis）
- `32baf3b90b8ed961`：ramp_2（five_axis）
- `d13f4bbcd386b080`：turnaround_1（five_axis）

### 没有找到结果：20 条

- `e5f81602ec534784`：cliff_3（five_axis）
- `645e80adcac7aa0b`：cliff_6（five_axis）
- `390aea71b9cc217d`：cross_4（five_axis）
- `59aedc56a46a43f6`：ramp_3（five_axis）
- `41def24cc9783941`：ramp_6（five_axis）
- `33c4a3207aea9776`：turnaround_4（five_axis）
- `e21033ab8b0e270c`：turnaround_8796_template（five_axis）
- `1ec33434bae3e9e5`：turnaround_4_tuning（five_axis）
- `fe99043c80742250`：turnaround_1_dynamic（six_axis）
- `929f70daa74cfcd7`：RoadC（six_axis）
- `04a1f40c70c74093`：bridge_4（six_axis）
- `402c4af3a7ab695a`：cross_3（six_axis）
- `2b110d3ed5565dfc`：cross_4（six_axis）
- `5af8e85b784abd44`：mounarea_6（six_axis）
- `8b2e7c13886a99f4`：narrow_4（six_axis）
- `d01d64e359d188ec`：cliff_1（six_axis）
- `5092ea59f5c30498`：cliff_2（six_axis）
- `cb5ab5cec055356c`：narrow_1（six_axis）
- `ba62e96011ce4f85`：ramp_1（six_axis）
- `32744a2b35c6ccc4`：ramp_2（six_axis）

## 9 条未确认路线

### 找到指标但没有最终标签：1 条

- `309d95fbe41bd18a`：cross_1（six_axis）

该路线通过 `route_file_sha256` 命中六轴 `special_action_manifest.json` 的 `cross_1` 项，报告包含连续边界净空和点云净空等 `global_validation` 指标，但没有最终通过/失败字段。

### 没有找到结果：8 条

- `82bea6cdf0deaa0c`：bridge_4（five_axis）
- `ec6a3c490e92fc1f`：mounarea_3（five_axis）
- `6ef28b4eccb9fcf2`：turnaround_2（five_axis）
- `8750337ef091c655`：turnaround_1（unknown_or_conflict）
- `67c5ef8c5a0ae4b0`：cliff_3（six_axis）
- `5928b9b07115143c`：turnaround_4（six_axis）
- `bc56b4605db87979`：turnaround_4（six_axis）
- `07c8050948749e67`：turnaround_1（six_axis）

## 补传文件

需要补传并已纳入本分支的结果证据共 2 份：

- `analysis/data_recovery/outcome_recovery/evidence/ZYJisBoss/active_deployment.json`：支撑 12 条 `aggregate_only` 结论。
- `analysis/data_recovery/outcome_recovery/evidence/ZYJisBoss/six_axis_special_action_manifest.json`：支撑 1 条 `has_metrics_no_final_label` 结论。

原始 NPZ 需要补传：0 份。本轮没有恢复任何真实通过/失败标签，因此没有新增训练样本需要随结果报告补传。与两份证据关联的 13 条路线仍保留在回查 CSV 中，可按原始路径和 SHA-256 复核；不重复上传无标签 NPZ。

## 回查范围与方法

未重新扫描全盘。先复用既有 `local_file_inventory.csv`，再限定在 41 条路线的邻近目录和三个已知项目根中：

- 检查指定的 9 类 JSON 报告文件名；
- 检查 `certificate`、`validation`、`reports`、`outputs`、`runs`、`results` 等结果目录；
- 用 41 条 `route_file_sha256` 对 JSON、CSV、JSONL、日志和文本做精确内容匹配；
- 发现的车辆标定验收报告和通用性能曲线没有目标路线哈希，未用于结果分类。

逐条状态、证据路径、证据哈希和匹配方法见 `analysis/data_recovery/outcome_recovery/ZYJisBoss_outcome_recovery.csv`；补传文件见 `analysis/data_recovery/outcome_recovery/ZYJisBoss_files_to_transfer.csv`。
