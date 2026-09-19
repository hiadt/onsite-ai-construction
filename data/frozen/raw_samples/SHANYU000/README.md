# SHANYU000 五轴六轴代表原始样本

本目录仅包含经过哈希校验、可由 NumPy 正常读取且具有逐样本硬标签和报告哈希的代表路线 NPZ。

- 原始报告未复制：历史报告可能包含本机绝对路径，清单仅保留经重新计算验证的 `report_file_sha256`。
- `comparison_status` 仅按 `route_file_sha256` 与 `baseline_v1_359.csv` 精确比较，不等价于历史 `route_input_hash` 去重。
- `hard_certificate_passed=1` 表示历史报告通过；不构成实车验证、安全认证或闭环仿真的替代。
- `manifest.csv` 是文件、标签、车型结构、地图和报告证据的唯一对应清单。
- `SHA256SUMS.txt` 用于校验仓库中的冻结 NPZ。
