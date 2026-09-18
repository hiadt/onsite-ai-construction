# 新增无标签路线代表样冻结

输入仅为 `confirmed_new_unlabeled_routes.csv` 与 `local_file_inventory.csv`，未重新扫描磁盘。

- 输入候选：32 条
- 车型结构 × 地图分组：32 组
- 冻结代表样：32 条
- five_axis：20 条
- six_axis：12 条
- unknown_or_conflict：0 条
- 损坏或哈希不一致：0 条

每组优先选择路线长度、点数和文件哈希完整的记录。复制前后均校验 SHA-256，并实际读取 NPZ 全部数组；此前候选集合中的重复副本未上传。

详细原始路径、仓库路径、车型证据和哈希见 `frozen_raw_samples_manifest.csv`。
