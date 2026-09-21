# Gate 2 执行前特征提取报告（ZYJisBoss）

## 1. 对齐范围与标签隔离

本次以 `d423f0e:data/frozen/raw_samples/SHANYU000/manifest.csv` 的 24 条记录作为当前 V3 映射。匹配顺序为 `route_file_sha256` 优先、`sample_id` 次优。标签仅在数值特征完成计算后按 `sample_id` 回填为整数 `true_label`，原有通过/失败结果未修改。

本次统一采用 `pathguard_preexec_v2.0.0`。唯一契约为 `data/derived_features/feature_contract_v2.json`，训练列数量为 55，`schema_sha256` 为 `b365d5b00779cc9f1e2858e695f0063112d94771a2cbc611c734d1110983d334`。主表列顺序严格等于契约中的 `metadata_columns` 后接 `training_feature_columns`。

特征计算函数只接收路线、计划速度、控制向量、静态左右边界余量和车辆结构信息。它不接收硬证书结果、失败原因、报告路径、构建批次、运行后动态净空、最大滑移率、最大轮胎利用率或其他结果字段。未训练模型。

## 2. 结果统计

| 项目 | 数量 |
|---|---:|
| 扫描 NPZ 文件 | 91 |
| 唯一文件哈希 | 62 |
| 有效路线特征记录 | 55 |
| V3 记录 | 24 |
| V3 精确命中特征记录 | 1 |
| 未命中 V3 的唯一候选哈希 | 54 |
| 损坏文件 | 1 |
| 非路线 NPZ 排除 | 6 |
| 数值特征缺失记录 | 0 |
| 含回退或派生说明的特征记录 | 22 |
| 回退或派生说明总项数 | 43 |
| 五轴记录 | 27 |
| 六轴记录 | 27 |
| 未知轴型记录 | 1 |
| 通过标签 | 0 |
| 失败标签 | 1 |
| 由既有证书原因识别的局部风险覆盖 | 0 |

覆盖的地图类别为：cliff, cross, narrow/overpass, other, ramp, turnaround。每条记录保留 `map_id` 和 `route_id`；后续切分应按 `route_id` 或地图组留出，禁止把同一路线组随机拆入训练集和测试集。

## 3. 特征定义

- 路线：长度、点数、采样间距均值/中位数/95 分位数。
- 曲率：有符号均值、绝对值均值/最大值/50、90、95、99 分位数及单位距离变化率。
- 坡度：由 `z_m` 对 `s_m` 求导，并统计坡度及单位距离坡度变化。
- 速度：均值、最大值及 50、90、95 分位数。
- 速度曲率耦合：`v·|kappa|`、`v²·|kappa|` 及速度与绝对曲率相关系数；`v²·|kappa|` 仅为执行前几何-速度耦合代理量，不是运行后测量指标。
- 转向：优先读取 `steer_ff_rad`；缺失时使用执行前 `beta_ref_rad`，并在质量表记录回退。多维转向取同一点各维绝对值最大值，再统计幅值与单位距离变化率。
- 横摆率：优先读取 `yaw_rate_ref_radps`；缺失时由 `yaw_per_m_ref_1pm·v_profile_mps` 推导，再统计幅值和单位距离变化。
- 局部窗口：固定 10 m 后向空间窗口，统计窗口均值的最大绝对曲率、最大转向变化和最小左右静态边界余量。
- 车辆结构：`vehicle_structure` 仅使用 `five_axis`、`six_axis`、`unknown`，证据来源单列记录。`vehicle_axis_count` 和配置指纹摘要属于元数据。`steering_source_dim` 仅描述实际参与计算的转向数组末维，不代表车辆轴数。

所有数值特征均使用字段名中的单位；生成过程拒绝 NaN 和无穷值。`feature_missing_reason` 记录允许的执行前回退或派生方式，不使用结果字段补值。

## 4. 覆盖限制

当前本机只精确命中 1 条 V3 记录，其中通过 0 条、失败 1 条、局部风险 0 条。因此无法同时达到“五轴和六轴的通过、失败、局部风险各至少 5 条”。本次只保留 V3 原标签；未命中候选的 `true_label` 均为空，不根据路线名、目录名或测试结果补标签。

仓库仍未发现明确命名为“数据冻结 V3”的文件。本次沿用 Gate 2 已采用的提交 `d423f0e` 映射，仍待数据负责人确认。

## 5. 输出与数据边界

- `data/derived_features/ZYJisBoss/pre_execution_features.csv`：仅 V3 命中的主建模候选表。
- `data/derived_features/ZYJisBoss/feature_quality.csv`：哈希、可读性、有限值和回退说明。
- `data/derived_features/ZYJisBoss/unmatched_candidates.csv`：未命中 V3 的候选，不进入主表。
- `data/derived_features/ZYJisBoss/damaged_files.csv`：损坏或无法提取记录。
- `data/derived_features/feature_contract_v2.json`：全项目唯一 v2 特征契约。
- `reports/data_recovery/ZYJisBoss/gate2_feature_extraction.md`：本报告。

未提交原始 NPZ、本机绝对路径、报告路径、地图文件名或构建批次名。
