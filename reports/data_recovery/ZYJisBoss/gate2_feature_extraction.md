# Gate 2 执行前特征提取报告（ZYJisBoss）

## 1. 结果

本机三个已知 PathGuard 工程根目录共发现 91 个 NPZ，去重后为 62 个文件哈希。其中 6 个是校准、模型或辅助数据，不是路线样本；1 个无法读取；其余 55 个有效路线样本均完成执行前特征提取。

V3 以 `route_file_sha256` 为第一匹配键，以 `sample_id` 为第二匹配键。当前本机只有 1 个路线哈希命中 V3，因此主建模表只有 1 条；54 条未命中路线完整保留在 `unmatched_candidates.csv`，没有并入主建模表。

## 2. 数量与分布

| 项目 | 数量 |
|---|---:|
| 本机 NPZ 文件 | 91 |
| 唯一文件哈希 | 62 |
| V3 记录 | 24 |
| 主特征表有效样本 | 1 |
| 未命中候选 | 54 |
| 损坏文件 | 1 |
| 非路线 NPZ 排除 | 6 |
| 特征缺失记录 | 0 |

| 车型结构 | 全部有效路线 | V3 命中 |
|---|---:|---:|
| five_axis_explicit | 21 | 1 |
| six_axis_structural | 34 | 0 |
| unknown_or_conflict | 0 | 0 |

地图/路线组覆盖：ramp 8、cliff 8、cross 8、turnaround 11、narrow 4、overpass 4、other 12。`map_id` 和 `route_id` 仅用于按地图或路线分组留出，不属于数值输入特征。

当前主表只包含 1 条五轴失败样本。五轴/六轴的通过、失败、局部风险各 5 条覆盖目标受本机 V3 哈希命中数量限制，尚未满足；未命中样本没有依据名称或指标被补写标签。

## 3. 对齐与标签边界

- 主表只接受 V3 的精确 `route_file_sha256` 命中；sample_id 仅在哈希不可用时作为第二匹配键。
- 本机路线文件没有携带可用于第二键对齐的 V3 sample_id，因此本轮实际命中均由第一键完成；未使用路线名称替代 sample_id。
- `true_label` 仅从 V3 清单复制，未读取 NPZ 中的结果字段，也未修改通过/失败结果。
- `hard_certificate_passed`、`hard_failure_reasons`、动态净空、最大滑移率、最大轮胎利用率、报告路径、地图文件名和构建批次名均不作为输入特征。
- `map_id`、`route_id`、`label_status` 和 `true_label` 是对齐/分组/监督元数据，训练前必须从输入矩阵剔除。

## 4. 特征定义与单位

| 特征组 | 定义 |
|---|---|
| 路线几何 | `s_m` 的总长度、点数，以及相邻 `s_m` 的均值和分位数；单位 m。 |
| 曲率 | `kappa_1pm` 的均值、绝对最大值和分位数；曲率变化优先使用 `dkappa_ds_1pm2`，单位 1/m、1/m²。 |
| 坡度 | 相邻 z 差除以 s 差；坡度变化按中点距离求差分，单位为比值和 1/m。 |
| 速度 | `v_profile_mps` 的均值、最大值和分位数；单位 m/s。 |
| 速度—曲率耦合 | `abs(v_profile_mps * kappa_1pm)` 及速度绝对值与曲率绝对值相关系数。 |
| 转向前馈 | 优先 `steer_ff_rad`，其次 `beta_steer_ff_rad`，最后 `beta_ref_rad`；统计幅值和按距离变化率。 |
| 横摆 | 优先 NPZ 的 `yaw_rate_ref_radps` / `yaw_accel_ref_radps2`；缺失时由执行前速度、曲率和采样间距推导。 |
| 局部窗口 | 20 m 固定分块内的最大曲率和转向幅值峰峰值；静态左右边界余量取路线最小值。 |
| 车辆结构 | 五轴/六轴结构、转向向量维度，以及 NPZ 配置指纹前 16 位；缺失指纹时使用结构摘要哈希。 |

所有数值列均经过有限值检查；输出不含 NaN 或无穷值。`feature_quality.csv` 记录每条路线的匹配方法、可读性、有限值、缺失数量和本机副本数量。

## 5. 输出与数据边界

- `data/derived_features/ZYJisBoss/pre_execution_features.csv`
- `data/derived_features/ZYJisBoss/feature_quality.csv`
- `data/derived_features/ZYJisBoss/unmatched_candidates.csv`
- `data/derived_features/ZYJisBoss/damaged_files.csv`

主仓库只提交脱敏特征、哈希、字段说明和本报告，不提交原始 NPZ。没有训练模型，也没有把运行后指标写入输入特征。
