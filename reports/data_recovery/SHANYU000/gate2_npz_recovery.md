# Gate 2 NPZ 原始样本回查报告（SHANYU000）

## 1. 任务边界

本次仅对本机已有 PathGuard 原始 NPZ 进行哈希匹配、可读性检查、数组名称与形状提取，以及代表样本清单冻结。未训练模型、未修改标签，未把运行后的指标写入输入特征，也未向仓库复制原始 NPZ、工程源码、地图源码、控制器源码、模型文件或压缩备份。

仓库中未发现文件名或正文明确标注为“数据冻结 V3”的清单。本次将既有正式冻结提交 `d423f0e` 中的 24 条 `manifest.csv` 作为 Gate 2 的 V3 输入映射；该版本映射需要由数据负责人确认。所有标签、硬证书结果与报告哈希均原样继承该清单。

## 2. 扫描与验证结果

| 项目 | 数量 |
|---|---:|
| 冻结记录扫描 | 24 |
| 本机找到对应文件 | 24 |
| SHA-256 一致且 NPZ 可读取 | 24 |
| 损坏或缺失记录 | 0 |
| 新增代表样本清单记录 | 6 |
| 覆盖地图 | 6 |

NPZ 使用 `numpy.load(..., allow_pickle=False)` 打开，并逐一读取全部数组以确认结构可访问；`file_sha256` 与冻结清单中的 `route_file_sha256` 逐条比较。数组名称和形状已写入 `npz_manifest.csv`。

## 3. 代表样本覆盖

| 车型结构 | 通过 | 失败 | 局部风险 | 合计 |
|---|---:|---:|---:|---:|
| five_axis_explicit | 1 | 1 | 1 | 3 |
| six_axis_structural | 1 | 1 | 1 | 3 |
| 合计 | 2 | 2 | 2 | 6 |

代表样本共包含 2 条硬证书通过记录和 4 条硬证书失败记录。局部风险覆盖角色依据冻结清单已有的 `closed_loop_pcd_clearance`、`closed_loop_txt_clearance` 或 `closed_loop_heightmap_footprint` 失败原因确定，仅用于挑选代表样本，不构成新标签。

## 4. 受控存储说明

仓库清单中的 `controlled_storage_reference` 是供后续受控传输使用的逻辑标识，不是本机绝对路径，也不表示文件已上传到受控存储。6 个已验证的原始 NPZ 继续保留在源机器上，状态统一记录为 `verified_local_pending_controlled_transfer`。后续如需主线提取特征，应由数据负责人按清单哈希将文件传入获批的受控数据区，并在传输后再次核验 SHA-256。

主仓库本次只提交脱敏清单、哈希、数组名称与形状、损坏记录及本报告，不提交原始 NPZ。

## 5. 输出文件

- `data/recovered_npz/SHANYU000/npz_manifest.csv`
- `data/recovered_npz/SHANYU000/damaged_files.csv`
- `reports/data_recovery/SHANYU000/gate2_npz_recovery.md`

## 6. 待确认事项

- 待数据负责人确认提交 `d423f0e` 的 24 条冻结清单是否即项目所称“数据冻结 V3”。
- 待确定获批的受控原始数据存储位置；在此之前不执行 NPZ 传输。
