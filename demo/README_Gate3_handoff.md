# Gate 3 前端交付包

本目录包含前端演示所需的受控派生模型、特征契约、演示样本和验证指标。

## 文件

- `models/pathguard_gate3_model.joblib`：使用冻结final数据训练的演示模型。
- `data/feature_contract_v2.json`：55维特征名称、顺序和Schema。
- `data/sample_input.csv`：包含五轴、六轴、结构未定、通过和失败样本的演示输入。
- `data/gate3_test_metrics.json`：独立地图测试结果。
- `data/gate3_oof_metrics.json`：全量地图分组OOF结果。
- `data/repeated_group_hybrid_summary.json`：60次规则/模型审计结果。
- `data/failure_mechanism_cards.json`：独立测试集解释样本。

## 使用边界

这些是派生模型和脱敏演示数据，不包含原始NPZ、路线坐标、本机路径、密码或令牌。演示结果用于候选轨迹风险排序，不代表实车安全认证。

## 前端要求

前端必须校验模型的特征版本和Schema，模型缺失时明确显示规则预览状态，不得把规则预览伪装成完整AI模型结果。
