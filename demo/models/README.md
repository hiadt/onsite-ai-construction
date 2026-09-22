# Gate 3 模型待补

请由 Gate 3 负责人提供经过冻结和验证的模型文件：

`demo/models/pathguard_gate3_model.joblib`

模型必须提供 `predict_proba` 和 `classes_`。标签定义为 `1=pass, 0=fail`，应用读取类别 `0` 的概率作为失败风险。模型训练列及顺序必须与 `demo/data/feature_contract_v2.json` 的 `training_feature_columns` 完全一致。

本目录不得放置临时训练模型、原始数据、账号凭据或未经确认的模型文件。
