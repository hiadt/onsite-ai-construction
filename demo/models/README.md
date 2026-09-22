# Gate 3 演示模型

冻结并验证的演示模型位于：

`demo/models/pathguard_gate3_model.joblib`

模型为 `HistGradientBoostingClassifier`，由 `scikit-learn 1.0.2` 固化。运行环境必须使用兼容依赖，详见 `demo/requirements.txt`。模型提供 `predict_proba` 和 `classes_`；标签定义为 `1=pass, 0=fail`，应用读取类别 `0` 的概率作为失败风险。

应用启动时校验模型记录的55项特征名称与顺序、`feature_version` 和 `schema_sha256`，全部必须与 `demo/data/feature_contract_v2.json` 一致。校验或加载失败时只允许显示规则预览。

本目录不得放置临时训练模型、原始数据、账号凭据或未经确认的替代模型。
