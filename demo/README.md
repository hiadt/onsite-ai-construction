# PathGuard 一体化 Streamlit 演示

PathGuard 是工程车辆候选轨迹的离线风险排序与可信拒判原型。统一页面完成车辆结构筛选、候选特征输入、Gate 3模型推理、几何规则校验、风险解释和Top-K验证队列生成。系统不直接控制车辆，不构成安全认证。

## 安装与启动

Gate 3模型由 `scikit-learn 1.0.2` 固化，建议使用 Python 3.10 创建独立环境：

```powershell
py -3.10 -m venv .venv-demo
.\.venv-demo\Scripts\python.exe -m pip install -r demo\requirements.txt
.\.venv-demo\Scripts\python.exe -m streamlit run demo\app.py
```

产品入口只有：

```bash
streamlit run demo/app.py
```

## 输入格式

默认输入为 `demo/data/sample_input.csv`。上传CSV必须包含 `demo/data/feature_contract_v2.json` 中按顺序定义的全部55项 `training_feature_columns`，并满足：

- 所有训练特征都能转换为有限数值；
- 不缺列，不静默填充数值特征；
- 如提供 `feature_version` 或 `schema_sha256`，必须与冻结契约一致；
- `sample_id`、`map_id`、`route_id`、`vehicle_structure` 和标签状态仅作元数据显示，不进入模型矩阵。

冻结契约：

- 特征版本：`pathguard_preexec_v2.0.0`
- 训练特征：55项
- 局部窗口：10米
- 标签定义：`1=pass, 0=fail`
- Schema SHA-256：`b365d5b00779cc9f1e2858e695f0063112d94771a2cbc611c734d1110983d334`

## 真实模型与风险计算

模型路径：

```text
demo/models/pathguard_gate3_model.joblib
```

启动时会同时校验：模型可反序列化、`predict_proba`接口、55维输入、模型记录的特征名称与顺序、特征版本和Schema哈希。失败标签固定为0，学习模型风险取失败类别概率。

模型加载失败时，页面明确显示“学习模型未加载，当前为规则预览”，不会把规则分数伪装成模型结果。

几何规则只使用冻结55维中的曲率、速度—曲率耦合、横向加速度代理、转向、横摆率、坡度、静态边界净空和10米局部窗口指标。综合风险为：

```text
综合风险 = 0.6 × 学习模型失败风险 + 0.4 × 几何规则风险
```

该权重是演示配置，不是生产标定参数。风险等级为：高风险 `>=0.70`，中风险 `[0.45,0.70)`，低风险 `<0.45`。

## 验证依据

页面从以下仓库文件动态读取结果，不在前端手工写入指标：

- `gate3_test_metrics.json`：独立地图测试；
- `gate3_oof_metrics.json`：全量地图分组OOF；
- `repeated_group_hybrid_summary.json`：60次规则/模型地图分组审计；
- `failure_mechanism_cards.json`：独立测试失效机理证据卡；
- `freeze_data_card.json`：冻结数据口径。

所有指标均为离线地图分组验证结果，不是实车验证结果。PathGuard只用于安排候选路线验证优先级，不能代替闭环仿真、人工复核或实车验证。
