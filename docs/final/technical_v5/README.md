# V5数学模型与架构材料

项目书第3至8章补充方法细节，附录B提供代码入口。本版27组公式采用Word原生数学对象，可编辑；五幅技术图同时提供PNG与SVG，位于同级的 `../assets_v5/`。

- `equations.json`：公式编号、内部标识和LaTeX源。
- `source_audit.json`：实际部署参数、代码与工件哈希、7项数值定义核查。
- `source_snapshots/`：原训练脚本，保留历史来源。
- `draw_diagrams.py`：图示绘制源，运行会写入当前目录的figures子目录。

在仓库根目录使用兼容冻结模型的Python环境：

```powershell
python scripts/audit_document_math_v5.py --output reports/final/math_definition_audit_v5.json
python -m pytest demo -q
```

数值核查使用合成小例子验证公式与实现一致，不是新增性能实验或新增训练样本。

前端入口：风险工作台 → 专业依据 → 算法原理与系统架构。按需展开，包含四个子页和完整Word下载。
