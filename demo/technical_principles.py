"""Optional technical explanation, separate from the customer-facing workflow."""
from pathlib import Path


def render_technical_principles(st, repo_root: Path) -> None:
    assets = repo_root / "docs/final/assets_v5"

    def figure(name: str, caption: str) -> None:
        path = assets / f"{name}.png"
        if path.is_file():
            st.image(str(path), caption=caption, width="stretch")
        else:
            st.caption("技术图随V5项目书发布；当前副本尚未包含该图。")

    with st.expander("算法原理与系统架构", expanded=False):
        st.caption("面向技术评审的实现说明。各通道分别给出统计、几何和执行依据，不相加成一个未经标定的概率。")
        architecture, learning, geometry, governance = st.tabs(
            ["系统数据流", "学习模型与规则", "空间与执行公式", "验证与实验协议"]
        )
        with architecture:
            st.markdown("#### 从文件到工程验证任务")
            st.write("执行前路线进入学习、规则和空间通道；执行后的遥测进入偏差重算通道。结果关联到同一任务和版本，再追加验证记录。")
            figure("fig_system_detail", "当前模块、输入输出及离线迭代路径")
        with learning:
            st.markdown("#### 55维特征如何变成排序分值")
            st.write("曲率、坡度、计划速度、转向、横摆、静态距离和10米窗口统计构成固定输入。模型是直方图梯度提升树，逐轮学习历史通过与失败之间的非线性关系。")
            st.latex(r"F_T(x)=F_0+\eta\sum_{t=1}^{T}h_t(x),\qquad r(x)=1-\frac{1}{1+e^{-F_T(x)}}")
            st.caption("训练标签1为通过、0为失败；F对应通过类别。当前工件：250轮、学习率0.04、最多15叶、最小叶样本20、L2系数1。分值用于排序，不是现场事故概率。")
            st.markdown("#### 六组规则如何解释工程关注点")
            st.latex(r"\widehat F_f^{mid}(x)=\frac{\#(v<x)+\frac12\#(v=x)}{274},\quad R=\frac16\sum_{g=1}^{6}\frac{1}{|J_g|}\sum_{f\in J_g}s_f(x_f)")
            st.write("高值敏感指标取分位数，低值敏感的距离指标取其补数；先组内平均，再六组等权。参考分布固定，因此不会因本次上传数量变化而重新定义同一路线的应力。")
            st.caption("规则原因是独立工程提示，不是SHAP贡献。当前combined_risk字段沿用模型分值，没有启用模型与规则加权混合。")
            figure("fig_queue_detail", "普通验证预算与预算外事项分开组织")
        with geometry:
            st.markdown("#### 尺寸通过坐标变换影响外廓")
            st.latex(r"q_{ij}=p_i+R(\psi_i)c_j,\qquad c_j\in\{(-a,\pm W/2),(L-a,\pm W/2)\}")
            st.write("L是车长，W是车宽，a是参考点距车尾距离。调整尺寸会改变角点与周界位置；当前55维模型没有独立尺寸输入，因此不会随尺寸滑块自动改变模型分值。")
            st.markdown("#### 有可信边界时才计算局部余量")
            st.latex(r"\widetilde s_{ij}=\mathrm{clip}(s_i+c_{jx},s_0,s_{N-1}),\quad m_{ij}^{L}=d^L-u^L\cdot(q_{ij}-p(\widetilde s_{ij}))")
            st.caption("完整点列通过距离、方向、次序与连续性筛查后，16个车身周界点作局部截面试算。它是离散近似，不等同于连续碰撞检测。")
            st.markdown("#### 为什么长车对方向偏差更敏感")
            st.latex(r"e_y=(p_{act}-p_{ref})\cdot n_{ref},\qquad n_c=e_y+x_c\sin e_\psi+y_c\cos e_\psi")
            st.write("横向偏差平移整辆车；航向偏差通过前后端纵向距离放大角点占用。参考点居中的18.8米矩形，5度偏差对应的纵向力臂投影约0.819米；这是机制示例，不是实际道路损失的净空。")
            figure("fig_geometry_detail", "刚体变换、车身参考点与局部截面关系")
        with governance:
            st.markdown("#### 历史评估与当前部署各自回答什么")
            st.write("历史留出：274条开发、66条测试，按地图隔离。开发集内比较模型，并在开发子集校准；当前部署则在340条冻结标签上重训HGB，实际工件没有包裹校准器。")
            st.caption("当前历史样本的重新评分属于回放，不能替代原留出预测；历史校准阈值也不能直接视为当前部署的可靠安全阈值。")
            figure("fig_training_detail", "历史留出实验与全量部署工件的不同来源")
            st.latex(r"P@K=\frac{F_K}{K},\qquad Capture@K=\frac{F_K}{F},\qquad E[F_{random}]=\frac{KF}{N}")
            st.write("历史Top-10发现8条失败，表示选中路线中80%为失败，同时只覆盖全部39条失败的20.5%。代码中的pr_auc_failure列实际使用平均精确率AP，不能混同为梯形PR曲线积分。")
            figure("fig_evidence_detail", "同条件核对、冲突保留与训练候选审核")
            st.caption("新记录不会自动训练模型。标签明确、条件一致、已核对且没有同条件冲突后，才允许人工设置训练候选状态。")
        book = repo_root / "docs/final/PathGuard项目书正式版_v5.docx"
        if book.is_file():
            st.download_button("下载完整技术项目书 Word", book.read_bytes(),
                               file_name="PathGuard项目书_技术深化版_v5.docx",
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                               key="technical_book_v5")
