"""PathGuard candidate-route risk triage demonstration."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from demo_logic import apply_filters, evaluate_candidates, select_top_k
from feature_explain import label_text
from inference import FeatureValidationError, load_contract


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models" / "pathguard_gate3_model.joblib"
CONTRACT_PATH = DATA_DIR / "feature_contract_v2.json"
SAMPLE_PATH = DATA_DIR / "sample_input.csv"
SUMMARY_PATH = DATA_DIR / "demo_summary.json"

st.set_page_config(page_title="PathGuard 路线风险决策", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background: #f5f8fb;}
    [data-testid="stHeader"] {background: rgba(245,248,251,.92);}
    .hero {padding: 1.25rem 1.4rem; border-radius: 18px; color: white;
      background: linear-gradient(120deg,#123b5d,#0d7180); margin-bottom: 1rem;}
    .hero h1 {margin: 0; font-size: 2rem;} .hero p {margin: .45rem 0 0; opacity: .92;}
    .notice {padding: .8rem 1rem; border-left: 4px solid #e28a27; background: #fff7ea;
      border-radius: 8px; color: #69420e; margin: .4rem 0 1rem;}
    .vehicle-card {padding: .7rem 1rem; background:white; border:1px solid #dce7ef;
      border-radius:12px; min-height:108px;}
    .axles {display:flex; gap:7px; margin-top:12px}.axle {height:28px;width:9px;
      border-radius:4px;background:#157a8a;border:2px solid #0c4b61;}
    .small-note {font-size:.84rem;color:#526675;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_static_assets() -> tuple[dict, dict, pd.DataFrame]:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    contract = load_contract(CONTRACT_PATH)
    sample = pd.read_csv(SAMPLE_PATH, dtype={"route_file_sha256": str})
    return summary, contract, sample


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.metric(label, value, help=help_text or None)


def vehicle_illustration(title: str, count: int, caption: str) -> None:
    axles = "".join('<span class="axle"></span>' for _ in range(count))
    st.markdown(
        f'<div class="vehicle-card"><strong>{title}</strong><div class="axles">{axles}</div>'
        f'<div class="small-note">{caption}</div></div>',
        unsafe_allow_html=True,
    )


summary, contract, sample_frame = load_static_assets()
st.markdown(
    f'<section class="hero"><h1>PathGuard</h1><p>{summary["value_proposition"]}</p></section>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="notice">离线研发决策演示：用于候选轨迹风险排序、原因解释与验证优先级安排；'
    '不替代闭环仿真、实车测试或安全认证。</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("输入与筛选")
    upload = st.file_uploader("上传执行前特征 CSV", type=["csv"], help="必须包含契约规定的55项数值特征。")
    if upload is None:
        input_frame = sample_frame.copy()
        st.caption("当前使用：仓库内真实冻结特征示例")
    else:
        try:
            input_frame = pd.read_csv(upload)
            st.caption(f"已读取 {len(input_frame)} 条上传记录")
        except Exception as error:
            st.error(f"CSV读取失败：{error}")
            st.stop()
    structure_label = st.radio("车型结构", ["全部", "五轴", "六轴", "结构未定"], horizontal=True)
    structure_map = {"全部": "all", "五轴": "five_axis", "六轴": "six_axis", "结构未定": "unknown"}
    top_selection = st.selectbox("优先队列", ["Top 10", "Top 20", "Top 10%"])
    search = st.text_input("检索样本 / 路线 / 哈希")

try:
    evaluated, rule_components, runtime = evaluate_candidates(input_frame, contract, MODEL_PATH)
except (FeatureValidationError, ValueError) as error:
    st.error(f"输入未通过特征契约校验：{error}")
    st.stop()

if runtime["model_available"]:
    st.success(f'Gate 3 模型已加载：{runtime["model_name"]}')
else:
    st.warning(runtime["model_error"])
    st.caption("当前仅显示批次相对的几何规则预览；不生成模型分数、组合风险或高/中/低风险结论。")

active_structure = structure_map[structure_label]
structure_view = (
    evaluated
    if active_structure == "all"
    else evaluated[evaluated["vehicle_structure"] == active_structure]
)

overview_tab, queue_tab, detail_tab, evidence_tab = st.tabs(["项目总览", "风险队列", "路线详情", "验证依据"])

with overview_tab:
    snapshot = summary["data_snapshot"]
    cols = st.columns(4)
    with cols[0]: metric_card("监督学习最终样本", f'{snapshot["supervised_final"]} 条')
    with cols[1]: metric_card("已提取执行前特征", f'{snapshot["features_extracted"]} 条')
    with cols[2]: metric_card("无标签 / 指标池", f'{snapshot["unlabeled_metric_pool"]} 条')
    with cols[3]: metric_card("NPZ/V3 SHA 精确命中", f'{snapshot["npz_v3_sha256_exact_matches"]} 条')
    cols = st.columns(4)
    with cols[0]: metric_card("五轴 final", str(snapshot["final_five_axis"]))
    with cols[1]: metric_card("六轴 final", str(snapshot["final_six_axis"]))
    with cols[2]: metric_card("结构未定 final", str(snapshot["final_unknown_structure"]))
    with cols[3]: metric_card("执行前数值特征", f'{summary["feature_count"]} 项')
    st.markdown(
        "**产品闭环：** 候选轨迹输入　→　AI风险排序　→　工程规则校验　→　原因解释　→　补测 / 仿真复核 / 客户共创"
    )
    model_status = runtime["model_name"] if runtime["model_available"] else "Gate 3 模型待提供"
    st.caption(f"当前模型：{model_status}；无正式模型时仅提供透明规则预览，不形成风险等级结论。")
    st.subheader("车型结构入口")
    vehicle_cols = st.columns(3)
    with vehicle_cols[0]: vehicle_illustration("五轴", 5, "轴型来自冻结元数据；图示不表达尺寸或载荷。")
    with vehicle_cols[1]: vehicle_illustration("六轴", 6, "转向数组维度不用于推断车辆轴数。")
    with vehicle_cols[2]: vehicle_illustration("结构未定", 3, "需补充结构证据后再进入分结构决策。")
    labeled_pass = structure_view["true_label"].astype(str).isin(["1", "1.0"]).sum()
    labeled_fail = structure_view["true_label"].astype(str).isin(["0", "0.0"]).sum()
    st.caption(
        f"当前结构筛选：{structure_label} · 输入样本 {len(structure_view)} 条 · "
        f"通过标签 {labeled_pass} 条 · 失败标签 {labeled_fail} 条"
    )
    st.info(
        f'特征标准：`{summary["feature_version"]}` · 55项执行前数值特征 · 10 m局部窗口 · '
        f'契约哈希 `{summary["schema_sha256"]}`'
    )

map_options = ["全部"] + sorted(evaluated["map_id"].astype(str).unique().tolist())
with queue_tab:
    filter_cols = st.columns(2)
    with filter_cols[0]: risk_filter = st.selectbox("风险等级", ["全部", "高风险", "中风险", "低风险", "模型待补"])
    with filter_cols[1]: map_filter = st.selectbox("地图", map_options)
    filtered = apply_filters(evaluated, structure_map[structure_label], risk_filter, map_filter, search)
    queue = select_top_k(filtered, top_selection, runtime["model_available"]).reset_index(drop=True)
    score_title = "组合风险" if runtime["model_available"] else "规则预览"
    display = queue[["sample_id", "map_id", "route_id", "vehicle_structure", "true_label", "risk_level", "model_risk", "rule_risk", "combined_risk", "risk_reasons", "next_action"]].copy()
    display["true_label"] = display["true_label"].map(label_text)
    display = display.rename(columns={
        "sample_id": "样本", "map_id": "地图", "route_id": "路线", "vehicle_structure": "结构",
        "true_label": "冻结标签", "risk_level": "风险等级", "model_risk": "模型风险",
        "rule_risk": "规则预览", "combined_risk": "组合风险", "risk_reasons": "主要原因", "next_action": "下一步动作",
    })
    st.caption(f"按{score_title}降序展示 {len(display)} 条；规则分数是当前输入批次内的相对分位，不是安全概率。")
    queue_event = st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="risk_queue_table",
    )
    selected_rows = queue_event.selection.rows
    if selected_rows:
        st.session_state["selected_sample_id"] = str(queue.iloc[selected_rows[0]]["sample_id"])
        st.caption("已选中该路线；切换到“路线详情”可查看解释卡。")

with detail_tab:
    if structure_view.empty:
        st.info("当前结构筛选下没有候选路线；请切换车型结构或上传对应数据。")
        st.stop()
    candidate_ids = structure_view["sample_id"].astype(str).tolist()
    preferred_id = st.session_state.get("selected_sample_id", candidate_ids[0])
    preferred_index = candidate_ids.index(preferred_id) if preferred_id in candidate_ids else 0
    selected_id = st.selectbox("选择候选路线", candidate_ids, index=preferred_index)
    selected_index = evaluated.index[evaluated["sample_id"].astype(str) == selected_id][0]
    row = evaluated.loc[selected_index]
    components = rule_components.loc[selected_index].sort_values(ascending=False)
    left, right = st.columns([1, 1])
    with left:
        st.subheader(selected_id)
        st.write(f'地图 / 路线：`{row["map_id"]}` / `{row["route_id"]}`')
        st.write(f'车型结构：`{row["vehicle_structure"]}`；冻结标签：{label_text(row["true_label"])}')
        st.write(f'证据状态：{row["evidence_status"]}')
        st.write(f'建议动作：**{row["next_action"]}**')
        st.write(f'主要原因：{row["risk_reasons"]}')
    with right:
        if runtime["model_available"]:
            st.metric("组合风险", f'{float(row["combined_risk"]):.3f}', row["risk_level"])
            st.caption(f'0.6 × 模型风险 {float(row["model_risk"]):.3f} + 0.4 × 规则风险 {float(row["rule_risk"]):.3f}')
        else:
            st.metric("规则预览", f'{float(row["rule_risk"]):.3f}', "模型待补")
            st.caption("组合风险不计算；规则预览只用于演示透明排序逻辑。")
        st.bar_chart(components, horizontal=True)
    key_metrics = pd.DataFrame(
        [
            ("曲率绝对值 P95", row["curvature_abs_p95_1pm"], "1/m"),
            ("曲率变化 P95", row["curvature_change_abs_p95_1pm2"], "1/m²"),
            ("速度-曲率乘积 P95", row["speed_abs_curvature_product_p95_mpspm"], "1/s"),
            ("横向加速度代理 P95", row["lateral_accel_proxy_p95_mps2"], "m/s²"),
            ("转向变化 P95", row["steer_change_abs_p95_radpm"], "rad/m"),
            ("横摆率变化 P95", row["yaw_rate_change_abs_p95_radps_per_m"], "rad/(s·m)"),
            ("坡度变化 P95", row["slope_change_abs_p95_1pm"], "1/m"),
            ("最小静态边界余量", row["static_boundary_clearance_min_m"], "m"),
            ("10 m窗口最大平均曲率", row["window10m_max_mean_abs_curvature_1pm"], "1/m"),
            ("10 m窗口最大平均转向变化", row["window10m_max_mean_abs_steer_change_radpm"], "rad/m"),
            ("控制向量维度", row["steering_source_dim"], "按样本字段读取"),
        ],
        columns=["关键执行前指标", "数值", "单位"],
    )
    st.dataframe(key_metrics, width="stretch", hide_index=True)
    with st.expander("溯源元数据"):
        st.json({
            "route_file_sha256": str(row["route_file_sha256"]),
            "match_status": str(row["match_status"]),
            "feature_version": str(row["feature_version"]),
            "feature_source": str(row["feature_source"]),
            "vehicle_structure_evidence": str(row["vehicle_structure_evidence"]),
        })

with evidence_tab:
    st.subheader("评分与证据边界")
    st.write(f'学习模型类型：**{runtime["model_name"] if runtime["model_available"] else "待 Gate 3 模型补充"}**')
    st.markdown(
        "- **模型风险**：仅在正式 Gate 3 模型通过55维特征契约校验后输出，类别0表示失败。\n"
        "- **规则风险**：六组执行前几何与控制应力的批次相对分位均值，便于解释，不是监督模型或安全概率。\n"
        "- **组合风险**：演示配置为 `0.6 × 模型风险 + 0.4 × 规则风险`；高风险 ≥ 0.70，中风险 [0.45, 0.70)，低风险 < 0.45。\n"
        "- **输入边界**：不读取证书结果、失败原因、运行后动态净空、滑移率、轮胎利用率、报告路径或构建批次。"
    )
    validation = summary["validation"]
    if validation["status"] != "available":
        st.warning("独立地图 PR-AUC、ROC-AUC 与 Top-K 失败捕获率：待 Gate 3 验证产物补充。")
    st.code(
        f'feature_version = {summary["feature_version"]}\n'
        f'schema_sha256 = {summary["schema_sha256"]}\n'
        f'model_path = demo/models/pathguard_gate3_model.joblib',
        language="text",
    )
    st.caption(summary["disclaimer"])
