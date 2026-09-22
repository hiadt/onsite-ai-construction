"""PathGuard unified Streamlit demo for offline candidate-route risk triage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from demo_logic import apply_filters, evaluate_candidates, select_top_k
from feature_explain import label_text
from inference import FeatureValidationError, load_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "demo"
DATA_DIR = DEMO_DIR / "data"
MODEL_PATH = DEMO_DIR / "models" / "pathguard_gate3_model.joblib"
CONTRACT_PATH = DATA_DIR / "feature_contract_v2.json"
SAMPLE_PATH = DATA_DIR / "sample_input.csv"

st.set_page_config(page_title="PathGuard 路线风险决策", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background:#f5f8fb;}
    [data-testid="stHeader"] {background:rgba(245,248,251,.92);}
    .hero {padding:1.25rem 1.4rem;border-radius:18px;color:white;
      background:linear-gradient(120deg,#123b5d,#0d7180);margin-bottom:1rem;}
    .hero h1 {margin:0;font-size:2rem}.hero p {margin:.45rem 0 0;opacity:.92}
    .notice {padding:.8rem 1rem;border-left:4px solid #e28a27;background:#fff7ea;
      border-radius:8px;color:#69420e;margin:.4rem 0 1rem;}
    .vehicle-card {padding:.7rem 1rem;background:white;border:1px solid #dce7ef;
      border-radius:12px;min-height:112px;}
    .axles {display:flex;gap:7px;margin:12px 0 8px}.axle {height:28px;width:9px;
      border-radius:4px;background:#157a8a;border:2px solid #0c4b61;}
    .small-note {font-size:.84rem;color:#526675}.offline {color:#526675;font-size:.86rem;}
    .envelope-card {padding:1rem;background:white;border:1px solid #dce7ef;border-radius:14px;margin:.8rem 0 1rem;}
    .envelope-title {display:flex;justify-content:space-between;color:#173f59;margin-bottom:.35rem;}
    .envelope-title span {font-size:.82rem;color:#39718a;background:#e8f2f5;padding:.2rem .55rem;border-radius:999px;}
    .envelope-card svg {width:100%;height:auto;max-height:300px;background:#f7fafb;border-radius:10px;}
    .route-line {fill:none;stroke:#2b6f8d;stroke-width:3;stroke-dasharray:8 5;}
    .envelope-band {fill:none;opacity:.18;stroke-linecap:round;}
    .vehicle-body {fill:#eaf4f6;fill-opacity:.96;stroke:#123b5d;stroke-width:2;}
    .vehicle-cab {fill:#7fbec5;stroke:#123b5d;stroke-width:1.5;}
    .vehicle-nose {fill:#b9dfe1;stroke:#123b5d;stroke-width:1.5;}
    .axle-line {stroke:#2b7890;stroke-width:3;}
    .road-surface {fill:#dfe7eb;stroke:#9aabb5;stroke-width:1.2;}
    .road-edge {fill:none;stroke:#8c9ca5;stroke-width:2;stroke-dasharray:7 6;}
    .sweep-edge {fill:none;stroke:#cf5b45;stroke-width:1.6;stroke-dasharray:5 4;opacity:.72;}
    .svg-label {font-size:12px;fill:#526675;}
    .envelope-note {font-size:.78rem;color:#526675;margin-top:.45rem;line-height:1.45;}
    .section-kicker {font-size:.76rem;text-transform:uppercase;letter-spacing:.12em;color:#39718a;font-weight:700;margin:.2rem 0 .35rem;}
    .value-grid {display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem;margin:.5rem 0 1rem;}
    .value-card {background:#fff;border:1px solid #dce7ef;border-radius:14px;padding:1rem 1.05rem;min-height:128px;}
    .value-card h4 {margin:0 0 .4rem;color:#173f59;font-size:1rem;}
    .value-card p {margin:0;color:#526675;font-size:.88rem;line-height:1.55;}
    .flow-strip {display:flex;gap:.45rem;align-items:stretch;margin:.6rem 0 1rem;}
    .flow-step {flex:1;background:#edf5f6;border-radius:10px;padding:.7rem .75rem;color:#174d61;font-size:.87rem;}
    .flow-step b {display:block;color:#123b5d;margin-bottom:.2rem;}
    .judge-card {background:linear-gradient(120deg,#123b5d,#0d7180);color:#fff;border-radius:14px;padding:1rem 1.1rem;margin:.7rem 0 1rem;}
    .judge-card b {font-size:1.08rem;}.judge-card p {margin:.35rem 0 0;opacity:.92;line-height:1.5;font-size:.9rem;}
    .risk-summary {display:grid;grid-template-columns:1.15fr 1fr 1fr 1fr;gap:.55rem;margin:.65rem 0 .8rem;}
    .risk-box {background:#f7fafb;border:1px solid #dce7ef;border-radius:10px;padding:.65rem .75rem;min-height:76px;}
    .risk-box b {display:block;color:#173f59;font-size:.82rem;margin-bottom:.25rem;}
    .risk-box span {color:#304c5d;font-size:.86rem;line-height:1.4;}
    .risk-box.primary {background:#fff4ed;border-color:#efc4a9;}.risk-box.primary span{color:#a44c2f;font-weight:700;}
    .legend {display:flex;flex-wrap:wrap;gap:.65rem 1rem;margin:.45rem 0 0;color:#526675;font-size:.78rem;}
    .legend i {display:inline-block;width:22px;border-top:3px solid #2b6f8d;vertical-align:middle;margin-right:4px;}
    .legend i.dashed {border-top-style:dashed;}.legend i.band {border-top:7px solid #cf5b45;opacity:.55;}
    </style>
    """,
    unsafe_allow_html=True,
)


def read_json(name: str) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


@st.cache_data
def load_static_assets() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    assets: dict[str, Any] = {
        "freeze": read_json("freeze_data_card.json"),
        "test": read_json("gate3_test_metrics.json"),
        "oof": read_json("gate3_oof_metrics.json"),
        "audit": read_json("repeated_group_hybrid_summary.json"),
        "mechanisms": read_json("failure_mechanism_cards.json"),
    }
    contract = load_contract(CONTRACT_PATH)
    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
    return assets, contract, sample


def vehicle_illustration(title: str, count: int, caption: str) -> None:
    axles = "".join('<span class="axle"></span>' for _ in range(count))
    st.markdown(
        f'<div class="vehicle-card"><strong>{title}</strong><div class="axles">{axles}</div>'
        f'<div class="small-note">{caption}</div></div>',
        unsafe_allow_html=True,
    )


def dynamic_envelope(row: pd.Series) -> None:
    """Animate a normalized spatial-risk cue for the selected route."""
    structure = str(row.get("vehicle_structure", "unknown"))
    axle_count = 6 if structure == "six_axis" else 5
    signed_curvature = float(row.get("curvature_mean_1pm", 0.0))
    curvature = abs(float(row.get("curvature_abs_p95_1pm", 0.0)))
    steer_change = abs(float(row.get("steer_change_abs_p95_radpm", 0.0)))
    clearance = float(row.get("static_boundary_clearance_min_m", 0.0))
    caution = max(0.0, min(1.0, 1.0 - clearance / 4.0))
    bend = min(125.0, 24.0 + 520.0 * curvature + 45.0 * steer_change)
    direction = -1.0 if signed_curvature < 0 else 1.0
    bend *= direction
    envelope_width = 34.0 + 34.0 * caution
    body_length = 104.0 if axle_count == 6 else 92.0
    body_width = 30.0
    axle_lines = []
    for idx in range(axle_count):
        x = -body_length / 2 + 11.0 + idx * (body_length - 22.0) / max(1, axle_count - 1)
        axle_lines.append(
            f'<line x1="{x:.1f}" y1="{-body_width/2-3:.1f}" '
            f'x2="{x:.1f}" y2="{body_width/2+3:.1f}" class="axle-line"/>'
        )
    structure_label = "六轴" if axle_count == 6 else "五轴" if structure == "five_axis" else "结构未定"
    sample_token = "".join(ch for ch in str(row.get("sample_id", "route")) if ch.isalnum())[-12:]
    route_id = f"motion-route-{sample_token or 'selected'}"
    route_d = (
        f"M 38 218 C 150 218, 190 {218-bend:.1f}, 300 {168-bend/2:.1f} "
        f"S 470 {76+bend/4:.1f}, 565 54"
    )
    attention = "较高" if caution >= 0.65 else "中等" if caution >= 0.35 else "较低"
    risk_color = "#cf5b45" if caution >= 0.65 else "#e09a42" if caution >= 0.35 else "#4c9a82"
    combined = float(row.get("combined_risk", row.get("rule_risk", 0.0)))
    level = str(row.get("risk_level", "待判定"))
    if level in {"高风险", "高"} or combined >= .67:
        result = "预测为优先复核路线"
        consequence = "可能出现净空不足或轨迹执行偏差"
        action = str(row.get("next_action", "优先补测 / 人工复核"))
    elif level in {"中风险", "中"} or combined >= .34:
        result = "预测为需要关注路线"
        consequence = "局部指标接近关注阈值，需结合场景复核"
        action = str(row.get("next_action", "安排复核"))
    else:
        result = "预测为当前批次低优先级路线"
        consequence = "当前证据未显示明显风险，但不代表免检"
        action = str(row.get("next_action", "暂缓处理 / 按计划验证"))
    reasons = str(row.get("risk_reasons", "曲率、转向变化、净空等指标"))
    svg = f'''<div class="envelope-card">
      <div class="envelope-title"><strong>这条路线会发生什么？</strong><span>{structure_label} · 风险等级：{level}</span></div>
      <div class="risk-summary">
        <div class="risk-box primary"><b>预测结论</b><span>{result}</span></div>
        <div class="risk-box"><b>可能的工程情况</b><span>{consequence}</span></div>
        <div class="risk-box"><b>主要原因</b><span>{reasons}</span></div>
        <div class="risk-box"><b>建议动作</b><span>{action}</span></div>
      </div>
      <svg viewBox="0 0 600 290" role="img" aria-label="候选路线动态空间风险示意">
        <defs><marker id="pg-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#2b6f8d"/></marker></defs>
        <path d="{route_d}" class="envelope-band" style="stroke:{risk_color};stroke-width:{envelope_width:.1f}px"/>
        <path d="M 5 222 L 142 222 C 190 222, 217 {218-bend:.1f}, 300 {168-bend/2:.1f} S 474 {70+bend/4:.1f}, 595 47 L 595 5 L 5 5 Z" class="road-surface" opacity=".54"/>
        <path d="M 5 195 L 145 195 C 194 195, 220 {206-bend:.1f}, 302 {160-bend/2:.1f} S 477 {61+bend/4:.1f}, 595 34" class="road-edge"/>
        <path d="M 5 249 L 145 249 C 194 249, 220 {230-bend:.1f}, 302 {176-bend/2:.1f} S 477 {88+bend/4:.1f}, 595 60" class="road-edge"/>
        <path id="{route_id}" d="{route_d}" class="route-line" marker-end="url(#pg-arrow)"/>
        <circle r="7" fill="{risk_color}" opacity=".55"><animate attributeName="r" values="5;10;5" dur="1.4s" repeatCount="indefinite"/><animateMotion dur="6.5s" repeatCount="indefinite" rotate="auto"><mpath href="#{route_id}"/></animateMotion></circle>
        <g class="moving-vehicle">
          <rect x="{-body_length/2:.1f}" y="{-body_width/2:.1f}" width="{body_length:.1f}" height="{body_width:.1f}" rx="10" class="vehicle-body"/>
          <rect x="{body_length/2-38:.1f}" y="{-body_width/2+3:.1f}" width="24" height="{body_width-6:.1f}" rx="5" class="vehicle-cab"/>
          <path d="M {body_length/2-18:.1f} {-body_width/2:.1f} L {body_length/2:.1f} 0 L {body_length/2-18:.1f} {body_width/2:.1f} Z" class="vehicle-nose"/>
          {''.join(axle_lines)}
          <path d="M {-body_length/2-2:.1f} {-body_width/2-8:.1f} L {body_length/2+5:.1f} {-body_width/2-8:.1f}" class="sweep-edge"/>
          <path d="M {-body_length/2-2:.1f} {body_width/2+8:.1f} L {body_length/2+5:.1f} {body_width/2+8:.1f}" class="sweep-edge"/>
          <animateMotion dur="6.5s" repeatCount="indefinite" rotate="auto"><mpath href="#{route_id}"/></animateMotion>
        </g>
        <text x="18" y="257" class="svg-label">曲率P95 {curvature:.3f} · 转向变化P95 {steer_change:.3f} · 静态净空 {clearance:.2f}</text>
        <text x="18" y="275" class="svg-label">俯视道路、障碍物与车辆扫掠关注带；车辆沿当前特征生成的示意路径循环运动</text>
      </svg>
      <div class="legend"><span><i></i>蓝色虚线：车辆参考路线</span><span><i class="dashed"></i>灰色虚线：道路边界</span><span><i class="band"></i>彩色带：扫掠风险关注区</span></div>
      <div class="envelope-note">读图方法：车辆沿蓝色路线运动，彩色关注带越宽，表示当前样本的净空和转向风险越值得优先复核。本批演示数据没有统一的障碍物坐标，因此不绘制虚构障碍物；障碍物避碰需接入原始场景坐标后再计算。</div>
    </div>'''
    st.markdown(svg, unsafe_allow_html=True)


def percent(value: float) -> str:
    return f"{float(value):.1%}"


assets, contract, sample_frame = load_static_assets()
freeze = assets["freeze"]

st.markdown(
    '<section class="hero"><h1>PathGuard</h1><p>让每一条候选工程车辆路线，都有可解释的风险等级和下一步动作。</p></section>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="notice">离线风险排序与可信拒判原型：用于闭环仿真和实车验证前的候选路线排序、原因解释与验证资源安排；'
    '系统不直接控制车辆，不构成安全认证。</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("输入与筛选")
    upload = st.file_uploader(
        "上传执行前特征 CSV", type=["csv"], help="必须包含冻结契约规定的全部55项数值特征。"
    )
    if upload is None:
        input_frame = sample_frame.copy()
        st.caption("当前使用：仓库内离线派生演示输入")
    else:
        try:
            input_frame = pd.read_csv(upload, encoding="utf-8-sig")
            st.caption(f"已读取 {len(input_frame)} 条上传记录")
        except Exception as error:
            st.error(f"CSV读取失败：{error}")
            st.stop()
    structure_label = st.radio("车辆结构", ["全部", "五轴", "六轴", "结构未定"], horizontal=True)
    structure_map = {"全部": "all", "五轴": "five_axis", "六轴": "six_axis", "结构未定": "unknown"}
    top_selection = st.selectbox("优先验证队列", ["Top 10", "Top 20", "Top 34"])
    search = st.text_input("检索样本或路线编号")

try:
    evaluated, rule_components, runtime = evaluate_candidates(input_frame, contract, MODEL_PATH)
except (FeatureValidationError, ValueError) as error:
    st.error(f"输入未通过冻结特征契约校验：{error}")
    st.stop()

if runtime["model_available"]:
    st.success(f'学习模型已加载：{runtime["model_name"]}；55维特征、版本与Schema校验通过。')
else:
    st.warning("学习模型未加载，当前为规则预览。")
    st.caption(runtime["model_error"])

active_structure = structure_map[structure_label]
structure_view = (
    evaluated
    if active_structure == "all"
    else evaluated[evaluated["vehicle_structure"] == active_structure]
)

overview_tab, queue_tab, detail_tab, evidence_tab = st.tabs(
    ["项目总览", "风险队列", "路线详情", "验证依据"]
)

with overview_tab:
    st.markdown('<div class="section-kicker">先看懂产品</div><h2 style="margin:.1rem 0 .25rem;color:#173f59;">PathGuard 在替工程师决定：哪条路线应该先验证？</h2>', unsafe_allow_html=True)
    st.markdown(
        '<div class="value-grid">'
        '<div class="value-card"><h4>路线很多，验证资源有限</h4><p>把候选路线按风险和证据强弱排队，先处理最值得复核的路线。</p></div>'
        '<div class="value-card"><h4>结果要能解释</h4><p>不仅给出分数，还指出曲率、转向变化、净空等工程原因。</p></div>'
        '<div class="value-card"><h4>不确定性也要被看见</h4><p>区分真实标签、指标风险和代理排序，避免把数据不足误判成安全。</p></div>'
        '</div>', unsafe_allow_html=True,
    )
    st.markdown('<div class="flow-strip">'
        '<div class="flow-step"><b>① 输入候选路线</b>执行前特征与车辆结构</div>'
        '<div class="flow-step"><b>② AI + 规则排序</b>识别高风险候选</div>'
        '<div class="flow-step"><b>③ 解释风险原因</b>告诉工程师为什么</div>'
        '<div class="flow-step"><b>④ 安排下一步</b>补测、复核或共创</div>'
        '</div>', unsafe_allow_html=True)
    st.markdown('<div class="judge-card"><b>一句话结果</b><p>PathGuard 不替车辆做控制决策，而是把“先测哪条路线、为什么先测、下一步怎么验证”变成可追溯的工程队列。</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">数据与证据底座</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    cols[0].metric("SHA-256精确匹配", f'{freeze["sha256_exact_matches"]} 条')
    cols[1].metric("成功提取特征", f'{freeze["features_extracted"]} 条')
    cols[2].metric("final监督样本", f'{freeze["final_samples"]} 条')
    cols[3].metric("通过 / 失败", f'{freeze["passed"]} / {freeze["failed"]}')
    cols[4].metric("训练特征", f'{freeze["feature_count"]} 项')
    cols = st.columns(3)
    cols[0].metric("五轴 final", freeze["five_axis_final"])
    cols[1].metric("六轴 final", freeze["six_axis_final"])
    cols[2].metric("结构未定 final", freeze["unknown_structure_final"])
    model_status = runtime["model_name"] if runtime["model_available"] else "学习模型未加载"
    st.caption(f"当前状态：{model_status}；综合风险权重为演示配置，不是生产标定参数。")

    st.subheader("车辆结构")
    vehicle_cols = st.columns(3)
    with vehicle_cols[0]:
        vehicle_illustration("五轴", 5, "结构来自样本元数据；图示不表达尺寸、载荷或动力学参数。")
    with vehicle_cols[1]:
        vehicle_illustration("六轴", 6, "转向来源数组维度不用于替代车辆轴数证据。")
    with vehicle_cols[2]:
        vehicle_illustration("结构未定", 3, "结构未定，建议人工复核。")
    st.caption(f"当前结构筛选：{structure_label} · 演示输入 {len(structure_view)} 条")
    st.markdown('<div class="section-kicker">评委如何使用</div>', unsafe_allow_html=True)
    st.info("先进入“风险队列”看优先级，再点开一条路线看“风险原因—证据等级—下一步动作”；“验证依据”用于查看模型和离线分组结果。")
    st.info(
        f'特征版本 `{freeze["feature_version"]}` · Schema `{freeze["schema_sha256"]}` · '
        "55项执行前数值特征 · 10米局部窗口"
    )

map_options = ["全部"] + sorted(evaluated["map_id"].astype(str).unique().tolist())
with queue_tab:
    st.markdown('<div class="section-kicker">第二层：直接看决策结果</div><h2 style="margin:.1rem 0 .25rem;color:#173f59;">风险队列：先看哪条路线？</h2>', unsafe_allow_html=True)
    st.caption("这里的排序结果服务于验证资源安排：高风险不等于已经失败，低风险也不等于可以免检。")
    filter_cols = st.columns(2)
    risk_options = ["全部", "高风险", "中风险", "低风险"]
    if not runtime["model_available"]:
        risk_options.append("模型待补")
    with filter_cols[0]:
        risk_filter = st.selectbox("风险等级", risk_options)
    with filter_cols[1]:
        map_filter = st.selectbox("地图", map_options)
    filtered = apply_filters(evaluated, active_structure, risk_filter, map_filter, search)
    queue = select_top_k(filtered, top_selection, runtime["model_available"]).reset_index(drop=True)
    display_columns = [
        "sample_id", "map_id", "vehicle_structure", "model_risk", "rule_risk",
        "combined_risk", "risk_level", "validation_priority", "risk_reasons", "next_action",
    ]
    display = queue[display_columns].rename(columns={
        "sample_id": "样本编号", "map_id": "地图编号", "vehicle_structure": "车辆结构",
        "model_risk": "学习模型风险", "rule_risk": "规则风险", "combined_risk": "综合风险",
        "risk_level": "风险等级", "validation_priority": "验证优先级",
        "risk_reasons": "主要风险因素", "next_action": "下一步动作",
    })
    score_title = "综合风险" if runtime["model_available"] else "规则预览"
    st.caption(
        f"按{score_title}降序生成 {top_selection}，当前显示 {len(display)} 条；"
        "规则风险是当前候选批次内的相对应力，不是安全概率。"
    )
    queue_event = st.dataframe(
        display, width="stretch", hide_index=True, on_select="rerun",
        selection_mode="single-row", key="risk_queue_table",
    )
    if queue_event.selection.rows:
        selected_row = queue_event.selection.rows[0]
        st.session_state["selected_sample_id"] = str(queue.iloc[selected_row]["sample_id"])
        st.caption("已选中该路线；可切换到“路线详情”查看解释卡。")

with detail_tab:
    st.markdown('<div class="section-kicker">第三层：展开专业证据</div><h2 style="margin:.1rem 0 .25rem;color:#173f59;">路线解释卡：为什么需要优先复核？</h2>', unsafe_allow_html=True)
    if structure_view.empty:
        st.info("当前车辆结构筛选下没有候选路线；请切换结构或上传对应数据。")
    else:
        candidate_ids = structure_view["sample_id"].astype(str).tolist()
        preferred = st.session_state.get("selected_sample_id", candidate_ids[0])
        selected_id = st.selectbox(
            "选择候选路线", candidate_ids,
            index=candidate_ids.index(preferred) if preferred in candidate_ids else 0,
        )
        selected_index = evaluated.index[evaluated["sample_id"].astype(str) == selected_id][0]
        row = evaluated.loc[selected_index]
        components = rule_components.loc[selected_index].sort_values(ascending=False)
        dynamic_envelope(row)
        left, right = st.columns([1, 1])
        with left:
            st.subheader(selected_id)
            st.write(f'地图 / 路线：`{row["map_id"]}` / `{row["route_id"]}`')
            st.write(f'车辆结构：`{row["vehicle_structure"]}`；标签：{label_text(row["true_label"])}')
            st.write(f'证据状态：**{row["evidence_status"]}**')
            st.write(f'验证优先级：**{row["validation_priority"]}**')
            st.write(f'建议动作：**{row["next_action"]}**')
            st.write(f'主要风险因素：{row["risk_reasons"]}')
        with right:
            if runtime["model_available"]:
                st.metric("综合风险", f'{float(row["combined_risk"]):.3f}', row["risk_level"])
                st.caption(
                    f'0.6 × 学习模型风险 {float(row["model_risk"]):.3f} + '
                    f'0.4 × 规则风险 {float(row["rule_risk"]):.3f}'
                )
            else:
                st.metric("规则预览", f'{float(row["rule_risk"]):.3f}', "学习模型未加载")
            st.bar_chart(components, horizontal=True)
        key_metrics = pd.DataFrame([
            ("曲率绝对值 P95", row["curvature_abs_p95_1pm"], "1/m"),
            ("曲率变化 P95", row["curvature_change_abs_p95_1pm2"], "1/m²"),
            ("速度—曲率乘积 P95", row["speed_abs_curvature_product_p95_mpspm"], "1/s"),
            ("横向加速度代理 P95", row["lateral_accel_proxy_p95_mps2"], "m/s²"),
            ("转向幅值 P95", row["steer_abs_p95_rad"], "rad"),
            ("转向变化 P95", row["steer_change_abs_p95_radpm"], "rad/m"),
            ("横摆率变化 P95", row["yaw_rate_change_abs_p95_radps_per_m"], "rad/(s·m)"),
            ("坡度绝对值 P95", row["slope_abs_p95"], "无量纲"),
            ("最小静态边界净空", row["static_boundary_clearance_min_m"], "m"),
            ("10米窗口最大平均曲率", row["window10m_max_mean_abs_curvature_1pm"], "1/m"),
            ("10米窗口最大平均转向变化", row["window10m_max_mean_abs_steer_change_radpm"], "rad/m"),
        ], columns=["执行前指标", "数值", "单位"])
        st.dataframe(key_metrics, width="stretch", hide_index=True)

with evidence_tab:
    test = assets["test"]
    oof = assets["oof"]["overall"]
    audit = assets["audit"]
    st.subheader("离线地图分组验证")
    st.caption("以下结果均来自离线地图分组验证，不是实车验证结果，也不构成安全认证。")
    cols = st.columns(4)
    cols[0].metric("独立测试失败 PR-AUC", f'{test["pr_auc_failure"]:.4f}')
    cols[1].metric("独立测试失败 ROC-AUC", f'{test["roc_auc_failure"]:.4f}')
    cols[2].metric("全量OOF失败 PR-AUC", f'{oof["pr_auc_failure"]:.4f}')
    cols[3].metric("全量OOF失败 ROC-AUC", f'{oof["roc_auc_failure"]:.4f}')

    top_rows: list[dict[str, str | int]] = []
    for k in ("10", "20", "34"):
        test_item = test["top_k"][k]
        oof_item = assets["oof"]["top_k"][k]
        top_rows.append({
            "队列": f"Top-{k}",
            "独立测试捕获失败数": test_item["failures_captured"],
            "独立测试失败捕获率": percent(test_item["capture_rate"]),
            "全量OOF捕获失败数": oof_item["failures_captured"],
            "全量OOF失败捕获率": percent(oof_item["capture_rate"]),
        })
    st.dataframe(pd.DataFrame(top_rows), width="stretch", hide_index=True)
    st.caption("Top-K捕获率分母为对应离线验证集合中的失败样本数。")

    st.subheader("60次规则 / 模型地图分组审计")
    audit_rows = []
    for weight, item in audit["summary_by_model_weight"].items():
        audit_rows.append({
            "模型权重": float(weight),
            "失败PR-AUC均值": item["pr_auc_failure"]["mean"],
            "失败PR-AUC P10–P90": f'{item["pr_auc_failure"]["p10"]:.3f}–{item["pr_auc_failure"]["p90"]:.3f}',
            "Top-10捕获率均值": item["top_10_capture"]["mean"],
            "Top-20捕获率均值": item["top_20_capture"]["mean"],
            "Top-34捕获率均值": item["top_34_capture"]["mean"],
        })
    st.dataframe(pd.DataFrame(audit_rows), width="stretch", hide_index=True)
    st.caption(
        f'审计共 {audit["n_splits"]} 次地图分组；按平均失败PR-AUC观察到的最佳模型权重为 '
        f'{audit["best_weight_by_mean_pr_auc"]:.2f}。页面采用0.60仅作为产品演示配置。'
    )

    with st.expander("独立测试失效机理证据卡"):
        for card in assets["mechanisms"]:
            st.markdown(
                f'**{card["sample_id"]}** · `{card["map_id"]}` · `{card["vehicle_structure"]}` · '
                f'失败风险 {float(card["failure_risk"]):.3f} · {label_text(card["true_label"])}'
            )
            st.dataframe(pd.DataFrame(card["reasons"]), width="stretch", hide_index=True)

    st.code(
        f'model = {runtime["model_name"] if runtime["model_available"] else "未加载"}\n'
        f'feature_version = {contract["feature_version"]}\n'
        f'schema_sha256 = {contract["schema_sha256"]}\n'
        'risk = 0.6 × model_failure_risk + 0.4 × geometry_rule_risk',
        language="text",
    )
    st.caption("PathGuard只安排候选路线验证优先级；最终结论仍需闭环仿真、人工复核或实车验证。")
