"""PathGuard unified Streamlit demo for offline candidate-route risk triage."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:  # Works both with `streamlit run demo/app.py` and AppTest from repo root.
    from demo_logic import apply_filters, evaluate_candidates, select_validation_queue
    from feature_explain import label_text
    from inference import FeatureValidationError, load_contract
    from route_input import compute_boundary_rule, parse_route, render_evidence
    from feedback_store import load_feedback, save_feedback, classify_disagreement, make_condition_key, set_training_review
    from task_store import create_task, export_task, list_tasks, load_routes, save_attachment, save_assessment
    from batch_evaluation import evaluate_verified_task
except ModuleNotFoundError:  # pragma: no cover - exercised by Streamlit AppTest.
    from demo.demo_logic import apply_filters, evaluate_candidates, select_validation_queue
    from demo.feature_explain import label_text
    from demo.inference import FeatureValidationError, load_contract
    from demo.route_input import compute_boundary_rule, parse_route, render_evidence
    from demo.feedback_store import load_feedback, save_feedback, classify_disagreement, make_condition_key, set_training_review
    from demo.task_store import create_task, export_task, list_tasks, load_routes, save_attachment, save_assessment
    from demo.batch_evaluation import evaluate_verified_task


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "demo"
DATA_DIR = DEMO_DIR / "data"
MODEL_PATH = DEMO_DIR / "models" / "pathguard_gate3_model.joblib"
CONTRACT_PATH = DATA_DIR / "feature_contract_v2.json"
SAMPLE_PATH = DATA_DIR / "sample_input.csv"
FROZEN_CASES_PATH = DATA_DIR / "modeling_features_final.csv"
HERO_IMAGE_PATH = DEMO_DIR / "assets" / "pathguard-heavy-haul-hero.png"


def image_data_uri(path: Path) -> str:
    """Return a self-contained image URL so Streamlit can render local assets reliably."""
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


hero_image_uri = image_data_uri(HERO_IMAGE_PATH)

st.set_page_config(page_title="PathGuard 路线风险决策", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")
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
    .page-head {padding:1.4rem 1.55rem;margin:.35rem 0 1rem;border-radius:18px;background:linear-gradient(125deg,#09253b,#0a6672);color:#fff;box-shadow:0 16px 38px rgba(6,54,70,.16);}
    .page-head h1 {margin:0;color:#fff;font-size:2rem}.page-head p {margin:.45rem 0 0;color:#dff4f5;max-width:860px;line-height:1.65;}
    .landing-proof {display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem;margin:1rem 0 1.2rem;}
    .landing-proof>div {padding:1rem 1.1rem;border:1px solid #d9e5ec;border-radius:14px;background:rgba(255,255,255,.94);}
    .landing-proof b {display:block;color:#0e3048;margin-bottom:.3rem}.landing-proof span {font-size:.88rem;color:#5d7282;line-height:1.55;}
    .audience-grid {display:grid;grid-template-columns:repeat(2,1fr);gap:.8rem;margin:.7rem 0 1.2rem;}
    .audience-card {padding:1.15rem 1.2rem;border-radius:15px;background:#fff;border:1px solid #d9e5ec;box-shadow:0 10px 26px rgba(9,45,67,.06);}
    .audience-card h4 {margin:0 0 .4rem;color:#0e3048}.audience-card p {margin:0;color:#5d7282;line-height:1.6;font-size:.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    :root {
      --pg-navy: #071a2b; --pg-navy-2: #0e3048; --pg-teal: #00a7a7;
      --pg-amber: #ff9d2e; --pg-ink: #102f43; --pg-muted: #5d7282;
      --pg-line: #d9e5ec; --pg-surface: rgba(255,255,255,.96);
    }
    html, body, [class*="css"] {font-family: Inter, "Microsoft YaHei", "PingFang SC", sans-serif;}
    .stApp {
      color:var(--pg-ink);
      background:#f4f8fb;
    }
    [data-testid="stHeader"] {background:rgba(5,24,39,.72);backdrop-filter:blur(16px);border-bottom:1px solid rgba(255,255,255,.1);}
    [data-testid="stToolbar"] {color:white;}
    [data-testid="stSidebar"] {
      background:linear-gradient(180deg,#081f32 0%,#0b2b42 54%,#0b3347 100%);
      border-right:1px solid rgba(255,255,255,.08);box-shadow:14px 0 38px rgba(4,20,34,.16);
    }
    [data-testid="stSidebar"] * {color:#e9f5f7;}
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {color:#fff;letter-spacing:-.02em;}
    [data-testid="stSidebar"] [data-baseweb="select"]>div,
    [data-testid="stSidebar"] [data-baseweb="input"]>div,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
      background:rgba(255,255,255,.07);border-color:rgba(145,214,216,.3);border-radius:12px;
    }
    [data-testid="stSidebar"] details {background:rgba(255,255,255,.05);border:1px solid rgba(145,214,216,.16);border-radius:12px;}
    [data-testid="stMainBlockContainer"] {max-width:1480px;padding-top:4.4rem;padding-bottom:4rem;}
    .hero {
      position:relative;overflow:hidden;min-height:330px;display:flex;flex-direction:column;
      justify-content:center;align-items:flex-start;padding:3.6rem clamp(1.5rem,5vw,5rem);margin:0 0 1.1rem;
      border:1px solid rgba(255,255,255,.2);border-radius:26px;
      background:linear-gradient(90deg,rgba(3,18,31,.98) 0%,rgba(4,29,47,.88) 37%,rgba(5,30,45,.28) 70%,rgba(5,19,31,.1) 100%),
        url("__HERO_IMAGE__") center/cover no-repeat;box-shadow:0 28px 70px rgba(3,24,39,.34);
    }
    .hero:after {content:"";position:absolute;inset:auto 0 0;height:5px;background:linear-gradient(90deg,var(--pg-teal),var(--pg-amber),transparent 78%);}
    .hero-kicker {display:inline-flex;align-items:center;gap:.55rem;color:#a9eeee;font-weight:700;font-size:.78rem;letter-spacing:.16em;text-transform:uppercase;margin-bottom:.8rem;}
    .hero-kicker:before {content:"";width:28px;height:2px;background:var(--pg-amber);}
    .hero h1 {margin:0;color:#fff;font-size:clamp(2.7rem,6vw,5.4rem);line-height:.95;letter-spacing:-.055em;text-shadow:0 8px 30px rgba(0,0,0,.25);}
    .hero p {max-width:650px;margin:1.05rem 0 1.35rem;color:#dff4f5;font-size:clamp(1rem,1.55vw,1.32rem);line-height:1.7;opacity:.95;}
    .hero-badges {display:flex;flex-wrap:wrap;gap:.55rem;}
    .hero-badges span {padding:.42rem .72rem;border:1px solid rgba(171,235,235,.25);border-radius:999px;background:rgba(6,31,48,.58);backdrop-filter:blur(8px);color:#e8ffff;font-size:.78rem;}
    .notice {padding:.92rem 1.1rem;border:1px solid #f0d7b6;border-left:4px solid var(--pg-amber);background:rgba(255,249,239,.96);box-shadow:0 8px 28px rgba(98,66,24,.06);border-radius:12px;color:#6d4919;margin:.45rem 0 1.35rem;}
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
      gap:.5rem;padding:.45rem;background:rgba(255,255,255,.92);border:1px solid var(--pg-line);border-radius:16px;
      box-shadow:0 12px 32px rgba(9,45,67,.08);position:sticky;top:3.7rem;z-index:20;backdrop-filter:blur(16px);
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {height:44px;padding:0 1.2rem;border-radius:11px;color:#4f6676;font-weight:650;}
    [data-testid="stTabs"] [aria-selected="true"] {background:linear-gradient(135deg,var(--pg-navy-2),#087580);color:#fff!important;box-shadow:0 8px 18px rgba(6,73,89,.2);}
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {display:none;}
    [data-testid="stMetric"] {background:var(--pg-surface);border:1px solid var(--pg-line);border-radius:15px;padding:1rem 1.1rem;box-shadow:0 10px 28px rgba(9,45,67,.07);min-height:112px;}
    [data-testid="stMetricLabel"] {color:var(--pg-muted);font-weight:650;}
    [data-testid="stMetricValue"] {color:var(--pg-navy-2);font-weight:760;letter-spacing:-.035em;}
    [data-testid="stDataFrame"],[data-testid="stTable"],[data-testid="stPlotlyChart"],
    [data-testid="stForm"],[data-testid="stFileUploaderDropzone"] {
      background:var(--pg-surface);border:1px solid var(--pg-line);border-radius:16px;box-shadow:0 12px 32px rgba(9,45,67,.07);overflow:hidden;
    }
    [data-testid="stExpander"] {background:rgba(255,255,255,.88);border:1px solid var(--pg-line);border-radius:14px!important;box-shadow:0 8px 24px rgba(9,45,67,.05);overflow:hidden;}
    .stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button {
      border:0;border-radius:10px;background:linear-gradient(135deg,#0d6172,var(--pg-teal));color:#fff;font-weight:700;
      box-shadow:0 8px 18px rgba(0,128,139,.2);transition:transform .16s ease,box-shadow .16s ease;
    }
    .stButton>button:hover,.stDownloadButton>button:hover,[data-testid="stFormSubmitButton"]>button:hover {transform:translateY(-1px);box-shadow:0 12px 24px rgba(0,128,139,.28);color:#fff;}
    .vehicle-card,.value-card,.envelope-card,.risk-box {background:var(--pg-surface);border-color:var(--pg-line);box-shadow:0 12px 32px rgba(9,45,67,.07);}
    .vehicle-card {border-top:3px solid var(--pg-teal);}
    .value-card {position:relative;overflow:hidden;transition:transform .18s ease,box-shadow .18s ease;}
    .value-card:hover {transform:translateY(-2px);box-shadow:0 18px 38px rgba(9,45,67,.12);}
    .flow-step {border:1px solid #d8e9eb;background:linear-gradient(145deg,#f3faf9,#edf4f7);}
    .judge-card {background:linear-gradient(130deg,#09253b,#086b76);box-shadow:0 18px 38px rgba(6,54,70,.18);border:1px solid rgba(255,255,255,.1);}
    h1,h2,h3 {color:var(--pg-navy-2);letter-spacing:-.028em;} hr {border-color:#dce7ed!important;}
    @media (max-width:900px) {
      [data-testid="stMainBlockContainer"] {padding-top:3.7rem;padding-left:1rem;padding-right:1rem;}
      .hero {min-height:300px;padding:2.3rem 1.4rem;background-position:64% center;}
      .hero p {max-width:88%;}.value-grid,.risk-summary,.landing-proof,.audience-grid {grid-template-columns:1fr!important;}.flow-strip {flex-direction:column;}
      [data-testid="stTabs"] [data-baseweb="tab"] {padding:0 .65rem;font-size:.82rem;}
    }
    </style>
    """.replace("__HERO_IMAGE__", hero_image_uri),
    unsafe_allow_html=True,
)


def read_json(name: str) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def read_route_geometry() -> dict[str, Any]:
    full_asset = DATA_DIR / "route_geometry_all.json.gz"
    if full_asset.exists():
        with gzip.open(full_asset, "rt", encoding="utf-8") as source:
            return json.load(source)
    return read_json("route_geometry.json")


@st.cache_data
def load_static_assets() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    assets: dict[str, Any] = {
        "freeze": read_json("freeze_data_card.json"),
        "test": read_json("gate3_test_metrics.json"),
        "oof": read_json("gate3_oof_metrics.json"),
        "audit": read_json("repeated_group_hybrid_summary.json"),
        "mechanisms": read_json("failure_mechanism_cards.json"),
        "route_geometry": read_route_geometry(),
    }
    boundary_summary = REPO_ROOT / "analysis" / "boundary_geometry" / "boundary_coverage_summary.json"
    assets["boundary"] = json.loads(boundary_summary.read_text(encoding="utf-8")) if boundary_summary.exists() else {}
    integrity_summary = REPO_ROOT / "reports" / "final" / "boundary_integrity_reaudit_summary.json"
    assets["boundary_integrity"] = json.loads(integrity_summary.read_text(encoding="utf-8")) if integrity_summary.exists() else {}
    batch_summary = REPO_ROOT / "reports" / "final" / "candidate_batch_proxy_summary.json"
    assets["batch_proxy"] = json.loads(batch_summary.read_text(encoding="utf-8")) if batch_summary.exists() else {}
    assets["heldout_methods_v3"] = pd.read_csv(REPO_ROOT / "reports" / "final" / "tables" / "heldout_methods_v3.csv")
    contract = load_contract(CONTRACT_PATH)
    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
    assets["frozen_features"] = pd.read_csv(FROZEN_CASES_PATH)
    assets["frozen_cases"] = assets["frozen_features"][
        ["sample_id", "map_id", "route_id", "vehicle_structure", "true_label", "route_length_m"]
    ]
    return assets, contract, sample


def vehicle_illustration(title: str, count: int, caption: str) -> None:
    axles = "".join('<span class="axle"></span>' for _ in range(count))
    st.markdown(
        f'<div class="vehicle-card"><strong>{title}</strong><div class="axles">{axles}</div>'
        f'<div class="small-note">{caption}</div></div>',
        unsafe_allow_html=True,
    )


def _svg_paths(geometry: dict[str, Any], focus_station: float | None = None) -> tuple[str, str, str, float]:
    center = geometry["centerline"]
    station = geometry.get("station_m", list(range(len(center))))
    station_array = pd.Series(station, dtype=float).to_numpy()
    if focus_station is None:
        focus_station = float(station_array[len(station_array) // 2])
    selected = (station_array >= focus_station - 35.0) & (station_array <= focus_station + 35.0)
    if selected.sum() < 2:
        nearest = int(abs(station_array - focus_station).argmin())
        lo, hi = max(0, nearest - 1), min(len(center), nearest + 2)
        selected = [i >= lo and i < hi for i in range(len(center))]
    center_path = [point for point, keep in zip(center, selected) if keep]
    groups = [center_path]
    for key in ("left_boundary", "right_boundary"):
        values = geometry.get(key, [])
        groups.append(
            [point for point, keep in zip(values, selected) if keep]
            if len(values) == len(center) else []
        )
    points = [point for group in groups for point in group]
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    span_x = max(max(xs) - min(xs), 1e-9)
    span_y = max(max(ys) - min(ys), 1e-9)
    scale = min(540.0 / span_x, 220.0 / span_y)
    offset_x = 30.0 + (540.0 - span_x * scale) / 2.0
    offset_y = 20.0 + (220.0 - span_y * scale) / 2.0

    def path(group: list[list[float]]) -> str:
        if not group:
            return ""
        transformed = [
            (offset_x + (float(x) - min(xs)) * scale, offset_y + (max(ys) - float(y)) * scale)
            for x, y in group
        ]
        return " ".join(
            ("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}"
            for index, (x, y) in enumerate(transformed)
        )

    return path(groups[0]), path(groups[1]), path(groups[2]), scale


def effective_vehicle_config(geometry: dict[str, Any] | None) -> dict[str, Any]:
    if not geometry:
        return {}
    source = dict(geometry.get("vehicle_parameters", {}))
    token = str(geometry.get("source_sha256", ""))[:12]
    if "length_" + token in st.session_state:
        source["length_m"] = float(st.session_state["length_" + token])
    if "width_" + token in st.session_state:
        source["width_m"] = float(st.session_state["width_" + token])
    if "reference_" + token in st.session_state and "length_m" in source:
        source["reference_from_rear_m"] = float(source["length_m"]) * float(st.session_state["reference_" + token]) / 100.0
    source["dimensions_confirmed"] = bool(st.session_state.get("use_dimensions_" + token, False))
    if source["dimensions_confirmed"]:
        source["dimension_source"] = "operator_confirmed_from_ui"
    return source


def render_full_route_context(st, geometry: dict[str, Any], boundary_rule: dict[str, Any] | None = None) -> None:
    """Show where the animated local segment sits on the complete supplied route."""
    import numpy as np
    import plotly.graph_objects as go
    center = np.asarray(geometry.get("centerline", []), dtype=float)
    stations = np.asarray(geometry.get("station_m", []), dtype=float)
    if len(center) < 2 or len(stations) != len(center):
        return
    events = geometry.get("events", [])
    rule_index = None
    if geometry.get("body_clearance_m") and len(geometry["body_clearance_m"]) == len(stations):
        rule_index = int(np.argmin(np.asarray(geometry["body_clearance_m"], dtype=float)))
    focus = (float(boundary_rule["minimum_station_m"]) if boundary_rule and boundary_rule.get("available")
             else float(stations[rule_index]) if rule_index is not None else float(stations[len(stations)//2]))
    local = (stations >= focus - 35) & (stations <= focus + 35)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=center[:,0], y=center[:,1], mode="lines", name="输入路线全程",
                             line=dict(color="#9aacb7", width=3)))
    if local.sum() >= 2:
        fig.add_trace(go.Scatter(x=center[local,0], y=center[local,1], mode="lines", name="下方动画区段",
                                 line=dict(color="#d35435", width=6)))
    if events:
        fig.add_trace(go.Scatter(x=[item["x_m"] for item in events], y=[item["y_m"] for item in events],
                                 mode="markers", text=[item["label"] for item in events],
                                 name="可定位工程指标", marker=dict(size=9, color="#e7a548")))
    fig.update_layout(height=310, margin=dict(l=10,r=10,t=10,b=10),
                      xaxis_title="x / m", yaxis_title="y / m", legend=dict(orientation="h"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.markdown("#### 输入路线全貌与局部位置")
    st.caption(f"输入文件覆盖 {stations[-1]-stations[0]:.1f} m、原始 {geometry.get('point_count_original', len(center))} 个点；橙色区段为里程 {max(float(stations[0]),focus-35):.1f}—{min(float(stations[-1]),focus+35):.1f} m，对应下方局部动画。该文件是否代表完整工程任务路线，由数据提供方确认。")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "scrollZoom": True})
    st.caption("图表操作：拖动框选放大，滚轮缩放，双击图表恢复全图。")


def dynamic_envelope(row: pd.Series, route_geometries: dict[str, Any], boundary_rule: dict[str, Any] | None = None) -> None:
    """Animate a meter-scaled rigid-body proxy along the selected real route segment."""
    structure = str(row.get("vehicle_structure", "unknown"))
    curvature = abs(float(row.get("curvature_abs_p95_1pm", 0.0)))
    steer_change = abs(float(row.get("steer_change_abs_p95_radpm", 0.0)))
    clearance = float(row.get("static_boundary_clearance_min_m", 0.0))
    structure_label = "六轴" if structure == "six_axis" else "五轴" if structure == "five_axis" else "结构未定"
    sample_token = "".join(ch for ch in str(row.get("sample_id", "route")) if ch.isalnum())[-12:]
    route_id = f"motion-route-{sample_token or 'selected'}"
    geometry = route_geometries.get(str(row.get("sample_id", "")))
    if not geometry or len(geometry.get("centerline", [])) < 2:
        st.info("当前输入没有原始路线坐标，不能生成动态车辆或空间风险图。请上传含 x_m、y_m 的原始路线 NPZ；仅凭汇总特征不绘制虚构道路、障碍物或车身运动。")
        return
    parameters = geometry.get("vehicle_parameters", {})
    defaults = {"five_axis": (12.0, 3.2, 5.0), "six_axis": (15.0, 3.4, 6.0), "unknown": (12.0, 3.2, 5.0)}[structure]
    token = str(geometry.get("source_sha256", ""))[:12]
    length = float(st.session_state.get("length_" + token, parameters.get("length_m", defaults[0])))
    width = float(st.session_state.get("width_" + token, parameters.get("width_m", defaults[1])))
    reference_ratio = float(st.session_state.get("reference_" + token, round(float(parameters.get("reference_from_rear_m", defaults[2])) / max(length, 1e-6) * 100)))
    reference = length * reference_ratio / 100.0
    dimensions_are_assumed = (
        not all(key in parameters for key in ("length_m", "width_m", "reference_from_rear_m"))
        or parameters.get("dimension_source") == "illustrative_demo_values"
    )
    rule = boundary_rule or {}
    if rule.get("available"):
        focus_station = float(rule["minimum_station_m"])
    else:
        body = geometry.get("body_clearance_m", [])
        station = geometry.get("station_m", [])
        focus_station = float(station[int(abs(pd.Series(body, dtype=float).argmin()))]) if body and len(body) == len(station) else float(station[len(station) // 2])
    route_d, left_boundary_d, right_boundary_d, meter_scale = _svg_paths(geometry, focus_station)
    boundary_markup = ""
    if geometry.get("boundary_semantics_verified") and left_boundary_d and right_boundary_d:
        boundary_markup = f'<path d="{left_boundary_d}" class="road-edge"/><path d="{right_boundary_d}" class="road-edge"/>'
    body_length = length * meter_scale
    body_width = width * meter_scale
    body_x = -reference * meter_scale
    geometry_badge = f'真实路线局部视图 · {geometry["point_count_original"]}原始点 / {geometry["point_count_display"]}显示点'
    geometry_note = "路线坐标来自当前NPZ；视图聚焦在当前最小净空或边界估算点前后各35米。矩形按输入车长、车宽和参考点比例缩放，颜色不表示风险概率。"
    if dimensions_are_assumed:
        geometry_note += " 未上传尺寸配置，当前长宽和参考点为可编辑的示例假设值，不代表某一车型的公开参数。"
    level = str(row.get("risk_level", "待判定"))
    if structure == "unknown" or level == "证据不足":
        result = "证据不足，拒绝确定判断"
        consequence = "车辆结构未定，当前风险分不能作为车型结论"
        action = "补充车型/轴位证据后重新评估"
    elif level in {"高风险", "高"}:
        result = "预测为优先复核路线"
        consequence = "排序提示需结合下方工程量值和边界证据复核；风险分数不是概率"
        action = str(row.get("next_action", "优先补测 / 人工复核"))
    elif level in {"中风险", "中"}:
        result = "预测为需要关注路线"
        consequence = "请查看对应工程指标和边界证据，确认风险成因"
        action = str(row.get("next_action", "安排复核"))
    else:
        result = "预测为当前批次低优先级路线"
        consequence = "当前排序靠后不代表路线安全，也不构成免检依据"
        action = str(row.get("next_action", "暂缓处理 / 按计划验证"))
    reasons = str(row.get("risk_reasons", "曲率、转向变化、净空等指标"))
    svg = f'''<div class="envelope-card">
      <div class="envelope-title"><strong>这条路线会发生什么？</strong><span>{structure_label} · {geometry_badge}</span></div>
      <div class="risk-summary">
        <div class="risk-box primary"><b>预测结论</b><span>{result}</span></div>
        <div class="risk-box"><b>可能的工程情况</b><span>{consequence}</span></div>
        <div class="risk-box"><b>主要原因</b><span>{reasons}</span></div>
        <div class="risk-box"><b>建议动作</b><span>{action}</span></div>
      </div>
      <svg viewBox="0 0 600 290" role="img" aria-label="米制局部路线与车身外廓动态示意">
        <defs><marker id="pg-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#2b6f8d"/></marker></defs>
        {boundary_markup}
        <path id="{route_id}" d="{route_d}" class="route-line" marker-end="url(#pg-arrow)"/>
        <g class="moving-vehicle">
          <rect x="{body_x:.2f}" y="{-body_width/2:.2f}" width="{body_length:.2f}" height="{body_width:.2f}" rx="2" class="vehicle-body"/>
          <animateMotion dur="6.5s" repeatCount="indefinite" rotate="auto"><mpath href="#{route_id}"/></animateMotion>
        </g>
        <text x="18" y="257" class="svg-label">车长 {length:.2f} m · 车宽 {width:.2f} m · 参考点距车尾 {reference:.2f} m</text>
        <text x="18" y="275" class="svg-label">局部净空 {clearance:.2f} m · 曲率P95 {curvature:.3f} 1/m · 转向变化P95 {steer_change:.3f} rad/m</text>
      </svg>
      <div class="legend"><span><i></i>蓝线：当前NPZ路线参考点轨迹</span><span><i class="dashed"></i>灰线：通过语义与左右顺序检查的边界点列</span><span>浅色矩形：车长/车宽按米制比例绘制的刚性外廓</span></div>
      <div class="envelope-note">{geometry_note} 矩形不含轴位/铰接结构、车身形变、轮胎或动力学参数；坐标不支持可信边界时仅显示路线与车身示意，不推断障碍物碰撞。</div>
    </div>'''
    st.markdown(svg, unsafe_allow_html=True)


def percent(value: float) -> str:
    return f"{float(value):.1%}"


assets, contract, sample_frame = load_static_assets()
freeze = assets["freeze"]

site_page = st.radio("主导航", ["首页", "产品介绍", "风险工作台"], horizontal=True,
                     label_visibility="collapsed", key="site_page")

if site_page == "首页":
    st.markdown(
        '''<section class="hero" style="min-height:560px;">
          <div class="hero-kicker">PathGuard · 工程车辆路线风险决策</div>
          <h1>把有限的验证资源<br/>留给最值得看的路线</h1>
          <p>在闭环仿真和实车测试之前，先从大量候选路线中找出高风险项，定位风险发生在哪里，并给出下一步验证建议。</p>
          <div class="hero-badges"><span>五轴 / 六轴工程车辆</span><span>候选路线优先级</span><span>风险位置解释</span><span>验证结果回流</span></div>
        </section>''', unsafe_allow_html=True)
    st.markdown(
        '<div class="landing-proof">'
        '<div><b>先看哪条</b><span>同一任务的候选路线一起输入，先安排值得核查的路线；节省工时的效果待真实任务验证。</span></div>'
        '<div><b>为什么要看</b><span>把曲率、转向、坡度、净空和尺寸边界结果落到具体位置。</span></div>'
        '<div><b>下一步怎么做</b><span>区分优先复核、补充边界、闭环仿真和暂缓处理。</span></div>'
        '</div>', unsafe_allow_html=True)
    action_cols = st.columns([1,1,2])
    action_cols[0].button("了解产品", use_container_width=True, on_click=lambda: st.session_state.update(site_page="产品介绍"))
    action_cols[1].button("进入风险工作台", use_container_width=True, on_click=lambda: st.session_state.update(site_page="风险工作台"))
    st.caption("当前背景图为临时视觉素材，后续可直接替换为你提供的项目主视觉，不影响页面结构。")
    st.stop()

if site_page == "产品介绍":
    st.markdown('<section class="page-head"><h1>候选路线很多，工程师只需要先看最关键的几条</h1><p>PathGuard服务于规划、测试和项目交付团队：把执行前路线数据转成一张可追溯的验证清单，让风险位置、判断依据和下一步动作落在同一条工作流里。</p></section>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">客户面临的问题</div>', unsafe_allow_html=True)
    st.markdown('<div class="audience-grid">'
        '<div class="audience-card"><h4>路线算出来了，仍然不知道先测哪条</h4><p>算法内部可能产生几十到几百条候选，人工逐条查看会消耗大量日志分析和仿真时间。</p></div>'
        '<div class="audience-card"><h4>一个风险分数无法支持工程决策</h4><p>工程师还需要知道风险发生在哪里、涉及哪个指标，以及换路线、限速或补充数据能否解决。</p></div>'
        '</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">PathGuard如何工作</div><div class="flow-strip">'
        '<div class="flow-step"><b>① 建立工程任务</b>说明场景版本、车辆与输入候选</div>'
        '<div class="flow-step"><b>② 形成优先队列</b>模型排序，固定规则交叉检查</div>'
        '<div class="flow-step"><b>③ 定位工程原因</b>查看风险指标、边界余量和证据等级</div>'
        '<div class="flow-step"><b>④ 留存验证结果</b>记录仿真或现场结论与适用条件</div>'
        '</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">客户最终得到什么</div>', unsafe_allow_html=True)
    st.markdown('<div class="value-grid">'
        '<div class="value-card"><h4>一张先后有序的路线清单</h4><p>按现有模型排序和证据状态组织复核顺序；是否减少漏检，需要在真实候选批次中继续验证。</p></div>'
        '<div class="value-card"><h4>一张能落到位置的解释卡</h4><p>展示风险原因、最小边界余量、影响部位和尺寸变化后的结果。</p></div>'
        '<div class="value-card"><h4>一套可审核的验证记录</h4><p>保留原件、条件和多次结果；审核后形成训练候选数据。</p></div>'
        '</div>', unsafe_allow_html=True)
    st.info("PathGuard用于验证前的风险排序与复核安排。最终安全结论仍由闭环仿真、人工复核或实车测试给出。")
    st.button("打开风险工作台", on_click=lambda: st.session_state.update(site_page="风险工作台"))
    st.stop()

st.markdown('<section class="page-head"><h1>风险工作台</h1><p>先在候选队列中确定验证优先级，再进入路线分析查看通俗结论、专业指标和尺寸联动边界结果；最后在验证依据中核对数据与模型证据。</p></section>', unsafe_allow_html=True)
st.markdown('<div class="notice">工作台输出用于验证资源安排。高风险代表优先复核，低风险不代表免检；证据不足时系统会明确拒绝确定判断。</div>', unsafe_allow_html=True)

with st.expander("① 任务与输入 · 选择历史案例或建立新任务", expanded=True):
    st.header("工程任务与输入")
    task_mode = st.radio("工作入口", ["历史案例库", "新建评估任务", "已保存任务"],
                         horizontal=True, key="task_mode",
                         on_change=lambda: st.session_state.pop("active_task_id", None))
    tasks = list_tasks()
    active_task = None
    route_uploads = None
    config_upload = None
    upload = None
    if task_mode == "新建评估任务":
        st.caption("同一任务中的候选路线应使用相同的场景版本和车辆配置；系统不会从地图编号猜测它们可直接比较。")
        task_title = st.text_input("任务名称", placeholder="例如：东侧运输线方案复核")
        task_scene = st.text_input("场景 / 地图名称", placeholder="例如：矿区东侧道路")
        environment_version = st.text_input("环境版本 / 日期", placeholder="例如：2026-09-23测绘版")
        intake_status = st.radio("输入状态", ["待评估", "已有验证记录"], horizontal=True)
        route_uploads = st.file_uploader('候选路线 NPZ（可多选）', type=['npz'], accept_multiple_files=True)
        config_upload = st.file_uploader('车辆配置 JSON', type=['json'])
        with st.expander('下载路线与配置示例'):
            for name in ['real_route.npz', 'vehicle_config.json']:
                st.download_button('下载 '+name, (DEMO_DIR/'examples'/name).read_bytes(), file_name=name)
        st.caption("已有验证结果在路线详情中逐次登记；是否用于训练需单独审核。")
        if st.button("保存任务并评估", type="primary"):
            try:
                if not route_uploads:
                    raise ValueError("请上传至少一条候选路线。")
                config = json.load(config_upload) if config_upload else {}
                # Validate every file against the frozen feature extractor before persisting.
                for uploaded_route in route_uploads:
                    parse_route(uploaded_route.getvalue(), uploaded_route.name, config)
                active_task = create_task(
                    title=task_title, scene=task_scene, environment_version=environment_version,
                    intake_status=intake_status, vehicle_config=config,
                    routes=[(item.name, item.getvalue()) for item in route_uploads],
                    feature_version=contract["feature_version"], schema_sha256=contract["schema_sha256"],
                    model_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() if MODEL_PATH.exists() else "",
                )
                st.session_state["active_task_id"] = active_task["task_id"]
            except (ValueError, TypeError, OSError, json.JSONDecodeError) as error:
                st.error(f"任务未保存：{error}")
        if active_task is None:
            active_task = next((item for item in list_tasks() if item["task_id"] == st.session_state.get("active_task_id")), None)
    elif task_mode == "已保存任务":
        if tasks:
            selected_task_id = st.selectbox("选择任务", [item["task_id"] for item in tasks],
                                            format_func=lambda item_id: next(item["title"] for item in tasks if item["task_id"] == item_id))
            active_task = next(item for item in tasks if item["task_id"] == selected_task_id)
        else:
            st.info("本机尚无保存的任务。")
    else:
        history_scope = st.radio("历史记录范围", ["冻结记录全集（340）", "快速讲解示例（15）"], horizontal=True)
        st.caption("全部340条冻结记录已从哈希匹配的原始NPZ恢复路线坐标；可逐条查看动态路径。可靠的左右边界和车辆尺寸仍须另行核验。不同编号不代表同一任务的候选方案。")
        upload = st.file_uploader("上传执行前特征 CSV", type=["csv"],
                                  help="必须包含冻结契约规定的全部55项数值特征。")
    uploaded_geometry = {}
    if active_task is not None:
        try:
            config = dict(active_task["vehicle_config"])
            config["map_id"] = active_task["scene"]
            frames = []
            for route_name, payload in load_routes(active_task):
                frame, key, geometry = parse_route(payload, route_name, config)
                frames.append(frame)
                if geometry:
                    uploaded_geometry[key] = geometry
            input_frame = pd.concat(frames, ignore_index=True).drop_duplicates('sample_id')
            st.caption(f'任务「{active_task["title"]}」：{len(input_frame)} 条输入候选；场景版本 {active_task["environment_version"]}。')
        except Exception as error:
            st.error(f'已保存任务无法复现：{error}')
            st.stop()
    elif task_mode != "历史案例库":
        st.info("请创建或选择任务后开展评估。")
        st.stop()
    elif upload is None:
        input_frame = (sample_frame if history_scope == "快速讲解示例（15）"
                       else assets["frozen_features"]).copy()
        st.caption(f"当前使用：{history_scope}；记录之间不预设为同一候选批次。")
    else:
        try:
            input_frame = pd.read_csv(upload, encoding="utf-8-sig")
            st.caption(f"已读取 {len(input_frame)} 条上传记录")
        except Exception as error:
            st.error(f"CSV读取失败：{error}")
            st.stop()
    structure_label = st.radio("车辆结构", ["全部", "五轴", "六轴", "结构未定"], horizontal=True)
    structure_map = {"全部": "all", "五轴": "five_axis", "六轴": "six_axis", "结构未定": "unknown"}
    top_selection = (st.selectbox("常规优先验证名额", ["Top 10", "Top 20", "Top 34"],
                                  format_func=lambda value: f'优先 {value.split()[-1]} 条')
                     if task_mode != "历史案例库" else "历史案例")
    search = st.text_input("检索样本或路线编号")

try:
    evaluated, rule_components, runtime = evaluate_candidates(input_frame, contract, MODEL_PATH)
except (FeatureValidationError, ValueError) as error:
    st.error(f"输入未通过冻结特征契约校验：{error}")
    st.stop()

# Attach geometry evidence to the same evaluated route used by the queue and detail view.
route_geometries: dict[str, Any] = {}
for _, candidate_row in evaluated.iterrows():
    sid = str(candidate_row.get("sample_id", ""))
    geometry_key = str(candidate_row.get("geometry_key", ""))
    geometry = uploaded_geometry.get(geometry_key) if geometry_key else None
    if geometry is None and upload is None and not route_uploads:
        built_in = assets["route_geometry"]["routes"].get(sid)
        if built_in and built_in.get("source_sha256") == str(candidate_row.get("route_file_sha256", "")):
            geometry = built_in
    if geometry:
        route_geometries[sid] = geometry

geometry_results = {}
geometry_states = []
geometry_margins = []
for _, candidate_row in evaluated.iterrows():
    sid = str(candidate_row.get("sample_id", ""))
    geometry = route_geometries.get(sid)
    rule = {"available": False, "reason": "没有可用的原始坐标或语义校验通过的边界。"}
    if geometry and geometry.get("boundary_semantics_verified"):
        params = geometry.get("vehicle_parameters", {})
        token = str(geometry.get("source_sha256", ""))[:12]
        has_dimensions = (
            all(key in params for key in ("length_m", "width_m", "reference_from_rear_m"))
            and params.get("dimension_source") != "illustrative_demo_values"
        )
        dimensions_enabled = st.session_state.get("use_dimensions_" + token, has_dimensions)
        try:
            if not dimensions_enabled:
                raise ValueError("请先确认车辆长宽和参考点配置，再启用尺寸联动规则。")
            length = float(st.session_state.get("length_" + token, params.get("length_m")))
            width = float(st.session_state.get("width_" + token, params.get("width_m")))
            default_reference = float(params.get("reference_from_rear_m", 0.0))
            ratio = float(st.session_state.get("reference_" + token, round(default_reference / max(length, 1e-6) * 100)))
            reference = length * ratio / 100.0
            margin = float(st.session_state.get("margin_" + token, 0.5))
            rule = compute_boundary_rule(
                geometry,
                length,
                width,
                reference,
                margin,
            )
        except (KeyError, TypeError, ValueError):
            rule = {"available": False, "reason": "车辆长宽或参考点配置缺失，无法计算边界余量。"}
    geometry_results[sid] = rule
    geometry_states.append(rule.get("state", "未评估"))
    geometry_margins.append(rule.get("minimum_margin_m"))
evaluated["geometry_state"] = geometry_states
evaluated["geometry_min_margin_m"] = geometry_margins
geometry_forced = evaluated["geometry_state"].eq("局部外廓估算越界")
unknown_forced = evaluated["vehicle_structure"].eq("unknown")
conflict_forced = evaluated["decision_status"].isin(["模型/规则冲突，人工复核", "证据不足，拒绝确定结论"])
priority_forced = evaluated["validation_priority"].astype(str).str.startswith("P0")
evaluated["mandatory_review"] = geometry_forced | unknown_forced | conflict_forced | priority_forced
evaluated["mandatory_review_reason"] = ""
evaluated.loc[unknown_forced, "mandatory_review_reason"] = "车型结构未确认"
evaluated.loc[conflict_forced & ~unknown_forced, "mandatory_review_reason"] = "模型与冻结规则不一致"
evaluated.loc[priority_forced & ~unknown_forced & ~conflict_forced, "mandatory_review_reason"] = "进入P0风险等级"
evaluated.loc[geometry_forced, "mandatory_review_reason"] = "局部外廓估算越过语义校验边界"
if active_task:
    save_assessment(active_task, evaluated, runtime.get("model_name", "模型未加载"))

if active_task:
    st.info(f'当前任务：{active_task["title"]} · 场景 {active_task["scene"]} · 环境 {active_task["environment_version"]} · {len(evaluated)} 条候选 · 输入状态 {active_task["intake_status"]}。本任务内的路线按同一申报条件比较，验证结果逐次记录。')
    if len(evaluated) < 2:
        st.warning("当前只有一条输入路线：可以解释其风险，但无法比较候选方案或证明排序节省了复核工作量。请上传同场景、同配置的多条路线。")
else:
    st.info(f'当前是历史记录浏览：展示 {len(evaluated)} 条，冻结监督记录共 {len(assets["frozen_cases"])} 条；全部可逐条查看动态路线。它们不代表同一次规划生成的候选路线，也不能用本页的复核占比估算日常工作量。')

if runtime["model_available"]:
    st.success("当前使用学习模型为路线排序，工程规则单独提示需要核查的情况。模型版本与输入格式已经核对。")
else:
    st.warning("当前为规则预览：冻结模型与本机运行时版本不一致，系统已安全回退，不影响界面与流程演示。")
    with st.expander("查看技术状态"):
        st.caption(runtime["model_error"])

active_structure = structure_map[structure_label]
structure_view = (
    evaluated
    if active_structure == "all"
    else evaluated[evaluated["vehicle_structure"] == active_structure]
)

overview_tab, queue_tab, detail_tab, evidence_tab = st.tabs(
    ["任务概览" if active_task else "历史概览", "验证队列" if active_task else "历史记录", "路线分析", "验证证据"]
)

with overview_tab:
    if active_task:
        st.markdown('<div class="section-kicker">当前工程任务</div><h2 style="margin:.1rem 0 .25rem;color:#173f59;">先确定验证顺序，再看每条路线为什么值得关注</h2>', unsafe_allow_html=True)
        status_cols = st.columns(3)
        status_cols[0].metric("本任务候选路线", len(evaluated))
        status_cols[1].metric("当前筛选后", len(structure_view))
        status_cols[2].metric("有路线坐标可解释", len(route_geometries))
        st.info(f'任务「{active_task["title"]}」使用场景「{active_task["scene"]}」、环境版本「{active_task["environment_version"]}」。请在“验证队列”查看每条路线的建议去向，选择路线后在“路线分析”查看位置、指标和验证记录。')
    else:
        st.markdown('<div class="section-kicker">历史记录浏览</div><h2 style="margin:.1rem 0 .25rem;color:#173f59;">选择一条历史路线，看清风险提示来自哪里</h2>', unsafe_allow_html=True)
        status_cols = st.columns(3)
        status_cols[0].metric("当前浏览记录", len(evaluated))
        status_cols[1].metric("冻结记录总数", len(assets["frozen_cases"]))
        status_cols[2].metric("有路线坐标可解释", len(route_geometries))
        st.info("历史记录供查看路线分析与已知结果，不是同一次规划任务的候选批次。要比较同一任务的多条新路线，请在顶部切换到“新建评估任务”。")
    st.markdown('<div class="flow-strip">'
        '<div class="flow-step"><b>① 看路线清单</b>了解当前输入与优先级</div>'
        '<div class="flow-step"><b>② 看具体位置</b>查看路线、车辆外廓和关注点</div>'
        '<div class="flow-step"><b>③ 定下一步动作</b>记录人工、仿真或现场验证结论</div>'
        '</div>', unsafe_allow_html=True)
    st.caption("模型效果、数据覆盖和实验细节集中放在“验证证据”页，供需要核查技术依据时查看。")

map_options = ["全部"] + sorted(evaluated["map_id"].astype(str).unique().tolist())
with queue_tab:
    if active_task:
        st.markdown('<div class="section-kicker">第一步：确定验证顺序</div><h2 style="margin:.1rem 0 .25rem;color:#173f59;">本任务：先验证哪几条路线？</h2>', unsafe_allow_html=True)
        st.caption("同一任务的全部输入路线先接受评估，再分为本轮优先验证与其余按计划验证。高风险不等于已经失败，低风险也不等于免检。")
    else:
        st.markdown('<div class="section-kicker">历史材料 · 不是同一次任务</div><h2 style="margin:.1rem 0 .25rem;color:#173f59;">历史记录：查看全集，也可切换快速讲解示例</h2>', unsafe_allow_html=True)
        st.caption("15条快速讲解示例源自早期前端交付包的12条路线，后来加入3条通过边界完整性筛查的五轴路线。没有预先登记的代表性抽样方案；它们不能代表全量数据的风险或人工复核比例。")
    filter_cols = st.columns(2)
    risk_options = ["全部", "高风险", "中风险", "低风险", "证据不足"]
    if not runtime["model_available"]:
        risk_options.append("模型待补")
    with filter_cols[0]:
        risk_filter = st.selectbox("风险等级", risk_options)
    with filter_cols[1]:
        map_filter = st.selectbox("地图", map_options)
    filtered = apply_filters(evaluated, active_structure, risk_filter, map_filter, search)
    queue = (select_validation_queue(filtered, top_selection, runtime["model_available"])
             if active_task else filtered.sort_values("model_risk" if runtime["model_available"] else "rule_risk", ascending=False)).reset_index(drop=True)
    if not active_task:
        queue["queue_reason"] = "历史案例展示"
    display_columns = [
        "route_id", "route_length_m", "sample_id", "map_id", "vehicle_structure", "model_risk", "rule_risk",
        "risk_level", "validation_priority", "mandatory_review", "queue_reason",
        "geometry_state", "decision_status", "risk_reasons", "next_action",
    ]
    display = queue[display_columns].rename(columns={
        "route_id": "候选路线", "route_length_m": "输入长度/m", "sample_id": "追溯编号", "map_id": "场景", "vehicle_structure": "车辆结构",
        "model_risk": "主排序风险", "rule_risk": "规则风险",
        "risk_level": "风险等级", "validation_priority": "验证优先级", "decision_status": "一致性状态",
        "mandatory_review": "预算外必须复核", "queue_reason": "入队原因", "geometry_state": "车体边界估算",
        "risk_reasons": "主要风险因素", "next_action": "下一步动作",
    })
    display["车辆结构"] = display["车辆结构"].replace(
        {"five_axis": "五轴", "six_axis": "六轴", "unknown": "结构未定"}
    )
    score_title = "学习模型风险" if runtime["model_available"] else "规则预览"
    if active_task:
        total = len(filtered)
        mandatory_count = int(filtered["mandatory_review"].sum())
        planned_count = total - len(queue)
        status_cols = st.columns(4)
        status_cols[0].metric("当前输入路线", total)
        status_cols[1].metric("规则要求重点核查", mandatory_count)
        status_cols[2].metric("本轮队列", len(queue))
        status_cols[3].metric("其余按计划验证", planned_count)
        st.caption(
            f"先纳入必须核查项，再按{score_title}补足 {top_selection} 常规名额。"
            "这些是工作流状态，不是通过/失败结论；若全部入队，就表示当前规则未能节约本轮复核名额。"
        )
        with st.expander(f"查看筛选范围内全部 {total} 条输入路线及其去向"):
            all_routes = filtered[["route_id", "map_id", "vehicle_structure", "risk_level", "next_action"]].copy()
            all_routes["队列去向"] = ["本轮优先验证" if item in set(queue["sample_id"].astype(str)) else "其余按计划验证"
                                  for item in filtered["sample_id"].astype(str)]
            st.dataframe(all_routes.rename(columns={"route_id":"路线", "map_id":"场景", "vehicle_structure":"车辆结构", "risk_level":"风险等级", "next_action":"建议动作"}),
                         width="stretch", hide_index=True)
    else:
        history_cols = st.columns(4)
        history_cols[0].metric("当前浏览记录", len(evaluated))
        history_cols[1].metric("结构待核", int(evaluated["vehicle_structure"].eq("unknown").sum()))
        history_cols[2].metric("模型与规则分歧", int(evaluated["decision_status"].str.contains("冲突").sum()))
        history_cols[3].metric("冻结监督记录", len(assets["frozen_cases"]))
        st.caption(f"当前显示 {len(display)} 条历史记录；排序只方便浏览，不构成同任务 Top-K 实验。快速讲解15条没有经过代表性抽样，不能据此声称典型任务需要同样比例的人工复核。")
        if history_scope == "快速讲解示例（15）":
            with st.expander(f"预览其余冻结记录目录（共 {len(assets['frozen_cases'])} 条）"):
                st.caption("切换顶部“历史记录范围”可直接查看和分析全集。部署模型在这批记录上重训，全集评分不能当作独立测试成绩。全部记录可查看路线坐标；可靠边界和车型尺寸的覆盖仍须分别核验。")
                library = assets["frozen_cases"].rename(columns={"sample_id":"追溯编号", "map_id":"地图", "route_id":"路线来源", "vehicle_structure":"车辆结构", "true_label":"历史标签", "route_length_m":"输入长度/m"})
                st.dataframe(library, width="stretch", hide_index=True)
    st.caption("规则应力按冻结开发集参考分布计算，不会随本次上传批次变化，也不是失败概率。边界估算只在坐标语义校验通过且车辆尺寸可用时启用。当前模型/规则分差0.25的强制复核策略未经真实任务校准；复核比例不能当作产品收益。")
    queue_event = st.dataframe(
        display, width="stretch", hide_index=True, on_select="rerun",
        selection_mode="single-row", key="risk_queue_table",
    )
    if queue_event.selection.rows:
        selected_row = queue_event.selection.rows[0]
        st.session_state["selected_sample_id"] = str(queue.iloc[selected_row]["sample_id"])
        st.caption("已选中该路线；可切换到“路线分析”查看解释卡。")

with detail_tab:
    st.markdown('<div class="section-kicker">第二步：理解风险并安排动作</div><h2 style="margin:.1rem 0 .25rem;color:#173f59;">路线分析：哪里需要关注，下一步做什么？</h2>', unsafe_allow_html=True)
    if structure_view.empty:
        st.info("当前车辆结构筛选下没有候选路线；请切换结构或上传对应数据。")
    else:
        candidate_ids = structure_view["sample_id"].astype(str).tolist()
        preferred = st.session_state.get("selected_sample_id", candidate_ids[0])
        structure_names = {"five_axis": "五轴", "six_axis": "六轴", "unknown": "结构未定"}
        candidate_labels = {str(item["sample_id"]): f'{item["route_id"]} · {float(item["route_length_m"]):.0f} m · {structure_names.get(item["vehicle_structure"], "结构未定")}'
                            for _, item in structure_view.iterrows()}
        selected_id = st.selectbox(
            "选择候选路线", candidate_ids,
            index=candidate_ids.index(preferred) if preferred in candidate_ids else 0,
            format_func=lambda item_id: candidate_labels[item_id],
        )
        selected_index = evaluated.index[evaluated["sample_id"].astype(str) == selected_id][0]
        row = evaluated.loc[selected_index]
        components = rule_components.loc[selected_index].sort_values(ascending=False)
        geometry = route_geometries.get(selected_id)
        geometry_rule = geometry_results.get(selected_id, {})
        st.caption(f'当前输入：{row["route_id"]}，文件覆盖 {float(row["route_length_m"]):.1f} m；追溯编号 {selected_id}。工程关注点是同一输入路线上的位置，不是另外切出的候选路线。')
        if geometry and 'events' in geometry:
            render_full_route_context(st, geometry, geometry_rule)
            st.markdown('<div class="section-kicker">先看通俗结论</div>', unsafe_allow_html=True)
            dynamic_envelope(row, {selected_id: geometry}, geometry_rule)
            st.markdown('<div class="section-kicker">再看米制几何与专业证据</div>', unsafe_allow_html=True)
            render_evidence(st, geometry)
        else:
            dynamic_envelope(row, {})
            render_evidence(st, None)
        left, right = st.columns([1, 1])
        with left:
            st.subheader(selected_id)
            st.write(f'地图 / 路线：`{row["map_id"]}` / `{row["route_id"]}`')
            st.write(f'车辆结构：**{structure_names.get(row["vehicle_structure"], "结构未定")}**；标签：{label_text(row["true_label"])}')
            st.write(f'证据状态：**{row["evidence_status"]}**')
            st.write(f'验证优先级：**{row["validation_priority"]}**')
            st.write(f'建议动作：**{row["next_action"]}**')
            st.write(f'工程指标提示：{row["risk_reasons"]}')
            if geometry_rule.get("available"):
                st.write(f'尺寸联动估算：**{geometry_rule["state"]}**，最小局部余量 {geometry_rule["minimum_margin_m"]:.3f} m（{geometry_rule["affected_side"]}，里程 {geometry_rule["minimum_station_m"]:.2f} m）')
                st.caption(geometry_rule["explanation"])
            else:
                st.write(f'尺寸联动估算：**不提供结论**（{geometry_rule.get("reason", "缺少可信边界证据")}）')
            st.caption('规则提示来自开发集固定参考分布，不是学习模型的特征归因；模型/规则差值0.25是待验证的人工复核策略阈值。')
        with right:
            if runtime["model_available"]:
                st.metric("主排序风险", f'{float(row["model_risk"]):.3f}', row["risk_level"])
                st.caption(
                    f'学习模型风险 {float(row["model_risk"]):.3f}；'
                    f'规则交叉校验 {float(row["rule_risk"]):.3f}'
                )
                st.info(str(row["decision_status"]))
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
        st.markdown('<div class="section-kicker">验证结果回流</div>', unsafe_allow_html=True)
        st.caption("平台评估是预测记录；人工、闭环仿真和实车测试是独立验证记录。可信度按条件是否一致、结论是否核对、证据能否复查分层展示，不把测试方式换算成任意百分比。每次记录不覆盖旧结论。")
        with st.form(f"feedback_{selected_id}"):
            feedback_cols = st.columns(3)
            with feedback_cols[0]:
                feedback_result = st.selectbox("验证结果", ["待验证", "通过", "失败", "数据不足"], key=f"result_{selected_id}")
            with feedback_cols[1]:
                feedback_method = st.selectbox("验证方式", ["人工复核", "闭环仿真", "实车/现场测试", "其他测试平台"], key=f"method_{selected_id}")
            with feedback_cols[2]:
                severity = st.selectbox("事件严重程度", ["未分级", "一般", "严重"], key=f"severity_{selected_id}")
            condition_cols = st.columns(2)
            with condition_cols[0]:
                validation_environment = st.text_input("验证环境版本", value=active_task["environment_version"] if active_task else "历史条件未确认",
                                                       key=f"environment_{selected_id}")
                condition_consistency = st.selectbox("与本次评估条件", ["与本次评估一致", "存在变化", "未核对"], key=f"condition_{selected_id}")
            with condition_cols[1]:
                route_version = st.text_input("路线版本", value="原始上传" if active_task else "历史版本未确认",
                                              key=f"route_version_{selected_id}")
                evidence_completeness = st.selectbox("证据完整性", ["未核对", "有报告或日志", "仅口头结论"], key=f"evidence_{selected_id}")
            review_status = st.selectbox("结论复核状态", ["提交者填写", "已核对", "争议待复核"], key=f"review_{selected_id}")
            feedback_note = st.text_area("工程备注与条件变化", placeholder="例如：满载、低附着路面；左侧边界已更新", key=f"note_{selected_id}")
            evidence_file = st.file_uploader("验证报告或日志（可选；保存到本机任务证据包）",
                                             type=["pdf", "png", "jpg", "csv", "txt", "json", "log", "zip"],
                                             key=f"attachment_{selected_id}") if active_task else None
            submitted = st.form_submit_button("保存验证结果")
        if submitted:
            try:
                evidence_ref = save_attachment(active_task, evidence_file.name, evidence_file.getvalue()) if active_task and evidence_file else ""
                record_id = save_feedback({
                    "task_id": active_task["task_id"] if active_task else "",
                    "sample_id": selected_id, "route_sha256": row.get("route_file_sha256", ""),
                    "map_id": row.get("map_id", ""), "vehicle_structure": row.get("vehicle_structure", "unknown"),
                    "vehicle_config": effective_vehicle_config(geometry),
                    "environment_version": validation_environment, "route_version": route_version,
                    "severity": severity, "evidence_ref": evidence_ref,
                    "evidence_completeness": evidence_completeness,
                    "condition_consistency": condition_consistency, "review_status": review_status,
                    "feature_version": contract.get("feature_version", ""),
                    "schema_sha256": contract.get("schema_sha256", ""),
                    "model_name": runtime.get("model_name", "模型未加载"),
                    "model_risk": row.get("model_risk"), "rule_risk": row.get("rule_risk"),
                    "risk_level": row.get("risk_level", ""),
                    "queue_reason": str(row.get("mandatory_review_reason", "")) or str(row.get("validation_priority", "")),
                    "geometry_state": geometry_rule.get("state", "未评估"),
                    "geometry_min_margin_m": geometry_rule.get("minimum_margin_m"),
                    "validation_result": feedback_result, "validation_method": feedback_method,
                    "note": feedback_note or "",
                })
                st.success(f"验证记录 {record_id} 已保存；训练使用状态为待审核。")
            except (ValueError, OSError) as error:
                st.error(f"验证记录未保存：{error}")
        all_feedback = load_feedback(limit=10000)
        task_feedback = all_feedback[all_feedback["task_id"].fillna("") == active_task["task_id"]] if active_task else all_feedback[all_feedback["task_id"].fillna("") == ""]
        feedback_frame = task_feedback[task_feedback["sample_id"].astype(str) == selected_id]
        if not feedback_frame.empty:
            effective_config = effective_vehicle_config(geometry)
            current_key = make_condition_key(row.get("route_file_sha256", ""),
                                             active_task["environment_version"] if active_task else "历史条件未确认",
                                             "原始上传" if active_task else "历史版本未确认", effective_config)
            status = classify_disagreement(feedback_frame, row.get("model_risk"), current_key)
            if "冲突" in status or "漏判" in status:
                st.warning(status)
            else:
                st.info(status)
            st.dataframe(feedback_frame[["id", "created_at_utc", "validation_method", "validation_result",
                                        "severity", "environment_version", "route_version", "evidence_completeness",
                                        "condition_consistency", "review_status", "evidence_level", "training_review", "note"]],
                         width="stretch", hide_index=True)
            review_cols = st.columns([2, 2, 1])
            record_choice = review_cols[0].selectbox("选择记录审核", feedback_frame["id"].astype(int).tolist(),
                                                      key=f"review_id_{selected_id}")
            training_choice = review_cols[1].selectbox("训练数据审核", ["待审核", "可用于训练候选", "暂不采用"],
                                                        key=f"training_status_{selected_id}")
            if review_cols[2].button("更新审核", key=f"training_review_button_{selected_id}"):
                try:
                    set_training_review(record_choice, training_choice)
                    st.success("审核状态已更新。")
                except ValueError as error:
                    st.error(str(error))
        if active_task:
            st.download_button("导出本任务证据包 ZIP", export_task(active_task, task_feedback.to_csv(index=False).encode("utf-8-sig")),
                               file_name=f'pathguard_{active_task["task_id"]}.zip', mime="application/zip",
                               key=f"export_task_{selected_id}")
        approved = task_feedback[task_feedback["training_review"] == "可用于训练候选"]
        if not approved.empty:
            st.download_button("导出已审核训练候选记录 CSV", approved.to_csv(index=False).encode("utf-8-sig"),
                               file_name="pathguard_training_candidates.csv", mime="text/csv",
                               key=f"training_candidates_{selected_id}")
        st.caption("任务原件和附件保存在本机 LocalAppData/PathGuard/tasks；验证时间线保存在 SQLite。上传或导出不会自动重训模型。")

with evidence_tab:
    st.markdown("### 数据与模型基础")
    foundation_cols = st.columns(4)
    foundation_cols[0].metric("冻结监督记录", freeze["final_samples"])
    foundation_cols[1].metric("历史通过 / 失败", f'{freeze["passed"]} / {freeze["failed"]}')
    foundation_cols[2].metric("执行前特征", freeze["feature_count"])
    foundation_cols[3].metric("地图数量", freeze["map_count"])
    st.caption(f'当前模型：{runtime["model_name"] if runtime["model_available"] else "未加载"}。部署模型在340条冻结记录上重训；历史留出实验使用当时的隔离预测，页面内对这340条再次评分仅供查看，不代表独立验证。')
    if active_task:
        st.markdown("### 本任务的验证反馈与排序对照")
        task_audit = evaluate_verified_task(evaluated, load_feedback(limit=10000), active_task)
        if task_audit["status"] == "描述性结果":
            st.dataframe(pd.DataFrame(task_audit["rows"]), width="stretch", hide_index=True)
            st.caption(f'同任务已核对候选 {task_audit["verified_candidates"]} 条、失败 {task_audit["failures"]} 条；'
                       f'排除相互矛盾的候选 {task_audit["excluded_conflicts"]} 条。{task_audit["note"]}')
        else:
            st.info(f'{task_audit["status"]}：{task_audit["reason"]}')
        st.divider()
    st.markdown("### 冻结历史数据的独立验证依据")
    st.markdown('<div class="section-kicker">数据是否支持当前能力</div><h2 style="margin:.1rem 0 .25rem;color:#173f59;">验证证据：哪些结论已经有数据，哪些仍需补齐？</h2>', unsafe_allow_html=True)
    boundary = assets.get("boundary", {})
    if boundary:
        boundary_cols = st.columns(5)
        boundary_cols[0].metric("冻结NPZ可读取",f'{boundary["readable_npz"]} / {boundary["frozen_samples"]}')
        boundary_cols[1].metric("含完整边界坐标",boundary["full_boundary_samples"])
        boundary_cols[2].metric("距离语义校验通过",boundary["verified_boundary_semantics_samples"])
        boundary_cols[3].metric("完整点列筛查通过",assets.get("boundary_integrity",{}).get("orientation_and_continuity_screen_pass",0))
        boundary_cols[4].metric("含原配置车体净空",boundary["body_clearance_samples"])
        st.caption("距离语义只检查边界点到中心线的距离是否复现净空字段（误差≤1e-5m）。继续通过左右方向、次序和连续性筛查的只有3条；这仍不证明走廊无自交或完成连续碰撞检测。")
        coverage=pd.DataFrame(boundary["coverage_by_structure"])
        coverage["integrity_screen_passed"] = coverage["vehicle_structure"].map({"five_axis":3,"six_axis":0,"unknown":0}).fillna(0).astype(int)
        coverage=coverage[["vehicle_structure","samples","has_full_boundary","boundary_distance_semantics_verified","integrity_screen_passed","has_body_clearance","has_vehicle_profile_id"]].rename(columns={
            "vehicle_structure":"车辆结构","samples":"冻结样本","has_full_boundary":"完整边界","boundary_distance_semantics_verified":"距离语义通过","integrity_screen_passed":"完整点列筛查通过","has_body_clearance":"原配置车体净空","has_vehicle_profile_id":"车型配置ID"})
        st.dataframe(coverage,width="stretch",hide_index=True)
        st.subheader("空间特征实验状态")
        st.warning("旧实验预测文件中，55维基线和空间增广模型的逐条分数完全相同，且地图内预测恒定。因此旧实验不能回答空间特征是否有效；不将其并入训练模型。需先记录真实候选批次ID并重做独立对照。")
        st.divider()
    st.subheader('留出地图：模型、规则与混合排序对照')
    method_frame = assets["heldout_methods_v3"].copy()
    method_names = {
        "learning_model":"55维学习模型",
        "frozen_geometry_rule":"冻结开发集规则",
        "naive_uncalibrated_50_50_hybrid":"未校准50:50混合（仅对照）",
        "minimum_clearance":"最小静态净空",
    }
    method_frame["method_label"] = method_frame["method"].map(method_names)
    shown = method_frame[["method_label","budget","captured_failures","precision_at_k","failure_capture_rate","roc_auc_failure","pr_auc_failure","random_expected_captured"]]
    st.dataframe(shown.rename(columns={
        "method_label":"方法","budget":"验证预算","captured_failures":"捕获失败数",
        "precision_at_k":"入选失败比例","failure_capture_rate":"全部失败覆盖率",
        "roc_auc_failure":"失败ROC-AUC","pr_auc_failure":"失败PR-AUC",
        "random_expected_captured":"随机期望失败数",
    }),hide_index=True)
    st.caption("66条、5张已留出的地图；这是已检查留出集的事后重排，不是新独立验证。50:50分数混合未校准、未接入产品，也未据此选阈值。没有规划器调用ID，因此全局Top-K不等于同一次任务挑路线。")
    batch = assets.get("batch_proxy", {})
    if batch:
        st.markdown("#### 同地图、同确认车型的批次代理分析")
        proxy_rows=[]
        for item in batch["results"]:
            proxy_rows.append({
                "方法":method_names.get(item["method"],item["method"]),
                "代理组内20%名额捕获失败期望":item["expected_failures_captured_tie_aware"],
                "组内失败捕获率":item["capture_rate"],
                "相同名额随机期望":item["random_expected_failures_captured_same_group_budgets"],
            })
        st.dataframe(pd.DataFrame(proxy_rows),hide_index=True)
        st.warning(f"只有{batch['proxy_groups_with_at_least_two_candidates']}个地图×车型代理组，共{batch['proxy_candidates_in_eligible_groups']}条候选；该预算的随机期望高于各排序结果。缺少真实 planner_run_id/candidate_batch_id，当前不能证明模型带来同批挑选增益。")
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

    st.subheader("历史60次规则 / 模型地图分组审计")
    st.caption('历史审计采用旧规则定义；不作为本次固定开发集参考规则的验证结果。本次规则结果见上方同预算对比。')
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
        f'{audit["best_weight_by_mean_pr_auc"]:.2f}。页面主排序使用学习模型，规则作为独立工程交叉校验。'
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
        'queue_score = model_failure_risk\nrule_check = geometry_rule_risk\nconflict => manual_review',
        language="text",
    )
    st.caption("PathGuard只安排候选路线验证优先级；最终结论仍需闭环仿真、人工复核或实车验证。")
