"""Measured tracking replay; geometric sensitivity is never a failure prediction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st


def extra_lateral_extent(lateral, heading, front: float, rear: float, width: float):
    """Extra normal-direction extent of a rigid rectangle at matched station.

    Reference axes are the planned heading. This is not road clearance, a
    continuous swept volume, or a reconstruction of articulated dynamics.
    """
    if min(front, rear) < 0 or width <= 0:
        raise ValueError("车身前后距离须非负，车宽须大于零。")
    e, h = np.asarray(lateral, float), np.asarray(heading, float)
    if not np.isfinite(e).all() or not np.isfinite(h).all():
        raise ValueError("误差必须为有限数值。")
    corners = np.stack([e + x*np.sin(h) + y*np.cos(h)
                        for x in (front, -rear) for y in (-width/2, width/2)])
    left = corners.max(axis=0) - width/2
    right = -corners.min(axis=0) - width/2
    return np.maximum(np.maximum(left, right), 0)


@st.fragment
def render_tracking_evidence():
    import pandas as pd
    import plotly.graph_objects as go

    path = Path(__file__).parent / "data/tracking_execution_evidence.json"
    if not path.exists():
        st.info("执行遥测证据尚未导入。")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    st.subheader("规划路线与实际执行，相差在哪里？")
    st.write("这里叠加的是同一次历史平台运行中的计划路线与实际位置。先看跟踪偏差，再观察它对长车身横向占用的影响。")
    st.caption("独立历史遥测案例，不是340条冻结样本的新增训练记录；曲线是执行后实录，不是55维模型预测。")
    cases = data["cases"]
    index = st.selectbox("选择历史执行案例", range(len(cases)),
                         format_func=lambda i: cases[i]["display_name"], key="tracking_case")
    case = cases[index]
    series = pd.DataFrame(case["series"])
    stats = case["stats"]
    cols = st.columns(3)
    cols[0].metric("最大横向偏差", f'{stats["max_lateral_m"]:.3f} 米')
    cols[1].metric("95%帧的绝对偏差不超过", f'{stats["p95_lateral_m"]:.3f} 米')
    cols[2].metric("最大航向偏差", f'{stats["max_heading_deg"]:.2f}°')
    st.caption(f'统计使用全部 {stats["frames"]} 个有效遥测帧；图形仅抽点显示，保留误差峰值。帧数不等于独立实验次数。')
    position = st.slider("回放时刻", 0, len(series)-1, int(case["peak_display_index"]),
                         format="第 %d 个显示点", key=f"tracking_frame_{index}")
    row = series.iloc[position]
    st.write(f'当前时间 **{row.time_s:.2f} 秒**，路线里程 **{row.progress_s_m:.1f} 米**，横向偏差 **{row.lateral_m:.3f} 米**，航向偏差 **{row.heading_deg:.2f}°**。')
    local = st.checkbox("放大当前点附近30米", value=True, key="tracking_local")
    shown = series.loc[(series.progress_s_m-row.progress_s_m).abs() <= 15] if local else series
    fig = go.Figure()
    for x, y, name, color in [("plan_x", "plan_y", "计划参考轨迹", "#147f91"),
                              ("actual_x", "actual_y", "历史实际轨迹", "#d75935")]:
        fig.add_trace(go.Scatter(x=shown[x], y=shown[y], mode="lines", name=name,
                                line=dict(color=color, width=3)))
    fig.add_trace(go.Scatter(x=[row.plan_x, row.actual_x], y=[row.plan_y, row.actual_y],
                            mode="lines+markers", name="当前参考位置与实际位置", line=dict(color="#a86b18", dash="dot")))
    fig.update_layout(height=350, margin=dict(l=20,r=20,t=15,b=15),
                      legend=dict(orientation="h"), xaxis_title="相对东向位置 / 米", yaxis_title="相对北向位置 / 米",
                      yaxis=dict(scaleanchor="x",scaleratio=1), dragmode=False)
    st.plotly_chart(fig, use_container_width=True, config={"staticPlot":True,"displayModeBar":False}, key="tracking_xy")
    st.caption("两条轨迹采用同一坐标平移并保持横纵比例。点间连线用于观察位置差，不代表最近障碍物距离。")
    with st.expander("查看完整误差时间曲线"):
        chart = go.Figure(go.Scatter(x=series.time_s, y=series.lateral_m, mode="lines", name="重算横向误差"))
        chart.update_layout(height=240,xaxis_title="时间 / 秒",yaxis_title="横向误差 / 米",margin=dict(t=15,b=10))
        st.plotly_chart(chart,use_container_width=True,config={"staticPlot":True,"displayModeBar":False})
    st.markdown("#### 同样的偏差，为什么长车身需要更多关注？")
    st.write("把实录的横向与航向偏差作用于假设的刚性矩形车身，比较它在计划轨迹法向上的横向占用。这里不使用未经核对的道路边界。")
    a,b=st.columns(2)
    length=a.slider("假设车长 / 米",4.0,24.0,18.8,0.2,key="tracking_length")
    width=b.slider("假设车宽 / 米",2.0,5.0,3.74,0.02,key="tracking_width")
    half=length/2
    current=float(extra_lateral_extent(row.lateral_m,np.deg2rad(row.heading_deg),half,half,width))
    st.metric("当前横向外廓额外占用（几何假设）",f"{current:.3f} 米")
    def rectangle(e,h):
        x=np.array([-half,half,half,-half,-half]);y=np.array([-width/2,-width/2,width/2,width/2,-width/2])
        return x*np.cos(h)-y*np.sin(h), e+x*np.sin(h)+y*np.cos(h)
    body=go.Figure()
    for e,h,name,color in [(0,0,"理想姿态（假设尺寸）","#147f91"),
                           (row.lateral_m,np.deg2rad(row.heading_deg),"实录偏差下的姿态（假设尺寸）","#d75935")]:
        x,y=rectangle(e,h)
        body.add_trace(go.Scatter(x=x,y=y,mode="lines",name=name,line=dict(color=color,width=3)))
    body.update_layout(height=270,xaxis_title="沿计划航向 / 米",yaxis_title="横向占用 / 米",yaxis=dict(scaleanchor="x",scaleratio=1),
                       legend=dict(orientation="h"),margin=dict(t=15,b=10))
    st.plotly_chart(body,use_container_width=True,config={"staticPlot":True,"displayModeBar":False})
    st.info("这不是剩余道路净空，也不表示发生碰撞。默认尺寸参考备份六轴配置，但缺少与当次运行绑定的配置哈希，因此按假设参数处理；假定参考点在车身中心，仅分析横向偏差与航向变化。")
    with st.expander("有跟踪偏差，就代表实际碰撞了吗？"):
        st.write("不能这样判断。这两次归档的平台碰撞状态均为 clear（未报碰撞）。山路案例的文本边界检查与点云检查还给出了不同的余量结果；它们的环境语义没有对齐，不能把检查差异直接归因于跟踪误差。")
        checks=case['archived_checks']
        st.write(f"该案例归档值：文本边界余量 {checks['closed_loop_boundary_clearance_m']:.3f} 米；点云余量 {checks['closed_loop_pcd_clearance_m']:.3f} 米。它们是原报告中不同检查方式的结果，不是本轮重新计算的真实道路净空。")
        st.caption("后续需要同一环境版本、同一车体配置和一致的障碍物表示，才能定量分析执行后余量损失。")
    with st.expander("核对数据来源与计算方法"):
        st.json({"来源":"历史仿真平台执行遥测", "路线哈希":case["route_sha256"],
                 "遥测哈希":case["telemetry_sha256"], "证据状态":case["checks"],
                 "重算误差与记录误差的P95差异_米":stats["recorded_lateral_agreement_p95_m"],
                 "归档平台碰撞状态":case["archived_collision_status"],
                 "几何方法":"角点法向坐标=e_y+x*sin(e_psi)+y*cos(e_psi)，减去理想半车宽；不进行连续碰撞或铰接动力学计算"})
        st.caption("参考点按遥测里程与路线插值对齐；重算结果与控制器记录存在插值口径差异，已披露而非强制相等。")
    st.markdown("#### 原跟踪模型能直接接入吗？")
    st.write(data["model_decision"])
    with st.expander("查看重新计算的整图留出结果"):
        table=pd.DataFrame(data["shadow_validation"]["results"])
        st.dataframe(table[["held_out_map","confident_sample_fraction","lateral_p50_mae_m","lateral_p95_coverage"]].rename(columns={
            "held_out_map":"留出地图","confident_sample_fraction":"满足表格支持条件的帧比例",
            "lateral_p50_mae_m":"横向误差中位预测MAE / 米","lateral_p95_coverage":"P95预测界覆盖比例"}), hide_index=True)
        st.caption("支持条件是样本量与地图覆盖规则，不是统计置信度保证；覆盖率按相关遥测帧计，不能解释为独立场景成功率。")
