"""Strict raw-route adapter; extraction reuses the frozen training feature definitions."""
import hashlib, importlib.util, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("pg_frozen_extractor",ROOT/"scripts/data_audit/extract_gate2_pre_execution_features_shanyu000.py")
extractor=importlib.util.module_from_spec(spec); spec.loader.exec_module(extractor)
ARRAYS={"s_m","z_m","kappa_1pm","v_profile_mps","left_clearance_m","right_clearance_m",
        "dkappa_ds_1pm2","steer_ff_rad","beta_ref_rad","yaw_rate_ref_radps","yaw_per_m_ref_1pm",
        "x_m","y_m","yaw_rad"}
def parse_route(payload, name, config=None):
    config=config or {}
    if not isinstance(config,dict):
        raise ValueError('配置JSON必须是对象')
    if len(payload)>40*1024*1024:
        raise ValueError("路线文件超过40MB，请缩小单次输入。")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if sum(x.file_size for x in archive.infolist())>150*1024*1024:
            raise ValueError("路线展开后超过150MB。")
    with np.load(io.BytesIO(payload),allow_pickle=False) as raw:
        arrays={key:raw[key] for key in raw.files if key in ARRAYS}
    structure=config.get("vehicle_structure","unknown")
    if structure not in {"five_axis","six_axis","unknown"}:
        raise ValueError("vehicle_structure 必须为 five_axis、six_axis 或 unknown")
    for key in ("length_m","width_m","mass_kg"):
        if key in config and (not np.isfinite(float(config[key])) or float(config[key])<=0):
            raise ValueError(key+" 必须为正有限数值")
    values, used, derived=extractor.extract_features(arrays,structure)
    digest=hashlib.sha256(payload).hexdigest()
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    identity=digest+":"+config_hash
    sid="upload_"+digest[:16]+"_"+config_hash[:8]
    values.update(sample_id=sid,route_file_sha256=digest,geometry_key=identity,
                  route_id=Path(name).stem,map_id=str(config.get("map_id","uploaded")),
                  vehicle_structure=structure,vehicle_structure_evidence="user_configuration",
                  label_status="unlabeled",true_label="",feature_source="npz_pre_execution_arrays_only",
                  feature_missing_reason=";".join(derived),feature_version=extractor.FEATURE_VERSION)
    geometry=None
    if "x_m" in arrays and "y_m" in arrays:
        n=len(arrays["s_m"])
        x=extractor.finite_1d(arrays["x_m"],"x_m",n)
        y=extractor.finite_1d(arrays["y_m"],"y_m",n)
        s=extractor.finite_1d(arrays["s_m"],"s_m",n)
        k=np.abs(arrays["kappa_1pm"])
        steer,_=extractor.vector_magnitude(arrays.get("steer_ff_rad",arrays.get("beta_ref_rad")),n)
        change=np.abs(extractor.derivative(steer,s))
        events=[]
        for label, series, unit, mode in [
            ("左侧最小净空",arrays["left_clearance_m"],"m","min"),
            ("右侧最小净空",arrays["right_clearance_m"],"m","min"),
            ("最大绝对曲率",k,"1/m","max"),
            ("最大转向变化",change,"rad/m","max")]:
            i=int(np.argmin(series) if mode=="min" else np.argmax(series))
            events.append(dict(label=label,index=i,s_m=float(s[i]),x_m=float(x[i]),y_m=float(y[i]),
                               value=float(series[i]),unit=unit,source="NPZ逐点工程指标"))
        idx=np.unique(np.r_[np.linspace(0,n-1,min(n,800),dtype=int),[e["index"] for e in events]])
        geometry=dict(source_sha256=digest,config_sha256=config_hash,vehicle_structure=structure,
             centerline=np.column_stack([x[idx],y[idx]]).tolist(),
             left_boundary=[],right_boundary=[],point_count_original=n,point_count_display=len(idx),
             events=events,coordinate_note="原始路线坐标；净空参考定义未确认时不反推道路边界。",
             used_arrays=used,derived_fields=derived)
    return pd.DataFrame([values]), identity, geometry

def render_evidence(st, geometry):
    if not geometry:
        st.info("没有原始 x_m/y_m 点列，无法定位风险位置；请上传含坐标的路线 NPZ。")
        return
    import plotly.graph_objects as go
    events=geometry["events"]
    selected=st.selectbox("定位工程关注点",range(len(events)),format_func=lambda i:events[i]["label"],
                          key="event_"+geometry["source_sha256"][:12])
    event=events[selected]; xy=np.asarray(geometry["centerline"])
    fig=go.Figure(go.Scatter(x=xy[:,0],y=xy[:,1],mode="lines",name="原始路线"))
    fig.add_trace(go.Scatter(x=[e["x_m"] for e in events],y=[e["y_m"] for e in events],
        mode="markers",text=[e["label"] for e in events],name="工程关注点"))
    fig.add_trace(go.Scatter(x=[event["x_m"]],y=[event["y_m"]],mode="markers",
        marker=dict(size=18,color="#d35435"),name="当前定位"))
    zoom=st.checkbox("放大当前关注点",key="zoom_"+geometry["source_sha256"][:12])
    if zoom:
        fig.update_xaxes(range=[event["x_m"]-15,event["x_m"]+15])
        fig.update_yaxes(range=[event["y_m"]-15,event["y_m"]+15])
    fig.update_layout(xaxis_title="x / m",yaxis_title="y / m",height=380)
    fig.update_yaxes(scaleanchor="x",scaleratio=1)
    st.plotly_chart(fig,use_container_width=True)
    st.write(f'{event["label"]}：**{event["value"]:.4g} {event["unit"]}**；路线里程 {event["s_m"]:.2f} m，原始点索引 {event["index"]}。')
    st.caption("这些点是逐点工程指标的极值位置，不是模型预测出的碰撞位置。"+geometry["coordinate_note"])
