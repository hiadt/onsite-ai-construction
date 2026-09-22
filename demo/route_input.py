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


def compute_rigid_body_sweep(geometry, length_m, width_m, reference_from_rear_m):
    """Return four corner trajectories and one vehicle pose in world metres."""
    length_m=float(length_m); width_m=float(width_m); reference_from_rear_m=float(reference_from_rear_m)
    if not (length_m > 0 and width_m > 0 and 0 <= reference_from_rear_m <= length_m):
        raise ValueError("车长、车宽须为正数，参考点距车尾须位于0到车长之间。")
    xy=np.asarray(geometry["centerline"],dtype=float)
    yaw=np.asarray(geometry["yaw_rad"],dtype=float)
    if len(xy) != len(yaw) or len(xy) < 2:
        raise ValueError("路线坐标与航向角数量不一致。")
    local=np.asarray([
        [length_m-reference_from_rear_m, width_m/2],
        [length_m-reference_from_rear_m,-width_m/2],
        [-reference_from_rear_m,-width_m/2],
        [-reference_from_rear_m, width_m/2],
    ])
    c=np.cos(yaw)[:,None]; s=np.sin(yaw)[:,None]
    corner_x=xy[:,0,None]+c*local[None,:,0]-s*local[None,:,1]
    corner_y=xy[:,1,None]+s*local[None,:,0]+c*local[None,:,1]
    return corner_x,corner_y
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
    for key in ("length_m","width_m","mass_kg","reference_from_rear_m"):
        if key in config and (not np.isfinite(float(config[key])) or float(config[key])<=0):
            raise ValueError(key+" 必须为正有限数值")
    if "length_m" in config and "reference_from_rear_m" in config and float(config["reference_from_rear_m"]) > float(config["length_m"]):
        raise ValueError("reference_from_rear_m 不能大于 length_m")
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
        yaw_source=arrays["yaw_rad"] if "yaw_rad" in arrays else np.unwrap(np.arctan2(np.gradient(y),np.gradient(x)))
        yaw=extractor.finite_1d(yaw_source,"yaw_rad_or_xy_derived",n)
        idx=np.unique(np.r_[np.linspace(0,n-1,min(n,800),dtype=int),[e["index"] for e in events]])
        geometry=dict(source_sha256=digest,config_sha256=config_hash,vehicle_structure=structure,
             centerline=np.column_stack([x[idx],y[idx]]).tolist(),
             yaw_rad=yaw[idx].astype(float).tolist(),
             station_m=s[idx].astype(float).tolist(),point_indices=idx.astype(int).tolist(),
             left_boundary=[],right_boundary=[],point_count_original=n,point_count_display=len(idx),
             events=events,coordinate_note="原始路线坐标；净空参考定义未确认时不反推道路边界。",
             used_arrays=used,derived_fields=derived,vehicle_parameters=config)
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
    structure=geometry.get("vehicle_structure","unknown")
    defaults={"five_axis":(12.0,3.2,5.0,35000),"six_axis":(15.0,3.4,6.0,50000),"unknown":(12.0,3.2,5.0,35000)}[structure]
    supplied=geometry.get("vehicle_parameters",{})
    st.markdown("#### 真实尺度刚性车体扫掠")
    st.caption("参数可用于几何外廓重算；预设值仅供演示。重量不进入当前55维模型，也不改变风险分数。")
    cols=st.columns(4)
    token=geometry["source_sha256"][:12]
    length=cols[0].number_input("车长 / m",3.0,30.0,float(supplied.get("length_m",defaults[0])),0.1,key="length_"+token)
    width=cols[1].number_input("车宽 / m",1.0,8.0,float(supplied.get("width_m",defaults[1])),0.1,key="width_"+token)
    ref_default=min(float(supplied.get("reference_from_rear_m",defaults[2]))/length,1.0)
    reference_ratio=cols[2].slider("参考点距车尾 / %车长",0,100,int(round(ref_default*100)),1,key="reference_"+token)
    reference=length*reference_ratio/100.0
    mass=cols[3].number_input("总质量 / kg",1000,200000,int(supplied.get("mass_kg",defaults[3])),1000,key="mass_"+token)
    corner_x,corner_y=compute_rigid_body_sweep(geometry,length,width,reference)
    retained=np.asarray(geometry.get("point_indices",range(len(xy))))
    pose=int(np.argmin(np.abs(retained-int(event["index"]))))
    fig=go.Figure(go.Scatter(x=xy[:,0],y=xy[:,1],mode="lines",name="原始路线"))
    corner_names=["左前角","右前角","右后角","左后角"]
    for i,name in enumerate(corner_names):
        fig.add_trace(go.Scatter(x=corner_x[:,i],y=corner_y[:,i],mode="lines",line=dict(width=1,dash="dot"),name=name+"扫掠轨迹"))
    polygon_order=[0,1,2,3,0]
    fig.add_trace(go.Scatter(x=corner_x[pose,polygon_order],y=corner_y[pose,polygon_order],mode="lines",fill="toself",name="当前车体外廓",line=dict(width=3,color="#d35435")))
    fig.add_trace(go.Scatter(x=[xy[pose,0]],y=[xy[pose,1]],mode="markers",marker=dict(size=10,color="#172b3a"),name="路线参考点"))
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
    front=length-reference
    st.write(f'{event["label"]}：**{event["value"]:.4g} {event["unit"]}**；路线里程 {event["s_m"]:.2f} m，原始点索引 {event["index"]}。')
    st.caption(f"几何设置：参考点前方 {front:.2f} m、后方 {reference:.2f} m、半宽 {width/2:.2f} m；质量 {mass:,} kg 仅记录，不参与几何或模型计算。")
    st.caption("四条角点轨迹和红色车体外廓按米制坐标及逐点航向角重算。它们是刚性矩形几何扫掠，不包含铰接、轮胎侧偏、悬架、载荷转移或制动动力学。"+geometry["coordinate_note"])
