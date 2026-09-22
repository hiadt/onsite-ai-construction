"""Strict raw-route adapter; extraction reuses the frozen training feature definitions."""
import hashlib, importlib.util, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from geometry_integrity import screen_boundary_integrity
except ModuleNotFoundError:
    from demo.geometry_integrity import screen_boundary_integrity
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("pg_frozen_extractor",ROOT/"scripts/data_audit/extract_gate2_pre_execution_features_shanyu000.py")
extractor=importlib.util.module_from_spec(spec); spec.loader.exec_module(extractor)
ARRAYS={"s_m","z_m","kappa_1pm","v_profile_mps","left_clearance_m","right_clearance_m",
        "dkappa_ds_1pm2","steer_ff_rad","beta_ref_rad","yaw_rate_ref_radps","yaw_per_m_ref_1pm",
        "x_m","y_m","yaw_rad","left_boundary_xyz_m","right_boundary_xyz_m","body_clearance_m",
        "boundary_confidence","left_boundary_segment_id","right_boundary_segment_id"}


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


def compute_boundary_rule(geometry, length_m, width_m, reference_from_rear_m, review_margin_m=0.5):
    """Estimate local cross-section body margins only for verified, aligned boundaries."""
    calculation = geometry.get("calculation_geometry", geometry)
    left=np.asarray(calculation.get("left_boundary",[]),dtype=float)
    right=np.asarray(calculation.get("right_boundary",[]),dtype=float)
    center=np.asarray(calculation.get("centerline",[]),dtype=float)
    station=np.asarray(calculation.get("station_m",[]),dtype=float)
    if not geometry.get("boundary_integrity_screen_passed", False):
        return {"available":False,"reason":"边界距离检查虽通过，但完整点列的左右方向/连续性筛查未通过。"}
    if not geometry.get("boundary_semantics_verified", False):
        return {"available":False,"reason":"左右边界未通过中心线净空距离语义校验，拒绝尺寸联动判断。"}
    if not (len(center)>=2 and left.shape==center.shape and right.shape==center.shape and len(station)==len(center)):
        return {"available":False,"reason":"当前路线没有与中心线逐点对齐的左右边界坐标。"}
    if not (np.isfinite(center).all() and np.isfinite(left).all() and np.isfinite(right).all()
            and np.isfinite(station).all() and np.isfinite(np.asarray(calculation.get("yaw_rad",[]),dtype=float)).all()):
        return {"available":False,"reason":"路线、边界或航向含非有限值，拒绝输出尺寸联动判断。"}
    if np.any(np.diff(station)<=0):
        return {"available":False,"reason":"路线里程不是严格递增序列，无法进行前后车体边界插值。"}
    tangent=np.gradient(center,station,axis=0)
    tangent_norm=np.linalg.norm(tangent,axis=1)
    if np.any(tangent_norm<=1e-9):
        return {"available":False,"reason":"中心线存在零长度切向，无法稳定确定道路左右侧。"}
    tangent=tangent/tangent_norm[:,None]
    normal=np.column_stack([-tangent[:,1],tangent[:,0]])
    left_signed=np.sum((left-center)*normal,axis=1)
    right_signed=np.sum((right-center)*normal,axis=1)
    if np.mean(left_signed>0)<=0.99 or np.mean(right_signed<0)<=0.99:
        return {"available":False,"reason":"边界点未稳定落在路线左/右侧，左右语义或点列对应关系不可靠。"}
    if np.any(left_signed<=right_signed):
        return {"available":False,"reason":"左右边界在局部横断面上发生交叉或次序反转。"}
    length_m=float(length_m); width_m=float(width_m); reference_from_rear_m=float(reference_from_rear_m)
    corners_local=np.asarray([
        [length_m-reference_from_rear_m,width_m/2], [length_m-reference_from_rear_m,-width_m/2],
        [-reference_from_rear_m,-width_m/2], [-reference_from_rear_m,width_m/2],
    ])
    samples=[]
    for start,end in zip(corners_local,np.roll(corners_local,-1,axis=0)):
        for ratio in (0.0,0.25,0.5,0.75): samples.append(start+(end-start)*ratio)
    local=np.asarray(samples)
    yaw=np.asarray(calculation["yaw_rad"],dtype=float)
    c=np.cos(yaw)[:,None]; s=np.sin(yaw)[:,None]
    body_points=np.empty((len(center),len(local),2),dtype=float)
    body_points[:,:,0]=center[:,0,None]+c*local[None,:,0]-s*local[None,:,1]
    body_points[:,:,1]=center[:,1,None]+s*local[None,:,0]+c*local[None,:,1]
    target_station=np.clip(station[:,None]+local[None,:,0],station[0],station[-1])
    target=target_station.ravel()
    center_target=np.column_stack([np.interp(target,station,center[:,axis]) for axis in range(2)]).reshape(body_points.shape)
    left_target=np.column_stack([np.interp(target,station,left[:,axis]) for axis in range(2)]).reshape(body_points.shape)
    right_target=np.column_stack([np.interp(target,station,right[:,axis]) for axis in range(2)]).reshape(body_points.shape)
    left_vec=left_target-center_target; right_vec=right_target-center_target
    left_distance=np.linalg.norm(left_vec,axis=2); right_distance=np.linalg.norm(right_vec,axis=2)
    valid=np.isfinite(left_distance)&np.isfinite(right_distance)&(left_distance>1e-6)&(right_distance>1e-6)
    row_valid=np.all(valid,axis=1)
    if row_valid.sum()<2:
        return {"available":False,"reason":"边界坐标有效点不足，无法形成局部横断面。"}
    left_unit=left_vec/np.maximum(left_distance[:,:,None],1e-9)
    right_unit=right_vec/np.maximum(right_distance[:,:,None],1e-9)
    offsets=body_points-center_target
    left_point_margin=left_distance-np.sum(offsets*left_unit,axis=2)
    right_point_margin=right_distance-np.sum(offsets*right_unit,axis=2)
    left_margin=np.min(left_point_margin,axis=1)
    right_margin=np.min(right_point_margin,axis=1)
    residual=np.minimum(left_margin,right_margin)
    residual[~row_valid]=np.nan
    minimum_index=int(np.nanargmin(residual))
    violation=np.isfinite(residual)&(residual<0)
    low=np.isfinite(residual)&(residual<float(review_margin_m))
    ds=np.maximum(np.diff(station),0)
    violation_length=float(np.sum(ds*(violation[:-1]|violation[1:]))) if len(ds) else 0.0
    low_length=float(np.sum(ds*(low[:-1]|low[1:]))) if len(ds) else 0.0
    minimum=float(residual[minimum_index])
    affected_side="左侧" if left_margin[minimum_index]<=right_margin[minimum_index] else "右侧"
    if minimum<0:
        state="局部外廓估算越界"; priority="P0 人工复核"; action="优先复核对应位置；核实车辆参考点、边界和外廓后再安排仿真"
    elif minimum<float(review_margin_m):
        state="局部余量低于复核阈值"; priority="P1 优先补测"; action="复核低余量区间，确认边界更新和定位误差后再验证"
    else:
        state="局部横断面估算余量充足"; priority="按模型队列安排"; action="保留模型排序，按既定验证预算推进"
    confidence=np.asarray(calculation.get("boundary_confidence",geometry.get("boundary_confidence",[])),dtype=float)
    confidence_min=float(np.nanmin(confidence)) if confidence.size==len(center) and np.isfinite(confidence).any() else None
    return {
        "available":True,"method":"rigid_body_boundary_cross_section_v3","state":state,"priority":priority,"action":action,
        "review_margin_m":float(review_margin_m),"minimum_margin_m":minimum,"minimum_index":minimum_index,
        "minimum_station_m":float(station[minimum_index]),"affected_side":affected_side,
        "violation_point_count":int(violation.sum()),"violation_length_m":violation_length,
        "low_margin_point_count":int(low.sum()),"low_margin_length_m":low_length,
        "left_margin_m":left_margin.tolist(),"right_margin_m":right_margin.tolist(),
        "residual_margin_m":residual.tolist(),"boundary_confidence_min":confidence_min,
        "explanation":"基于原始全分辨率边界点列；该点列已通过距离一致性、左右朝向/顺序和保守连续性筛查。沿车体四边取16个外廓点，按纵向位置映射到局部路线里程，计算横断面估算余量。它不检查完整二维走廊拓扑、边界自交、所有车身表面或点间连续碰撞，也不是碰撞概率或安全认证。",
    }
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
    for key in ("length_m","width_m","reference_from_rear_m"):
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
        event_specs=[
            ("左侧最小净空",arrays["left_clearance_m"],"m","min"),
            ("右侧最小净空",arrays["right_clearance_m"],"m","min"),
            ("最大绝对曲率",k,"1/m","max"),
            ("最大转向变化",change,"rad/m","max")]
        if "body_clearance_m" in arrays:
            event_specs.insert(2,("原配置最小车体净空",arrays["body_clearance_m"],"m","min"))
        events=[]
        for label, series, unit, mode in event_specs:
            i=int(np.argmin(series) if mode=="min" else np.argmax(series))
            events.append(dict(label=label,index=i,s_m=float(s[i]),x_m=float(x[i]),y_m=float(y[i]),
                               value=float(series[i]),unit=unit,source="NPZ逐点工程指标"))
        yaw_source=arrays["yaw_rad"] if "yaw_rad" in arrays else np.unwrap(np.arctan2(np.gradient(y),np.gradient(x)))
        yaw=extractor.finite_1d(yaw_source,"yaw_rad_or_xy_derived",n)
        idx=np.unique(np.r_[np.linspace(0,n-1,min(n,800),dtype=int),[e["index"] for e in events]])
        left_boundary=[]; right_boundary=[]; boundary_confidence=[]
        boundary_semantics_verified=False; boundary_integrity_screen_passed=False
        boundary_integrity={}; calculation_geometry=None
        boundary_keys={"left_boundary_xyz_m","right_boundary_xyz_m"}
        if boundary_keys.issubset(arrays):
            left_raw=np.asarray(arrays["left_boundary_xyz_m"],dtype=float)
            right_raw=np.asarray(arrays["right_boundary_xyz_m"],dtype=float)
            if left_raw.ndim==2 and right_raw.ndim==2 and left_raw.shape[0]==n and right_raw.shape[0]==n and left_raw.shape[1]>=2 and right_raw.shape[1]>=2:
                if np.isfinite(left_raw[:,:2]).all() and np.isfinite(right_raw[:,:2]).all():
                    center_full=np.column_stack([x,y])
                    boundary_integrity=screen_boundary_integrity(
                        center_full,s,left_raw[:,:2],right_raw[:,:2],
                        arrays["left_clearance_m"],arrays["right_clearance_m"])
                    boundary_semantics_verified=bool(boundary_integrity.get("distance_semantics_pass",False))
                    boundary_integrity_screen_passed=bool(boundary_integrity.get("integrity_screen_pass",False))
                    if boundary_semantics_verified and boundary_integrity_screen_passed:
                        left_boundary=left_raw[idx,:2].tolist(); right_boundary=right_raw[idx,:2].tolist()
                        calc_confidence=(extractor.finite_1d(arrays["boundary_confidence"],"boundary_confidence",n)
                                         if "boundary_confidence" in arrays else np.full(n,np.nan))
                        calculation_geometry=dict(
                            centerline=center_full.tolist(),yaw_rad=yaw.astype(float).tolist(),station_m=s.astype(float).tolist(),
                            left_boundary=left_raw[:,:2].tolist(),right_boundary=right_raw[:,:2].tolist(),
                            boundary_confidence=calc_confidence.astype(float).tolist())
        if "boundary_confidence" in arrays:
            confidence=extractor.finite_1d(arrays["boundary_confidence"],"boundary_confidence",n)
            boundary_confidence=confidence[idx].astype(float).tolist()
        body_clearance=[]
        if "body_clearance_m" in arrays:
            body=extractor.finite_1d(arrays["body_clearance_m"],"body_clearance_m",n)
            body_clearance=body[idx].astype(float).tolist()
        has_boundaries=bool(left_boundary and right_boundary)
        geometry=dict(source_sha256=digest,config_sha256=config_hash,vehicle_structure=structure,
             centerline=np.column_stack([x[idx],y[idx]]).tolist(),
             yaw_rad=yaw[idx].astype(float).tolist(),
             station_m=s[idx].astype(float).tolist(),point_indices=idx.astype(int).tolist(),
             left_boundary=left_boundary,right_boundary=right_boundary,boundary_confidence=boundary_confidence,
             boundary_semantics_verified=boundary_semantics_verified,
             boundary_integrity_screen_passed=boundary_integrity_screen_passed,
             boundary_integrity=boundary_integrity,calculation_geometry=calculation_geometry,
             body_clearance_m=body_clearance,point_count_original=n,point_count_display=len(idx),
             events=events,coordinate_note=("原始路线左右边界通过距离、方向、顺序和保守连续性筛查；尺寸联动仅作局部横断面估算。" if has_boundaries else "原始路线坐标；当前文件没有通过完整边界完整性筛查的左右边界点列。"),
             used_arrays=used,derived_fields=derived,vehicle_parameters=config)
    return pd.DataFrame([values]), identity, geometry

def render_evidence(st, geometry):
    if not geometry:
        st.info("没有原始 x_m/y_m 点列，无法定位风险位置；请上传含坐标的路线 NPZ。")
        return
    import plotly.graph_objects as go
    events=geometry["events"]
    structure=geometry.get("vehicle_structure","unknown")
    defaults={"five_axis":(12.0,3.2,5.0),"six_axis":(15.0,3.4,6.0),"unknown":(12.0,3.2,5.0)}[structure]
    supplied=geometry.get("vehicle_parameters",{})
    has_dimensions=(all(key in supplied for key in ("length_m","width_m","reference_from_rear_m"))
                    and supplied.get("dimension_source")!="illustrative_demo_values")
    if not has_dimensions:
        if supplied.get("dimension_source")=="illustrative_demo_values":
            st.warning("当前配置标为示例假设尺寸。几何图只表示这些输入下的条件性结果，不代表真实车型参数。")
        else:
            st.warning("该路线配置未提供车长、车宽和参考点的确认值。以下为可编辑的示例假设值，几何图不代表真实车型参数。")
    st.markdown("#### 米制刚性车体扫掠")
    st.caption("调整车身长宽和参考点后，系统会重算四角轨迹；仅在左右边界语义检查通过时，才估算局部横断面余量。该计算不含车辆动力学，也不覆盖完整走廊拓扑。")
    cols=st.columns(4)
    token=geometry["source_sha256"][:12]
    length=cols[0].number_input("车长 / m",3.0,30.0,float(supplied.get("length_m",defaults[0])),0.1,key="length_"+token)
    width=cols[1].number_input("车宽 / m",1.0,8.0,float(supplied.get("width_m",defaults[1])),0.1,key="width_"+token)
    ref_default=min(float(supplied.get("reference_from_rear_m",defaults[2]))/length,1.0)
    reference_ratio=cols[2].slider("参考点距车尾 / %车长",0,100,int(round(ref_default*100)),1,key="reference_"+token)
    review_margin=cols[3].number_input("复核余量 / m",0.0,3.0,0.5,0.1,key="margin_"+token,
                                      help="用于标记低余量区间的演示阈值，不是法规或安全认证阈值。")
    reference=length*reference_ratio/100.0
    corner_x,corner_y=compute_rigid_body_sweep(geometry,length,width,reference)
    xy=np.asarray(geometry["centerline"])
    retained=np.asarray(geometry.get("point_indices",range(len(xy))))
    dimensions_enabled=st.checkbox(
        "确认使用当前车长、车宽和参考点计算边界余量",
        value=has_dimensions,
        key="use_dimensions_"+token,
        help="未启用时仍可查看条件性车体扫掠图，但不输出尺寸联动边界结论。",
    )
    boundary_rule=(compute_boundary_rule(geometry,length,width,reference,review_margin)
                   if dimensions_enabled else {"available":False,"reason":"车辆尺寸尚未确认。"})
    location_events=list(events)
    if boundary_rule["available"]:
        boundary_pose_display=int(np.argmin(np.abs(np.asarray(geometry["station_m"])-boundary_rule["minimum_station_m"])))
        location_events.insert(0,dict(label="可调尺寸最小边界余量",index=int(retained[boundary_pose_display]),
            s_m=boundary_rule["minimum_station_m"],x_m=float(xy[boundary_pose_display,0]),y_m=float(xy[boundary_pose_display,1]),
            value=boundary_rule["minimum_margin_m"],unit="m",source="车体包络与逐点边界实时计算"))
    selected=st.selectbox("定位工程关注点",range(len(location_events)),format_func=lambda i:location_events[i]["label"],
                          key="event_"+token)
    event=location_events[selected]
    st.info(f'当前定位：{event["label"]}，数值 {event["value"]:.4g} {event["unit"]}，位于路线里程 {event["s_m"]:.2f} m。车身外廓会同步显示在最近路线参考点处。')
    pose=int(np.argmin(np.abs(retained-int(event["index"]))))
    fig=go.Figure(go.Scatter(x=xy[:,0],y=xy[:,1],mode="lines",name="原始路线"))
    if boundary_rule["available"]:
        left_boundary=np.asarray(geometry["left_boundary"]); right_boundary=np.asarray(geometry["right_boundary"])
        fig.add_trace(go.Scatter(x=left_boundary[:,0],y=left_boundary[:,1],mode="lines",name="左侧边界",line=dict(color="#778996",width=2)))
        fig.add_trace(go.Scatter(x=right_boundary[:,0],y=right_boundary[:,1],mode="lines",name="右侧边界",line=dict(color="#778996",width=2)))
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
    if boundary_rule["available"]:
        bp=boundary_rule["minimum_index"]
        fig.add_trace(go.Scatter(x=[xy[bp,0]],y=[xy[bp,1]],mode="markers",
            marker=dict(size=15,color="#ff9d2e",symbol="diamond"),name="尺寸联动最小余量"))
    zoom=st.checkbox("放大当前关注点",key="zoom_"+geometry["source_sha256"][:12])
    if zoom:
        fig.update_xaxes(range=[event["x_m"]-15,event["x_m"]+15])
        fig.update_yaxes(range=[event["y_m"]-15,event["y_m"]+15])
    fig.update_layout(xaxis_title="x / m",yaxis_title="y / m",height=430,
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0),
        margin=dict(l=20,r=20,t=78,b=20))
    fig.update_yaxes(scaleanchor="x",scaleratio=1)
    st.plotly_chart(fig,use_container_width=True)
    st.caption("图例：路线实线为参考点轨迹；四条点线为车体四角轨迹；红色多边形为当前位置外廓；灰线为NPZ逐点边界；橙色菱形为当前尺寸下的最小余量位置。")
    front=length-reference
    st.write(f'{event["label"]}：**{event["value"]:.4g} {event["unit"]}**；路线里程 {event["s_m"]:.2f} m，原始点索引 {event["index"]}。')
    st.caption(f"几何设置：参考点前方 {front:.2f} m、后方 {reference:.2f} m、半宽 {width/2:.2f} m。")
    if boundary_rule["available"]:
        metric_cols=st.columns(4)
        metric_cols[0].metric("最小剩余余量",f'{boundary_rule["minimum_margin_m"]:.2f} m',boundary_rule["affected_side"])
        metric_cols[1].metric("越界区间",f'{boundary_rule["violation_length_m"]:.1f} m',f'{boundary_rule["violation_point_count"]} 个显示点')
        metric_cols[2].metric("低余量区间",f'{boundary_rule["low_margin_length_m"]:.1f} m',f'阈值 {review_margin:.1f} m')
        metric_cols[3].metric("尺寸联动规则",boundary_rule["state"],boundary_rule["priority"])
        message=f'{boundary_rule["state"]}：{boundary_rule["action"]}。最小余量位于 {boundary_rule["minimum_station_m"]:.2f} m，靠近{boundary_rule["affected_side"]}边界。'
        if boundary_rule["state"]=="包络越界": st.error(message)
        elif boundary_rule["state"]=="边界余量不足": st.warning(message)
        else: st.success(message)
        confidence=boundary_rule.get("boundary_confidence_min")
        confidence_text=f'；边界置信度最小值 {confidence:.2f}' if confidence is not None else "；文件未提供逐点边界置信度"
        st.caption(boundary_rule["explanation"]+confidence_text+"。该结果是确定性几何规则，不是学习模型概率。")
    else:
        baseline=np.asarray(geometry.get("body_clearance_m",[]),dtype=float)
        baseline_text=f' 原配置最小车体净空为 {np.nanmin(baseline):.2f} m；改变尺寸后不能复用该数值。' if baseline.size else ""
        st.info(boundary_rule["reason"]+baseline_text+" 当前仅展示扫掠包络，并将尺寸联动规则标记为证据不足。")
    st.info("决策链：冻结55维模型负责历史失败风险排序；可调尺寸几何规则负责边界越界与低余量检查。几何越界触发P0复核，但不会伪改学习模型概率。")
    st.caption("四条角点轨迹和红色车体外廓按米制坐标及逐点航向角重算。它们是刚性矩形几何扫掠，不包含铰接、轮胎侧偏、悬架、载荷转移或制动动力学。"+geometry["coordinate_note"])
