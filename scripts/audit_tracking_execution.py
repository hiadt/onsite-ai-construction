"""Rebuild measured execution evidence from hash-bound platform archives.

Run after the original tracking-table validation, passing its new output.
Original archives are read-only. No label or deployed model is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root, validation, output, audit_output):
    cases, audit, seen = [], [], set()
    for metadata_path in sorted((root / "platform_runs").rglob("replay.json")):
        directory = metadata_path.parent.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        entry = {"archive":str(metadata_path.relative_to(root)).replace('\\','/'),"map":metadata["map_name"]}
        try:
            route_path = directory / "input" / Path(metadata["route"].replace('\\','/')).name
            telemetry_path = directory / "input" / Path(metadata["source_telemetry"].replace('\\','/')).name
            route_sha, telemetry_sha = sha(route_path), sha(telemetry_path)
            if route_sha != metadata["route_sha256"] or telemetry_sha != metadata["source_telemetry_sha256"]:
                raise ValueError("replay metadata hash mismatch")
            with np.load(telemetry_path,allow_pickle=False) as raw:
                recorded = json.loads(str(raw["metadata_json"]))
                if recorded["route_sha256"] != route_sha:
                    raise ValueError("telemetry did not record this route hash")
                columns = {key:raw["scalar_values"][:,i].astype(float) for i,key in enumerate(recorded["scalar_fields"])}
            manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
            for file in (route_path,telemetry_path,metadata_path.with_suffix('.npz'),directory/'platform_summary.json'):
                key = str(file.relative_to(directory)).replace('/','\\')
                expected = manifest["files"][key]["sha256"]
                if sha(file) != expected: raise ValueError("archive manifest mismatch")
            identity = (route_sha,telemetry_sha)
            if identity in seen:
                entry["status"]="duplicate_same_run"; audit.append(entry); continue
            seen.add(identity)
            with np.load(route_path,allow_pickle=False) as z:
                route={key:z[key].copy() for key in ('s_m','x_m','y_m','yaw_rad')}
            if not all(np.isfinite(v).all() for v in route.values()):
                raise ValueError("route_pose_contains_nonfinite_values; continuous heading interpolation disabled")
            if not np.all(np.diff(route['s_m'])>0): raise ValueError("route station not strictly increasing")
            with np.load(metadata_path.with_suffix('.npz'),allow_pickle=False) as z:
                telemetry={key:z[key].copy() for key in z.files if key!='metadata_json'}
            for key in ('x_m','y_m','yaw_rad','progress_s_m','lateral_error_m','heading_error_rad'):
                if not np.array_equal(columns[key],telemetry[key]): raise ValueError("replay differs from original telemetry: "+key)
            s=telemetry['progress_s_m'];t=telemetry['time_s']
            if not np.all(np.diff(t)>0): raise ValueError("time not strictly increasing")
            if np.any((s<route['s_m'][0])|(s>route['s_m'][-1])): raise ValueError("station outside route")
            if np.any(telemetry['special_action_active']): raise ValueError("special action needs separate pose alignment")
            x=np.interp(s,route['s_m'],route['x_m']);y=np.interp(s,route['s_m'],route['y_m'])
            yaw=np.interp(s,route['s_m'],np.unwrap(route['yaw_rad']))
            lat=-(telemetry['x_m']-x)*np.sin(yaw)+(telemetry['y_m']-y)*np.cos(yaw)
            heading=np.arctan2(np.sin(telemetry['yaw_rad']-yaw),np.cos(telemetry['yaw_rad']-yaw))
            ref_error=np.hypot(x-telemetry['recorded_reference_x_m'],y-telemetry['recorded_reference_y_m'])
            if np.quantile(ref_error,.95)>.001: raise ValueError("recorded reference and route disagree")
            agreement=np.abs(lat-telemetry['lateral_error_m'])
            if np.quantile(agreement,.95)>.02: raise ValueError("lateral error reconciliation exceeds 2 cm")
            arrays={'time_s':t,'progress_s_m':s,'plan_x':x-x[0],'plan_y':y-y[0],
                    'actual_x':telemetry['x_m']-x[0],'actual_y':telemetry['y_m']-y[0],
                    'lateral_m':lat,'heading_deg':np.rad2deg(heading)}
            if not all(np.isfinite(v).all() for v in arrays.values()): raise ValueError("nonfinite telemetry")
            peak=int(np.argmax(abs(lat)))
            indices=np.unique(np.r_[np.linspace(0,len(t)-1,min(len(t),700),dtype=int),peak,np.argmax(abs(heading))])
            summary=json.loads((directory/'platform_summary.json').read_text(encoding='utf-8'))
            case={'display_name':{'mounarea_1':'山路历史运行：转弯跟踪偏差','ramp_1':'坡道历史运行：较小偏差对照'}.get(metadata['map_name'],metadata['map_name']),
                  'map':metadata['map_name'],'route_sha256':route_sha,'telemetry_sha256':telemetry_sha,
                  'archived_collision_status':metadata.get('collision_status','unknown'),
                  'archived_checks':{key:summary.get(key) for key in ('completed','platform_collision_status','closed_loop_boundary_clearance_m','closed_loop_pcd_clearance_m','hard_passed','hard_failure_reasons')},
                  'checks':{'route_hash_in_original_telemetry':True,'archive_manifest_hashes':True,'raw_replay_channels_identical':True,
                            'runtime_vehicle_config_bound':False,'trusted_boundary_bound':False},
                  'stats':{'frames':len(t),'max_lateral_m':float(np.max(abs(lat))),'p95_lateral_m':float(np.quantile(abs(lat),.95)),
                           'max_heading_deg':float(np.max(abs(np.rad2deg(heading)))),
                           'reference_alignment_p95_m':float(np.quantile(ref_error,.95)),
                           'recorded_lateral_agreement_p95_m':float(np.quantile(agreement,.95)),
                           'recorded_heading_agreement_p95_deg':float(np.rad2deg(np.quantile(abs(np.arctan2(np.sin(heading-telemetry['heading_error_rad']),np.cos(heading-telemetry['heading_error_rad']))),.95)))},
                  'peak_display_index':int(np.flatnonzero(indices==peak)[0]),
                  'series':[{k:round(float(v[i]),7) for k,v in arrays.items()} for i in indices]}
            cases.append(case);entry.update(status='accepted_measured_tracking',stats=case['stats'],route_sha256=route_sha,telemetry_sha256=telemetry_sha)
        except (ValueError,KeyError,OSError) as error:
            entry.update(status='excluded',reason=str(error))
        audit.append(entry)
    if not cases: raise RuntimeError("No run passes provenance and pose checks")
    report=json.loads(validation.read_text(encoding='utf-8'))
    decision='原跟踪响应表已重新执行11张地图整图留出。按地图平均，P95横向预测界实际覆盖约82.8%；山路2覆盖约54.6%，表格支持条件覆盖约10.3%。目前仅适合影子研究，不能作为95%保障的误差包络，也没有接入当前55维分类模型。'
    asset={'schema':'pathguard_measured_tracking_v1','cases':cases,'model_decision':decision,'shadow_validation':report,
           'validation_source_sha256':sha(validation),
           'validation_inputs_sha256':{p.name:sha(p) for p in (root/'builds/tracking_model_v1/tracking_samples_v1.npz',root/'builds/tracking_model_v1/run_index_v1.json',root/'tools/validate_tracking_response_table.py',root/'tools/build_tracking_response_table.py')},
           'scope':'Measured historical platform replay; geometry uses assumed dimensions; no clearance/collision label inferred.'}
    output.parent.mkdir(parents=True,exist_ok=True);audit_output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(asset,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    audit_output.write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'accepted_cases':len(cases),'archives':len(audit),'statuses':[a['status'] for a in audit]},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    args=parser.parse_args()
    build(args.source,args.validation,args.output,args.audit)
