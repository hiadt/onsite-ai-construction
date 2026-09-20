#!/usr/bin/env python3
"""Execute the seven-step PathGuard data pipeline.

The script is deliberately metadata-first: it works from audit CSVs already tracked in
this repository and never treats a proxy/weak label as a true pass/fail result.
"""
from __future__ import annotations
import csv, json, math, argparse
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "analysis" / "data_recovery"
BASELINE = ROOT / "data" / "manifests" / "baseline_v1_359.csv"
OUT = ROOT / "data" / "derived" / "pathguard_v2"

TRUE_LABELS = {"passed", "failed", "true_pass", "true_fail"}
NUMERIC = ["maximum_slip_ratio", "maximum_tire_utilization", "dynamic_pcd_clearance_m", "maximum_abs_lateral_error_m", "maximum_abs_heading_error_deg"]

def read_csv(path: Path) -> list[dict[str,str]]:
    if not path.exists(): return []
    with path.open(encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))

def first(row, *names):
    for n in names:
        v = row.get(n, "")
        if v not in (None, ""): return str(v)
    return ""

def axis(row):
    raw = (first(row,"vehicle_structure","vehicle_profile_id","build_family","vehicle_type") + " " + first(row,"parse_notes")).lower()
    dim = first(row,"steer_ff_rad_second_dimension","steer_ff_rad_second_dim")
    if "six_axis" in raw or "six axis" in raw or "6axis" in raw or dim == "12": return "six_axis"
    if "five_axis" in raw or "five axis" in raw or "5axis" in raw or dim == "10": return "five_axis"
    return "unknown"

def key(row):
    return first(row,"route_file_sha256","route_input_hash","sample_key","sample_id","route_path_local","report_path_local")

def canonical(row, source, group):
    r = dict(row); r["source_member"] = source; r["data_group"] = group; r["vehicle_structure"] = axis(r)
    r["canonical_key"] = key(r)
    r["true_label"] = first(r,"true_label","label","outcome_label")
    if r["true_label"] not in TRUE_LABELS:
        hp = first(r,"hard_certificate_passed","recovered_passed","recovered_failed")
        if hp == "1" or hp.lower() in ("true","passed"): r["true_label"] = "passed"
        elif hp == "0" or hp.lower() in ("false","failed"): r["true_label"] = "failed"
        else: r["true_label"] = ""
    return r

def load():
    rows=[]
    for r in read_csv(BASELINE): rows.append(canonical(r,"baseline_v1_359","baseline"))
    for member in ("SHANYU000","ZYJisBoss"):
        root=AUDIT/member
        for name in ("local_candidate_samples.csv","new_candidate_samples.csv","duplicate_or_conflict_samples.csv","confirmed_new_unlabeled_routes.csv","confirmed_new_labeled_samples.csv","unresolved_candidates.csv"):
            for r in read_csv(root/name):
                status=first(r,"comparison_status","classification","recovery_status")
                if "aggregate" in status or "aggregate" in first(r,"parse_notes"): group="aggregate_only"
                elif "metric" in status or "metric" in first(r,"parse_notes"): group="metric_only"
                elif "unlabeled" in status or "no_result" in status or "unresolved" in status: group="route_only"
                elif first(r,"hard_certificate_passed","true_label"): group="real_labeled"
                else: group="route_only"
                rows.append(canonical(r,member,group))
        for name in ("SHANYU000_outcome_recovery.csv","ZYJisBoss_outcome_recovery.csv"):
            for r in read_csv(root/"outcome_recovery"/name):
                status=first(r,"outcome_recovery_status","recovery_status","comparison_status")
                group = "real_labeled" if first(r,"recovered_passed","recovered_failed","hard_certificate_passed") else ("metric_only" if "metric" in status else "route_only")
                rows.append(canonical(r,member,group))
    return rows

def dedup(rows):
    # Prefer actual labels, then metric evidence, then route metadata.
    rank={"real_labeled":4,"metric_only":3,"aggregate_only":2,"route_only":1,"baseline":0}
    out={}; dup=0
    for r in rows:
        k=r["canonical_key"]
        if not k: continue
        if k in out: dup+=1
        if k not in out or rank.get(r["data_group"],0)>rank.get(out[k]["data_group"],0): out[k]=r
    return list(out.values()), dup

def num(r,n):
    try: return float(r.get(n,""))
    except (TypeError,ValueError): return None

def risk(r):
    vals=[]; reasons=[]
    # normalized engineering warning score, only where metrics are present
    checks=[("maximum_slip_ratio",0.15,"滑移率偏高"),("maximum_tire_utilization",0.95,"轮胎利用率偏高"),("dynamic_pcd_clearance_m",0.20,"间隙偏小"),("maximum_abs_lateral_error_m",0.30,"横向误差偏大"),("maximum_abs_heading_error_deg",8.0,"航向误差偏大")]
    for n,t,reason in checks:
        v=num(r,n)
        if v is not None:
            vals.append(min(1,max(0,v/t)))
            if v>t: reasons.append(reason)
    if not vals:
        # route-only proxy: length/point count are prioritization hints only
        length=num(r,"route_length"); points=num(r,"point_count")
        if length is not None: vals.append(min(1,length/500.0))
        if points is not None: vals.append(min(1,points/5000.0))
    score=round(sum(vals)/len(vals),4) if vals else ""
    if score=="": level="unknown"
    elif score>=0.8: level="high"
    elif score>=0.5: level="medium"
    else: level="low"
    return score, level, ";".join(reasons)

def write(name, rows):
    if not rows: rows=[{"note":"no rows"}]
    fields=sorted({k for r in rows for k in r})
    with (OUT/name).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def main():
    global ROOT,AUDIT,BASELINE,OUT
    ap=argparse.ArgumentParser(); ap.add_argument("--root", type=Path, default=ROOT); args=ap.parse_args()
    ROOT=args.root.resolve(); AUDIT=ROOT/"analysis"/"data_recovery"; BASELINE=ROOT/"data"/"manifests"/"baseline_v1_359.csv"; OUT=ROOT/"data"/"derived"/"pathguard_v2"; OUT.mkdir(parents=True,exist_ok=True)
    rows, dup=dedup(load())
    groups={g:[r for r in rows if r["data_group"]==g] for g in ("real_labeled","metric_only","route_only","aggregate_only")}
    for g,rs in groups.items(): write(g+".csv",rs)
    scored=[]; weak=[]
    for r in rows:
        s,l,why=risk(r); q=dict(r); q.update(risk_score=s,risk_level=l,risk_reasons=why,risk_source="metrics" if any(num(r,n) is not None for n in NUMERIC) else "proxy")
        scored.append(q)
        if l in ("high","medium") and q["risk_source"]=="metrics" and q["data_group"] in ("metric_only","route_only"):
            w=dict(q); w.update(weak_label="likely_risk",weak_label_confidence=("high" if l=="high" else "medium"),weak_label_source="metric_threshold_v1"); weak.append(w)
    write("risk_scored_all.csv",scored); write("weak_labels.csv",weak)
    stats=Counter((r["vehicle_structure"],r["data_group"]) for r in rows)
    summary={"total_unique":len(rows),"duplicates_removed":dup,"groups":{g:len(v) for g,v in groups.items()},"axis_by_group":{g:{a:stats[(a,g)] for a in ("five_axis","six_axis","unknown")} for g in groups},"true_labeled_count":sum(bool(r["true_label"]) for r in rows),"weak_label_count":len(weak),"warning":"risk_score and weak_label are prioritization/model-development outputs, not recovered final pass/fail labels"}
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# PathGuard 七点执行摘要","",f"唯一记录数：{len(rows)}；按主键去重移除：{dup} 条。","", "|层级|数量|用途|", "|---|---:|---|", "|真实标签|%d|训练与验证，保留原始/回查证据|"%len(groups['real_labeled']), "|指标无最终标签|%d|风险评分与后续补证据|"%len(groups['metric_only']), "|只有路线/文件信息|%d|代理风险排序、主动补测|"%len(groups['route_only']), "|只有汇总证据|%d|展示总体能力，不进逐样本训练|"%len(groups['aggregate_only']), "", "## 五轴/六轴", ""]
    for g in groups:
        lines.append("- %s：五轴 %d，六轴 %d，未知 %d" % (g, stats[("five_axis",g)], stats[("six_axis",g)], stats[("unknown",g)]))
    lines += ["", "## 已落地的第 4—7 点", "", "- `risk_scored_all.csv`：首版工程指标风险分数；路线-only 行标记为 `proxy`。", "- `weak_labels.csv`：只保留指标充分且风险较高/中等的弱标签，和真实标签分开。", "- `demo/pathguard_demo.html`：静态原型，打开后可展示四层数据和风险队列。", "- 模型训练入口：先用 `real_labeled_samples.csv` 做基线，弱标签只用于扩充/排序实验，不覆盖真实标签。", "", "## 下一步", "", "1. 在拥有完整仓库数据的电脑运行 `python scripts/data_audit/execute_seven_steps.py`。", "2. 审查 `summary.json` 和四层 CSV，确认没有把同一路线重复计数。", "3. 用真实标签训练第一版分类器；再用弱标签做对比实验。", "4. 选前 20 条高风险路线做展示和客户共创闭环。"]
    (OUT/"七点执行摘要.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

