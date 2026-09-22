"""Reproduce frozen-reference engineering comparison on the existing held-out maps."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
from rule_baseline import required_rule_features, compute_geometry_rule_risk
from inference import load_gate3_model, predict_failure_risk

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    df=pd.read_csv(args.features)
    contract=json.loads((ROOT/"demo/data/feature_contract_v2.json").read_text(encoding="utf-8"))
    original=json.loads((ROOT/"demo/data/gate3_test_metrics.json").read_text(encoding="utf-8"))
    dev=df[df.map_id.isin(original["development_maps"])].copy()
    test=df[df.map_id.isin(original["test_maps"])].copy()
    assert len(df)==340 and len(test)==66 and len(dev)==274
    assert not set(dev.map_id)&set(test.map_id)
    assert df.sample_id.is_unique
    ref={"version":"development_ecdf_v1","source_sha256":hashlib.sha256(args.features.read_bytes()).hexdigest(),
         "n_reference":len(dev),"development_maps":sorted(dev.map_id.unique()),
         "features":{c:sorted(dev[c].astype(float).tolist()) for c in required_rule_features()}}
    (ROOT/"demo/data/rule_reference.json").write_text(json.dumps(ref,ensure_ascii=False),encoding="utf-8")
    model,_=load_gate3_model(ROOT/"demo/models/pathguard_gate3_model.joblib",contract)
    held=pd.read_csv(ROOT/'reports/final/tables/held_out_test_predictions.csv').set_index('sample_id')
    assert set(test.sample_id)==set(held.index)
    assert np.array_equal(test.true_label.to_numpy(), held.loc[test.sample_id,'true_label'].to_numpy())
    scores={"model":held.loc[test.sample_id,'failure_risk'].to_numpy(),
            "frozen_geometry_rule":compute_geometry_rule_risk(test,ref)[0].to_numpy(),
            "minimum_clearance":-test.static_boundary_clearance_min_m.to_numpy()}
    y=(test.true_label.to_numpy()==0).astype(int)
    output=[]
    for name,score in scores.items():
        for group,mask in [("overall",np.ones(len(test),dtype=bool))]+[
            ("map:"+str(m),test.map_id.eq(m).to_numpy()) for m in sorted(test.map_id.unique())]+[
            ("structure:"+str(s),test.vehicle_structure.eq(s).to_numpy()) for s in sorted(test.vehicle_structure.unique())]:
            part=test.loc[mask].copy()
            part["score"]=np.asarray(score)[mask]; part["failure"]=y[mask]
            part=part.sort_values(["score","sample_id"],ascending=[False,True],kind="stable")
            for k in [10,20,34]:
                selected=part.head(k); failures=int(part.failure.sum()); n=len(part)
                output.append({"method":name,"group":group,"budget":k,"n":n,"failures":failures,
                    "selected":len(selected),"captured":int(selected.failure.sum()),
                    "precision_at_k":float(selected.failure.mean()),
                    "capture_rate":float(selected.failure.sum()/failures) if failures else None,
                    "random_expected_captured":len(selected)*failures/n,
                    "random_expected_capture_rate":len(selected)/n if failures else None,
                    "roc_auc":float(roc_auc_score(part.failure,part.score)) if part.failure.nunique()==2 else None,
                    "pr_auc":float(average_precision_score(part.failure,part.score)) if failures else None})
        test[name+"_score"]=score
    args.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(output).to_csv(args.output/"engineering_comparison.csv",index=False)
    test.to_csv(args.output/"heldout_scores.csv",index=False)
    rng=np.random.default_rng(20260922)
    maps=sorted(test.map_id.unique()); delta=[]
    for _ in range(2000):
        pieces=[test[test.map_id.eq(m)] for m in rng.choice(maps,len(maps),replace=True)]
        b=pd.concat(pieces,ignore_index=True); yy=b.true_label.eq(0).astype(int)
        if yy.nunique()==2:
            delta.append(average_precision_score(yy,b.model_score)-average_precision_score(yy,b.frozen_geometry_rule_score))
    summary={"reference_version":ref["version"],"development_n":len(dev),"test_n":len(test),
       "test_maps":maps,"bootstrap_unit":"map","bootstrap_repeats":len(delta),
       "model_minus_rule_pr_auc_ci95":np.quantile(delta,[.025,.975]).tolist(),
       "scope":"All held-out candidates; pre-execution hard-filter pass flags and original planner costs unavailable.",
       "notes":"Post-hoc audit on an already inspected test set, not a new independent validation. Random is analytical expectation. Ties broken by sample_id. No hyperparameter selection."}
    (args.output/"engineering_comparison.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(pd.DataFrame(output).query("group == 'overall'").to_string(index=False))
    print(json.dumps(summary,ensure_ascii=False))
if __name__=="__main__": main()
