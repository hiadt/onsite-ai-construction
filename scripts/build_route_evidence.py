"""Build demo location evidence and a real input example from matched raw routes."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"demo"))
from route_input import parse_route
def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);a=p.parse_args()
    samples=pd.read_csv(ROOT/"demo/data/sample_input.csv")
    manifest=pd.read_csv(a.manifest).set_index("sample_id")
    routes={}; columns=json.loads((ROOT/"demo/data/feature_contract_v2.json").read_text(encoding="utf-8"))["training_feature_columns"]
    example=False
    for _,sample in samples.iterrows():
        row=manifest.loc[sample.sample_id]
        if isinstance(row,pd.DataFrame):row=row.iloc[0]
        raw=Path(row.route_path_local)
        defaults={"five_axis":(12.0,3.2,5.0,35000),"six_axis":(15.0,3.4,6.0,50000),"unknown":(12.0,3.2,5.0,35000)}[sample.vehicle_structure]
        config={"vehicle_structure":sample.vehicle_structure,"map_id":sample.map_id,
                "length_m":defaults[0],"width_m":defaults[1],
                "reference_from_rear_m":defaults[2],"mass_kg":defaults[3]}
        features,key,geo=parse_route(raw.read_bytes(),raw.name,config)
        np.testing.assert_allclose(features[columns].to_numpy(float),sample[columns].to_numpy(float)[None,:],rtol=1e-5,atol=1e-7)
        assert features.iloc[0].route_file_sha256==row.route_file_sha256
        samples.loc[samples.sample_id.eq(sample.sample_id),'route_file_sha256']=row.route_file_sha256
        if geo:routes[sample.sample_id]=geo
        if not example and sample.vehicle_structure!="unknown":
            folder=ROOT/"demo/examples";folder.mkdir(exist_ok=True)
            shutil.copyfile(raw,folder/"real_route.npz")
            (folder/"vehicle_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
            example=True
    samples.to_csv(ROOT/'demo/data/sample_input.csv',index=False)
    (ROOT/"demo/data/route_geometry.json").write_text(json.dumps({"routes":routes},ensure_ascii=False),encoding="utf-8")
    print("Verified extractor parity and hashes for",len(samples),"routes; geometry",len(routes))
if __name__=="__main__":main()
