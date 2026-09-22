"""Regression coverage for fixed rules, decisions and raw NPZ upload."""
import io,json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from demo_logic import evaluate_candidates
from inference import load_contract
from rule_baseline import compute_geometry_rule_risk
from route_input import parse_route, compute_rigid_body_sweep
BASE=Path(__file__).parent
class ProductRegressionTests(unittest.TestCase):
    def setUp(self):
        self.sample=pd.read_csv(BASE/'data/sample_input.csv')
        self.contract=load_contract(BASE/'data/feature_contract_v2.json')
    def test_rule_batch_invariant(self):
        batch,parts=compute_geometry_rule_risk(self.sample)
        for i in self.sample.index:
            single,p=compute_geometry_rule_risk(self.sample.loc[[i]])
            self.assertAlmostEqual(single.iloc[0],batch.loc[i])
            np.testing.assert_allclose(p.iloc[0],parts.loc[i])
        expanded=pd.concat([self.sample,self.sample]*3,ignore_index=True)
        np.testing.assert_allclose(compute_geometry_rule_risk(expanded)[0][:12],batch)
    def test_conflict_priority_survives_and_unknown_refuses(self):
        with patch('demo_logic.predict_failure_risk',return_value=np.ones(12)),patch('demo_logic.compute_geometry_rule_risk',return_value=(pd.Series(np.zeros(12)),pd.DataFrame({'曲率与变化':np.zeros(12)}))):
            result,_,state=evaluate_candidates(self.sample,self.contract,BASE/'models/pathguard_gate3_model.joblib')
        self.assertTrue(state['model_available'])
        known=result.vehicle_structure.ne('unknown')
        self.assertTrue(result.loc[known,'validation_priority'].eq('P0 人工复核').all())
        self.assertTrue(result.loc[~known,'risk_level'].eq('证据不足').all())
    def test_real_upload_and_location(self):
        payload=(BASE/'examples/real_route.npz').read_bytes()
        config=json.loads((BASE/'examples/vehicle_config.json').read_text())
        frame,key,geo=parse_route(payload,'test.npz',config)
        result,_,state=evaluate_candidates(frame,self.contract,BASE/'models/pathguard_gate3_model.joblib')
        self.assertTrue(state['model_available'])
        self.assertEqual(len(geo['events']),4)
        self.assertEqual(geo['source_sha256'],frame.iloc[0].route_file_sha256)
        _,other,_=parse_route(payload,'same.npz',{**config,'length_m':15})
        self.assertNotEqual(key,other)
        self.assertEqual(geo['left_boundary'],[])
    def test_missing_fields_rejected(self):
        buffer=io.BytesIO();np.savez(buffer,x_m=np.arange(4))
        with self.assertRaisesRegex(ValueError,'missing required'):
            parse_route(buffer.getvalue(),'missing.npz')
    def test_object_arrays_rejected(self):
        buffer=io.BytesIO();np.savez(buffer,s_m=np.array([{}],dtype=object))
        with self.assertRaises(ValueError):
            parse_route(buffer.getvalue(),'object.npz')
    def test_dimensions_do_not_fake_model_or_rule_risk(self):
        payload=(BASE/'examples/real_route.npz').read_bytes()
        a,_,_=parse_route(payload,'a.npz',{'length_m':8,'width_m':2,'reference_from_rear_m':3})
        b,_,_=parse_route(payload,'b.npz',{'length_m':18,'width_m':5,'reference_from_rear_m':7})
        np.testing.assert_allclose(a[self.contract['training_feature_columns']],b[self.contract['training_feature_columns']])
        np.testing.assert_allclose(compute_geometry_rule_risk(a)[0],compute_geometry_rule_risk(b)[0])
    def test_real_scale_rigid_sweep(self):
        geometry={'centerline':[[0,0],[10,0]],'yaw_rad':[0,0]}
        x,y=compute_rigid_body_sweep(geometry,12,4,5)
        np.testing.assert_allclose(x,[[7,7,-5,-5],[17,17,5,5]])
        np.testing.assert_allclose(y,[[2,-2,-2,2],[2,-2,-2,2]])
        wider_y=compute_rigid_body_sweep(geometry,12,6,5)[1]
        self.assertGreater(np.ptp(wider_y[0]),np.ptp(y[0]))
        longer_x=compute_rigid_body_sweep(geometry,16,4,5)[0]
        self.assertGreater(longer_x[0].max(),x[0].max())
        with self.assertRaises(ValueError):
            compute_rigid_body_sweep(geometry,4,2,5)
if __name__=='__main__':unittest.main()
