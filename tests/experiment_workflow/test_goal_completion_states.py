from __future__ import annotations
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def load():
 p=ROOT/'scripts/experiment_goal_completion_state.py'; s=importlib.util.spec_from_file_location('goalstate',p); m=importlib.util.module_from_spec(s); assert s.loader; s.loader.exec_module(m); return m

def full_receipt():
 return {'receipt_type':'code_task_operational_calibration_deployability','decision':'deployable','profile':{'sha256':'p'},'hashes':{key:'x' for key in ('report_sha256','manifest_sha256','rendered_manifest_sha256','policy_sha256','history_index_sha256','prediction_contract_sha256','preflight_receipt_sha256')}}

def test_only_hash_bound_full_receipt_and_acceptance_completes():
 m=load(); full=full_receipt(); accepted={'verdict':'ACCEPTED','all_acceptance_criteria_pass':True,'calibration_receipt_sha256':'receipt-hash'}
 assert m.state(None,None)=='PENDING OPERATIONAL CALIBRATION'
 assert m.state({'decision':'stage12_calibrated'},None)=='PENDING OPERATIONAL CALIBRATION'
 assert m.state({'receipt_type':'code_task_operational_calibration_deployability','decision':'deployable'},accepted,'receipt-hash')=='PENDING OPERATIONAL CALIBRATION'
 assert m.state(full,accepted,None)=='PENDING INDEPENDENT ACCEPTANCE'
 assert m.state(full,{**accepted,'calibration_receipt_sha256':'wrong'},'receipt-hash')=='PENDING INDEPENDENT ACCEPTANCE'
 assert m.state(full,accepted,'receipt-hash')=='GOAL COMPLETE'
