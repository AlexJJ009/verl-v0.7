from __future__ import annotations
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load():
 p=ROOT/'scripts/experiment_goal_completion_state.py'; s=importlib.util.spec_from_file_location('goalstate',p); m=importlib.util.module_from_spec(s); assert s.loader; s.loader.exec_module(m); return m
def test_only_deployable_plus_fresh_acceptance_completes():
 m=load(); assert m.state(None,None)=='PENDING OPERATIONAL CALIBRATION'; assert m.state({'decision':'blocked'},None)=='PENDING OPERATIONAL CALIBRATION'; assert m.state({'decision':'inconclusive'},None)=='PENDING OPERATIONAL CALIBRATION'; assert m.state({'decision':'deployable'},None)=='PENDING INDEPENDENT ACCEPTANCE'; assert m.state({'decision':'deployable'},{'verdict':'ACCEPTED','all_acceptance_criteria_pass':True})=='GOAL COMPLETE'
