from __future__ import annotations
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load():
 p=ROOT/'scripts/check_code_task_operational_calibration.py'; s=importlib.util.spec_from_file_location('calcheck',p); m=importlib.util.module_from_spec(s); assert s.loader; s.loader.exec_module(m); return m
def fixture():
 manifest={'manifest_sha256':'m','resource_profile':{'sha256':'p'}}
 phases=[]
 for name in ('stage1','stage2','stage3'):
  phases.append({'phase':name,'profile_hash':'p','optimized':True,'predicted':{'validation_elapsed_seconds':100,'peak_rss_gib':100,'gpu_wait_fraction':.5},'observed':{'complete_validation_metrics':True,'validation_elapsed_seconds':110,'peak_rss_gib':110,'gpu_wait_fraction':.55}})
 return manifest,{'evidence_class':'infrastructure_calibration','manifest_sha256':'m','decision':'deployable','phases':phases}
def test_deployable_and_blocked_boundaries():
 m=load(); manifest,report=fixture(); assert m.check(report,manifest)['ok']; report['phases'][0]['observed']['validation_elapsed_seconds']=1801; result=m.check(report,manifest); assert not result['ok'] and result['decision']=='blocked'
