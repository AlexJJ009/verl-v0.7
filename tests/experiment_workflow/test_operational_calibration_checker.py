from __future__ import annotations
import importlib.util
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]
def load():
 p=ROOT/'scripts/check_code_task_operational_calibration.py'; s=importlib.util.spec_from_file_location('calcheck',p); m=importlib.util.module_from_spec(s); assert s.loader; s.loader.exec_module(m); return m
def fixture(tmp_path):
 manifest={'manifest_sha256':'m','resource_profile':{'sha256':'p'}}
 phases=[]
 for name in ('stage1','stage2','stage3'):
  reps=[]
  for i in range(4):
   artifacts={}
   for kind in ('status','resources','metrics'):
    path=tmp_path/f'{name}-{i}-{kind}.json'; path.write_text(json.dumps({'kind':kind}))
    artifacts[kind]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
   reps.append({'warmup':i==0,'status':{'returncode':0,'timed_out':False},'metrics':{'complete_validation_metrics':True,'validation_elapsed_seconds':110},'resources':{'peak_rss_gib':110,'gpu_wait_fraction':.55},'artifacts':artifacts})
  phases.append({'phase':name,'profile_hash':'p','optimized':True,'predicted':{'validation_elapsed_seconds':100,'peak_rss_gib':100,'gpu_wait_fraction':.5},'observed':{'complete_validation_metrics':True,'validation_elapsed_seconds':110,'peak_rss_gib':110,'gpu_wait_fraction':.55},'repetitions':reps})
 return manifest,{'evidence_class':'infrastructure_calibration','manifest_sha256':'m','decision':'deployable','phases':phases}
def test_deployable_and_blocked_boundaries(tmp_path):
 m=load(); manifest,report=fixture(tmp_path); assert m.check(report,manifest)['ok']; report['phases'][0]['observed']['validation_elapsed_seconds']=1801; result=m.check(report,manifest); assert not result['ok'] and result['decision']=='blocked'

def test_repetition_and_artifact_tampering_are_blocked(tmp_path):
 m=load(); manifest,report=fixture(tmp_path); report['phases'][0]['repetitions'][0]['warmup']=False
 assert not m.check(report,manifest)['ok']
 manifest,report=fixture(tmp_path); Path(report['phases'][0]['repetitions'][0]['artifacts']['metrics']['path']).write_text('tampered')
 assert not m.check(report,manifest)['ok']
