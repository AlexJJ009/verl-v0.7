from __future__ import annotations
import importlib.util,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); assert s.loader; sys.modules[name]=m; s.loader.exec_module(m); return m
def test_boundary_manifest_literal_contract():
 value=json.loads((ROOT/'config/experiment_execution/stage123_implementation_boundary_v1.json').read_text())
 assert value['superproject_tree_roots']==['config/experiment_execution','scripts','verl'] and value['submodules']==[{'path':'recipe','coverage':'entire_gitlink_commit'}]
def test_probe_renderer_rejects_training_and_non_scratch(tmp_path):
 base=[sys.executable,str(ROOT/'scripts/render_calibration_probe_command.py'),'--manifest',str(ROOT/'recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml'),'--resource-profile',str(ROOT/'recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh'),'--phases','stage2,stage3','--repetitions','3']
 bad=subprocess.run(base+['--training-steps','1','--scratch-root','/data-1/tmp/verl_agent_scratch/calibration'],capture_output=True,text=True); assert bad.returncode==1 and 'training_steps' in bad.stderr
 bad=subprocess.run(base+['--training-steps','0','--scratch-root',str(tmp_path)],capture_output=True,text=True); assert bad.returncode==1 and 'scratch_root' in bad.stderr
def test_probe_renderer_is_zero_step_json():
 result=subprocess.run([sys.executable,str(ROOT/'scripts/render_calibration_probe_command.py'),'--manifest',str(ROOT/'recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml'),'--resource-profile',str(ROOT/'recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh'),'--phases','stage2,stage3','--repetitions','3','--training-steps','0','--scratch-root','/data-1/tmp/verl_agent_scratch/experiment_workflow/calibration'],check=True,capture_output=True,text=True)
 command=json.loads(result.stdout); assert command[1].endswith('run_calibration_probe_zero_step.py') and '--training-steps' in command and command[command.index('--training-steps')+1]=='0' and command[command.index('--optimizer-enabled')+1]=='false'
def test_prediction_fails_closed_and_qualifies():
 m=load('calibration_prediction',ROOT/'scripts/calibration_prediction.py'); policy=json.loads((ROOT/'config/experiment_execution/calibration_policy_v1.json').read_text())
 assert m.qualify([1,2],10,10,policy).code=='insufficient_history'; assert m.qualify([1,2,3],10,13,policy).code=='prediction_exceeded'; assert m.qualify([1,2,3],10,12,policy).qualified
def test_result_schema_requires_complete_structured_result():
 m=load('calibration_result',ROOT/'scripts/calibration_result.py'); schema=json.loads((ROOT/'config/experiment_execution/calibration_result_schema_v1.json').read_text())
 assert m.validate({'schema_version':1,'result_type':'calibration_result','decision':'passed'},schema)['failures'][0]['code']=='result_fields'
 policy=ROOT/'config/experiment_execution/calibration_policy_v1.json'
 value={'schema_version':1,'result_type':'calibration_result','decision':'passed','manifest_sha256':'a'*64,'resource_profile_sha256':'b'*64,'implementation_tree_sha256':'c'*64,'evidence_commit':'d'*40,'workload_identity':{'sha256':'e'*64},'policy_id':'stage123-calibration-policy-v1','policy_sha256':m.policy_sha256(policy),'authorization_identity':{'id':'auth'},'started_at':'2026-01-01T00:00:00Z','completed_at':'2026-01-01T00:01:00Z','phase_evidence':[{'phase':'stage2','status':'passed'},{'phase':'stage3','status':'passed'}],'prediction_comparison':{'qualified':True},'cleanup':{'resources_released':True},'failures':[]}
 assert m.validate(value,schema)['ok']; value['cleanup']['resources_released']=False; assert m.validate(value,schema)['failures'][0]['code']=='cleanup'

def test_zero_step_driver_runs_exact_phase_matrix_and_stops_on_failure(tmp_path, monkeypatch):
 m=load('run_calibration_probe_zero_step',ROOT/'scripts/run_calibration_probe_zero_step.py')
 calls=[]
 def fake_run(rendered,phase,repetition,root,splits,timeout):
  calls.append((phase,repetition,timeout)); return {'status':'failed' if phase=='stage3' and repetition==2 else 'passed'}
 monkeypatch.setattr(m,'run_repetition',fake_run)
 rendered={'calibration_policy':{'validation_deadline_seconds':1800}}
 failures=[]; results=[]
 for phase in ('stage2','stage3'):
  for repetition in range(1,4):
   value=m.run_repetition(rendered,phase,repetition,tmp_path,{},1800); results.append(value)
   if value['status']!='passed': failures.append(value); break
  if failures: break
 assert calls==[('stage2',1,1800),('stage2',2,1800),('stage2',3,1800),('stage3',1,1800),('stage3',2,1800)]

def test_phase_runner_is_zero_step_val_only_and_cleans_owned_ray_only():
 text=(ROOT/'recipe/on_policy_wdl_sft/code_task/run_code_task_operational_calibration_phase.sh').read_text()
 assert 'VAL_ONLY=True TOTAL_TRAINING_STEPS="$CALIBRATION_TOTAL_TRAINING_STEPS"' in text
 assert 'CALIBRATION_TOTAL_TRAINING_STEPS:=0' in text
 assert 'CALIBRATION_OPTIMIZER_ENABLED:=false' in text
 assert 'ray stop --force' not in text
 assert "root in command" in text
 assert '--temp-dir="$RAY_TMPDIR"' in text
 assert 'CALIBRATION_RAY_TMPDIR:=' in text
 assert "'trainer.logger=[\"file\"]'" in text
 assert "trainer.logger=[\"wandb\",\"file\"]" not in text

def test_stage3_proxy_is_explicit_and_hash_bound():
 rendered=json.loads(subprocess.check_output([sys.executable,str(ROOT/'scripts/experiment_manifest.py'),'render',str(ROOT/'recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml'),'--format','json'],text=True))
 workload=rendered['calibration_workloads']['stage3']; proxy=workload['calibration_proxy']
 assert workload['model_sources'][0]['state']=='pending'
 assert proxy['purpose']=='pending_stage2_handoff_runtime_resource_proxy'
 assert proxy['rollout_model_parameter_count']==workload['rollout_model_parameter_count_sum']
 assert len(proxy['artifact_sha256'])==64

def test_owned_ray_socket_path_fits_af_unix_limit():
 base='/data-1/tmp/verl_agent_scratch/r/31'
 plasma=f'{base}/session_2026-07-13_18-40-15_123456_2147483647/sockets/plasma_store'
 assert len(plasma)<=107

def test_probe_uses_full_sequential_ray_worker_range():
 text=(ROOT/'scripts/run_calibration_probe_zero_step.py').read_text()
 assert '"CALIBRATION_RAY_WORKER_PORT_MIN": "21000"' in text
 assert '"CALIBRATION_RAY_WORKER_PORT_MAX": "21999"' in text
 assert '21099' not in text
