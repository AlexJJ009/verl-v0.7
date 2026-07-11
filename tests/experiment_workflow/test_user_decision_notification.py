from __future__ import annotations
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); assert s.loader; s.loader.exec_module(m); return m
def test_soft_failure_blocks_and_notifies_without_authorization(tmp_path:Path):
 gate=module('budget',ROOT/'scripts/check_code_task_preflight_budget.py'); notify=module('notify2',ROOT/'scripts/experiment_notification_policy.py')
 policy=json.loads((ROOT/'tests/experiment_workflow/fixtures/preflight_policy.json').read_text()); report={'contract':{'max_response_length':8192,'validation_datasets':['HumanEval+','MBPP+','LiveCodeBench']},'phases':[{'phase':'stage1','profile_hash':policy['hard']['required_profile_hash'],'metrics':{'complete_validation_metrics':True,'validation_elapsed_seconds':100,'timeout_rate':.2,'invalid_score_rate':0,'peak_rss_gib':100,'gpu_wait_fraction':.2}}]}
 result=gate.check(report,policy); assert result['decision']=='user_decision_required' and not result['ok']
 ledger=tmp_path/'ledger'; state={'run_id':'r','decision_required':True,'background':'soft threshold','evidence':json.dumps(result),'cost':'30 minutes GPU','recommendation':'review policy','local_paths':'/tmp/report'}
 sent=notify.process(state,ledger,None); assert sent['event']=='user_decision_required'; assert 'authorization' not in ledger.read_text().lower()
