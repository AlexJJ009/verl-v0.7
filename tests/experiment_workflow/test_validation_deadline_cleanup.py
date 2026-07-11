from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]

def tool():
    p=ROOT/'scripts/validation_deadline_controller.py'; s=importlib.util.spec_from_file_location('deadline',p); m=importlib.util.module_from_spec(s); assert s.loader; s.loader.exec_module(m); return m

def test_completion_before_deadline_and_active_state():
    m=tool(); base={'validation_ready_epoch_s':100,'deadline_seconds':1800}
    assert m.evaluate({**base,'complete_validation_metrics':True},200)['complete']
    assert not m.evaluate(base,200)['timed_out']
    assert m.evaluate(base,1901)['timed_out']
    assert m.evaluate({**base,'first_training_step':1,'complete_validation_metrics':False},1901)['timed_out']

def spawn(code: str):
    return subprocess.Popen(['python3','-c',code],start_new_session=True)

def test_graceful_and_forced_process_group_cleanup():
    m=tool(); graceful=spawn('import time; time.sleep(60)'); result=m.terminate_group(os.getpgid(graceful.pid),.2); graceful.wait(timeout=2); assert result['term_sent'] and not result['kill_sent']
    forced=spawn('import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'); time.sleep(.1); result=m.terminate_group(os.getpgid(forced.pid),.1); forced.wait(timeout=2); assert result['kill_sent']

def test_orphan_cleanup_and_idempotence(monkeypatch):
    m=tool(); child=spawn('import time; time.sleep(60)')
    monkeypatch.setattr(m,'command',lambda args:{'command':args,'returncode':0,'stdout':'','stderr':''})
    own={'descendant_pids':[child.pid],'gpu_pids':[],'tmux_sessions':[],'docker_containers':['fixture'],'container_init_pid':child.pid}
    first=m.cleanup(own,.01); child.wait(timeout=2); second=m.cleanup(own,.01)
    assert first['resources_released'] and second['resources_released']

def test_cleanup_failure_and_gpu_ownership_remain_blocked(monkeypatch):
    m=tool()
    def fake(args):
        if args[0]=='nvidia-smi': return {'command':args,'returncode':0,'stdout':'123\n','stderr':''}
        return {'command':args,'returncode':1,'stdout':'','stderr':'boom'}
    monkeypatch.setattr(m,'command',fake)
    report=m.cleanup({'gpu_pids':[123],'tmux_sessions':['x'],'docker_containers':[]},.01)
    assert not report['resources_released']; assert report['live_run_gpu_pids']==[123]; assert report['cleanup_failures']

def test_historical_76_minute_step0_trace_is_timed_out_and_release_blocked():
    m=tool(); fixture=ROOT/'tests/experiment_workflow/fixtures/validation_deadline/historical_76m_step0.json'; data=json.loads(fixture.read_text())
    state=m.evaluate(data,data['observed_epoch_s'])
    assert state['timed_out'] and data['release_gate_status']=='blocked' and data['sqlite_rows']==0 and data['wandb_sync_markers']==0 and data['residual_runtime_ownership']==[]
