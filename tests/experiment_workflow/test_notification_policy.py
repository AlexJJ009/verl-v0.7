from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def tool():
    p=ROOT/'scripts/experiment_notification_policy.py'; s=importlib.util.spec_from_file_location('notify',p); m=importlib.util.module_from_spec(s); assert s.loader; s.loader.exec_module(m); return m

def test_only_three_positive_events_and_non_events():
    m=tool()
    assert m.event_for({'training_step':1})=='run_started'
    assert m.event_for({'complete_validation_metrics':True})=='run_started'
    assert m.event_for({'terminal_failure':True,'cleanup_evidence':True})=='run_failed'
    assert m.event_for({'decision_required':True})=='user_decision_required'
    for state in ({'tmux':True},{'container':True},{'model_loading':True},{'training_step':0},{'healthy':True}): assert m.event_for(state) is None
    authority={'authority_type':'experiment_execution_core_event_v1','failure':None,'cleanup':None}
    assert m.event_for({**authority,'execution_status':'running'})=='run_started'
    assert m.event_for({**authority,'execution_status':'succeeded'}) is None
    assert m.event_for({**authority,'execution_status':'failed','failure':{'code':'child_exit'},'cleanup':{'resources_released':True}})=='run_failed'
    assert m.event_for({**authority,'execution_status':'failed'}) is None

def test_dedup_redaction_paths_and_delivery_failure(tmp_path: Path):
    m=tool(); ledger=tmp_path/'ledger.jsonl'; sender=tmp_path/'fail.sh'; sender.write_text('#!/bin/sh\nexit 7\n'); sender.chmod(0o755)
    state={'run_id':'r','training_step':1,'background':'TOKEN=secret','evidence':'e','cost':'c','recommendation':'r','local_paths':'/data-2/evidence'}
    first=m.process(state,ledger,[str(sender)]); second=m.process(state,ledger,[str(sender)])
    assert first=={'event':'run_started','sent':False,'reason':'delivery_failed'}
    assert second['reason']=='duplicate'
    record=json.loads(ledger.read_text()); assert 'secret' not in record['message']; assert '/data-2/evidence' in record['message']

def test_wxpusher_adapter_supports_exactly_reviewed_events():
    text=(ROOT/'scripts/wxpusher_event_sender.sh').read_text()
    assert '--title "$title" --body "$body"' in text
    assert all(event in text for event in ('run_started','run_failed','user_decision_required'))

def test_monitor_has_no_external_runtime_inference():
    text=(ROOT/'scripts/stage123_manifest_monitor.py').read_text()
    for forbidden in ('tmux','latest_checkpoint','checkpoint-root','queue-tmux','validation_deadlines'):
        assert forbidden not in text
