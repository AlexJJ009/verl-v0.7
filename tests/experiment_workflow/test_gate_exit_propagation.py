from __future__ import annotations

import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]

def test_checked_pipeline_preserves_child_failure_through_tee(tmp_path: Path):
    result=subprocess.run(['bash',str(ROOT/'scripts/run_checked_pipeline.sh'),str(tmp_path/'log'),'bash','-c','echo before; exit 23'])
    assert result.returncode==23 and 'before' in (tmp_path/'log').read_text()

def test_socket_deny_layer_blocks_attempted_network():
    env={**os.environ,'PYTHONPATH':str(ROOT/'tests/experiment_workflow')}
    result=subprocess.run(['python3','-c','import socket_deny,socket; socket.socket()'],env=env,capture_output=True,text=True)
    assert result.returncode!=0 and 'network disabled' in result.stderr

def test_full_gate_wires_queue_and_monitor_to_same_fresh_scratch():
    text=(ROOT/'scripts/check_experiment_workflow_full.sh').read_text()
    assert 'rm -rf "$SCRATCH"' in text
    assert 'STAGE123_SCRATCH_ROOT="$SCRATCH"' in text
    assert '--manifest "$SCRATCH/stage123.normalized.json"' in text
