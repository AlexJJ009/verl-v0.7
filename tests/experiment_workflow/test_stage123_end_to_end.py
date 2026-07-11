from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]

def digest(path: Path):
    return None if not path.exists() else hashlib.sha256(path.read_bytes()).hexdigest()

def test_dry_run_is_scratch_only_and_manifest_consistent(tmp_path: Path):
    registry=Path('/data-1/experiment_registry/experiment_registry.sqlite'); release=Path('/data-1/experiment_registry/training_release_gate.jsonl')
    before=(digest(registry),digest(release))
    scratch=tmp_path/'scratch'; status=tmp_path/'status.tsv'
    env={**os.environ,'DRY_RUN':'1','STAGE123_MANIFEST_PYTHON':'/opt/venv/bin/python','STAGE123_SCRATCH_ROOT':str(scratch),'QUEUE_STATUS_FILE':str(status)}
    result=subprocess.run(['bash',str(ROOT/'recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue.sh')],env=env,capture_output=True,text=True)
    assert result.returncode==0, result.stderr
    normalized=json.loads((scratch/'stage123.normalized.json').read_text())
    provenance=list(scratch.glob('**/*.provenance.json'))
    assert len(provenance)==len(normalized['runs'])
    assert all(json.loads(path.read_text())['release_eligible'] is False for path in provenance)
    assert before==(digest(registry),digest(release))
    assert 'tmux new-session' not in result.stdout and 'docker run' not in result.stdout
