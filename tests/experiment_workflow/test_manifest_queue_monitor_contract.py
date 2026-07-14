from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml"


def rendered():
    output = subprocess.check_output(["python3", str(ROOT / "scripts/experiment_manifest.py"), "render", str(MANIFEST), "--format", "json"], text=True)
    return json.loads(output)


def test_manifest_is_the_expected_shared_contract():
    report = rendered()
    queue_records = [(item["run_prefix"], item["final_step"], item["train_file"], item["tmux_name"]) for item in report["runs"]]
    monitor_records = [(item["run_prefix"], item["final_step"], item["train_file"], item["tmux_name"]) for item in report["runs"]]
    assert queue_records == monitor_records


def test_stage123_monitor_has_no_hardcoded_run_arrays_after_migration():
    monitor = (ROOT / "recipe/on_policy_wdl_sft/code_task/monitor_code_task_qwen3_1p7b_stage123_notify.sh").read_text()
    assert "RUN_PREFIXES=(" not in monitor
    assert "FINAL_STEPS=(" not in monitor
    assert "TRAIN_FILES=(" not in monitor
    assert "experiment_manifest.py" in monitor


def test_stage123_queue_reads_manifest():
    queue = (ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue_impl.sh").read_text()
    assert "experiment_manifest.py" in queue
    assert "STAGE123_FRACTIONS" not in queue
    assert "STAGE123_TRIGGERS" not in queue


def test_stage123_queue_enforces_validation_hard_wall():
    queue = (ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue_impl.sh").read_text()
    assert "validation batch [0-9]+/[0-9]+ start:" in queue
    assert "deadline_seconds':1800" in queue
    assert "validation_deadline_controller.py" in queue
    assert "return 124" in queue
    assert "DOCKER_CONTAINER_NAME" in queue
    assert "docker inspect" in queue


def test_stage123_monitor_uses_event_policy_not_legacy_tmux_started_notifications():
    monitor = (ROOT / "recipe/on_policy_wdl_sft/code_task/monitor_code_task_qwen3_1p7b_stage123_notify.sh").read_text()
    assert "stage123_manifest_monitor.py" in monitor
    assert "training_queue_monitor.sh" not in monitor
    assert "jq " not in monitor
    assert "jq " not in (ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue_impl.sh").read_text()
    python_monitor = (ROOT / "scripts/stage123_manifest_monitor.py").read_text()
    for forbidden in ("tmux", "checkpoint-root", "queue-tmux", "latest_checkpoint", "latest_checkpointed_iteration", "/data-1/checkpoints", "validation_deadlines"):
        assert forbidden not in python_monitor
    assert "--checkpoint-root" not in monitor and "--queue-tmux" not in monitor


def test_l40s_launcher_accepts_explicit_container_ownership_name():
    launcher = (ROOT / "scripts/l40s/run_train.sh").read_text()
    assert "DOCKER_CONTAINER_NAME" in launcher
    assert 'name_args=(--name "$DOCKER_CONTAINER_NAME")' in launcher
