from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml"


def rendered():
    output = subprocess.check_output(
        ["python3", str(ROOT / "scripts/experiment_manifest.py"), "render", str(MANIFEST), "--format", "json"],
        text=True,
    )
    return json.loads(output)


def test_manifest_is_the_expected_shared_contract():
    report = rendered()
    queue_records = [
        (item["run_prefix"], item["final_step"], item["train_file"], item["tmux_name"]) for item in report["runs"]
    ]
    monitor_records = [
        (item["run_prefix"], item["final_step"], item["train_file"], item["tmux_name"]) for item in report["runs"]
    ]
    assert queue_records == monitor_records


def test_stage123_monitor_has_no_hardcoded_run_arrays_after_migration():
    monitor = (ROOT / "recipe/on_policy_wdl_sft/code_task/monitor_code_task_qwen3_1p7b_stage123_notify.sh").read_text()
    assert "RUN_PREFIXES=(" not in monitor
    assert "FINAL_STEPS=(" not in monitor
    assert "TRAIN_FILES=(" not in monitor
    assert "experiment_manifest.py" in monitor


def test_stage123_queue_reads_manifest():
    queue = (ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue_impl.sh").read_text()
    assert "EXPERIMENT_BATCH_MANIFEST" in queue
    assert "experiment_execution_core.py" in queue
    assert "STAGE123_FRACTIONS" not in queue
    assert "STAGE123_TRIGGERS" not in queue


def test_stage123_queue_has_no_validation_or_lifecycle_authority():
    queue = (ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue_impl.sh").read_text()
    for forbidden in (
        "validation batch",
        "deadline_seconds",
        "validation_deadline_controller.py",
        "docker inspect",
        "status.tsv",
        "launch_and_wait",
        "latest_checkpoint",
    ):
        assert forbidden not in queue
    assert "batch-run" in queue


def test_stage123_phase_gate_does_not_treat_batch_environment_as_authorization():
    bundle = ROOT / "docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json"
    environment = {**os.environ, "STAGE123_ADMISSION_BUNDLE": str(bundle), "STAGE123_BATCH_EXECUTION": "1"}
    gate = ROOT / "recipe/on_policy_wdl_sft/code_task/stage123_manifest_gate.sh"
    command = ["bash", "-lc", f"source {gate}; stage123_require_formal_admission frac25-stage2"]
    assert subprocess.run(command, env=environment, text=True, capture_output=True).returncode != 0


def test_direct_stage123_phase_still_requires_full_admission():
    bundle = ROOT / "docs/joint_training/goals/stage123-execution-readiness/admission_bundle.json"
    gate = ROOT / "recipe/on_policy_wdl_sft/code_task/stage123_manifest_gate.sh"
    result = subprocess.run(
        ["bash", "-lc", f"source {gate}; stage123_require_formal_admission frac25-stage2"],
        env={**os.environ, "STAGE123_ADMISSION_BUNDLE": str(bundle)},
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0


def test_batch_core_has_no_socket_or_nonce_authorization_claim():
    core = (ROOT / "scripts/experiment_execution_core.py").read_text()
    assert "STAGE123_BATCH_EXECUTION" in core
    assert "AF_UNIX" not in core
    assert "token_urlsafe" not in core


def test_treatment_only_phase_gate_validates_its_own_admission():
    gate = (ROOT / "recipe/on_policy_wdl_sft/code_task/stage123_manifest_gate.sh").read_text()
    core = (ROOT / "scripts/experiment_execution_core.py").read_text()
    assert "STAGE123_TREATMENT_REUSE_ADMISSION" in gate
    assert "validate-treatment" in gate
    assert "STAGE123_TREATMENT_REUSE_ADMISSION" in core


def test_stage123_phase_wrappers_own_same_container_ray_lifecycle():
    for name in ("run_s2_code_qwen3_1p7b_stage123_common.sh", "run_s3_code_qwen3_1p7b_stage123_common.sh"):
        wrapper = (ROOT / "recipe/on_policy_wdl_sft/code_task" / name).read_text()
        assert 'RAY_TMPDIR="${STAGE123_RAY_TMPDIR:-/tmp/stage123-ray-${STAGE123_RUN_ID}}"' in wrapper
        assert "ray start --head --port=22000 --min-worker-port=21000 --max-worker-port=21999" in wrapper
        assert 'export RAY_ADDRESS="127.0.0.1:22000"' in wrapper
        assert "trap cleanup_stage123_ray EXIT" in wrapper
        assert "ray stop --force" in wrapper
        assert wrapper.index("trap cleanup_stage123_ray EXIT") < wrapper.index("ray start --head")


def test_stage123_monitor_uses_event_policy_not_legacy_tmux_started_notifications():
    monitor = (ROOT / "recipe/on_policy_wdl_sft/code_task/monitor_code_task_qwen3_1p7b_stage123_notify.sh").read_text()
    assert "stage123_manifest_monitor.py" in monitor
    assert "training_queue_monitor.sh" not in monitor
    assert "jq " not in monitor
    assert (
        "jq "
        not in (ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_queue_impl.sh").read_text()
    )
    python_monitor = (ROOT / "scripts/stage123_manifest_monitor.py").read_text()
    for forbidden in (
        "tmux",
        "checkpoint-root",
        "queue-tmux",
        "latest_checkpoint",
        "latest_checkpointed_iteration",
        "/data-1/checkpoints",
        "validation_deadlines",
    ):
        assert forbidden not in python_monitor
    assert "--checkpoint-root" not in monitor and "--queue-tmux" not in monitor


def test_l40s_launcher_accepts_explicit_container_ownership_name():
    launcher = (ROOT / "scripts/l40s/run_train.sh").read_text()
    assert "DOCKER_CONTAINER_NAME" in launcher
    assert 'name_args=(--name "$DOCKER_CONTAINER_NAME")' in launcher
