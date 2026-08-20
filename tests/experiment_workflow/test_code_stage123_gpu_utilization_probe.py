from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "scripts/run_code_stage123_gpu_utilization_probe.py"
PHASE = ROOT / "scripts/code_stage123_probe_phase.py"
WRAPPER = ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_stage123_gpu_utilization_probe.sh"


def module():
    spec = importlib.util.spec_from_file_location("code_stage123_gpu_probe", PROBE)
    result = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(result)
    return result


def admission_module():
    path = ROOT / "scripts/code_stage123_admission.py"
    spec = importlib.util.spec_from_file_location("code_stage123_admission", path)
    result = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(result)
    return result


def test_dry_run_covers_all_phases_candidates_and_worst_case_stage2_arms(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE),
            "--manifest",
            str(ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/code_qwen3_1p7b_stage123_cotmask_v3.yaml"),
            "--scratch-root",
            "/data-1/tmp/verl_agent_scratch/code-stage123-probe-test",
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    plan = json.loads(result.stdout)
    assert plan["candidates"] == [0.35, 0.4, 0.45, 0.5, 0.55]
    assert plan["zero_step_phases"] == ["stage1"]
    assert plan["throughput_arms"] == ["stage1-beta01", "stage2-m2kl-beta01"]
    assert plan["validation_scope"] == "full_code3"
    assert plan["validation_n"] == 3
    assert plan["max_response_length"] == 8192
    assert plan["entropy"] == {"actor": False, "joint": False}
    assert plan["tmux_required"] is True
    assert "highest requested candidate" in plan["selection_policy"]
    assert "admitted" in plan["real_run_manifest_requirement"]


def test_probe_default_headroom_floor_is_explicit_and_report_owned():
    text = PROBE.read_text()
    wrapper = WRAPPER.read_text()
    admission = (ROOT / "scripts/code_stage123_admission.py").read_text()
    assert "default=512" in text
    assert '"minimum_required_gpu_headroom_mib": args.minimum_headroom_mib' in text
    assert "MINIMUM_HEADROOM_MIB=${MINIMUM_HEADROOM_MIB:-512}" in wrapper
    assert 'minimum_required_headroom = int(report.get("minimum_required_gpu_headroom_mib", 0))' in admission
    assert "minimum_required_headroom != 512" in admission
    assert 'raise SystemExit(f"admission file missing: {args.admission}")' in admission


def test_admission_hash_cli_supports_negative_hash_verification(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"ok": true}\n')
    expected = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/code_stage123_admission.py"), "hash", "--path", str(source)],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


def test_probe_profile_hash_binds_candidate_utilization():
    probe = module()
    assert probe.profile_hash(0.4) != probe.profile_hash(0.55)


def test_reused_validation_must_come_from_equal_or_higher_utilization(tmp_path: Path):
    probe = module()
    source_root = tmp_path / "probe-source"
    result_path = source_root / "mem55" / "zero-step" / "stage1" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps({"status": "passed", "validation_complete": True}))
    source_manifest = {"resources": {"rollout_gpu_memory_utilization": 0.55}, "task": "code"}
    current_manifest = {"resources": {"rollout_gpu_memory_utilization": 0.50}, "task": "code"}
    (source_root / "input-manifest.yaml").write_text(yaml.safe_dump(source_manifest))
    manifest_path = tmp_path / "current-manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(current_manifest))
    reused = probe.reused_validation_result(result_path, 0.50, manifest_path)
    assert reused["reuse_validation_source_utilization"] == 0.55
    assert reused["reused_for_utilization"] == 0.50
    try:
        probe.reused_validation_result(result_path, 0.55 + 0.01, manifest_path)
    except SystemExit as exc:
        assert "equal or higher" in str(exc)
    else:
        raise AssertionError("lower-utilization validation was allowed to qualify a higher candidate")


def test_reused_validation_rejects_manifest_drift_beyond_utilization(tmp_path: Path):
    probe = module()
    source_root = tmp_path / "probe-source"
    result_path = source_root / "mem40" / "zero-step" / "stage1" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps({"status": "passed", "validation_complete": True}))
    (source_root / "input-manifest.yaml").write_text(
        yaml.safe_dump({"resources": {"rollout_gpu_memory_utilization": 0.40}, "seed": 1})
    )
    manifest_path = tmp_path / "current-manifest.yaml"
    manifest_path.write_text(yaml.safe_dump({"resources": {"rollout_gpu_memory_utilization": 0.35}, "seed": 2}))
    try:
        probe.reused_validation_result(result_path, 0.35, manifest_path)
    except SystemExit as exc:
        assert "differs beyond utilization" in str(exc)
    else:
        raise AssertionError("validation reuse accepted unrelated manifest drift")


def test_reused_throughput_is_bound_to_current_manifest_and_real_update(tmp_path: Path):
    probe = module()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("task: code\n")
    source = tmp_path / "source"
    source.mkdir()
    (source / "input-manifest.yaml").write_bytes(manifest.read_bytes())
    result_path = source / "mem50" / "one-step" / "stage1-beta01" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "run_id": "b01-stage1",
                "mode": "train",
                "training_contract_complete": True,
                "runtime_contract_complete": True,
                "optimizer_steps": 1,
                "observed_training_metrics": {
                    "wdl_sft/n_correct": 1,
                    "actor/wdl_sft_loss_positive": 123.0,
                    "actor/grad_norm": 4.0,
                },
                "formal_checkpoint_files": [],
            }
        )
    )
    reused = probe.reused_throughput_result(source, 0.50, "stage1-beta01", "b01-stage1", manifest)
    assert reused["reused_from"] == str(result_path)
    result_path.write_text(json.dumps({**reused, "training_contract_complete": False}))
    try:
        probe.reused_throughput_result(source, 0.50, "stage1-beta01", "b01-stage1", manifest)
    except SystemExit as exc:
        assert "not admissible" in str(exc)
    else:
        raise AssertionError("throughput result without a real update was accepted")


def test_non_dry_run_requires_tmux_before_gpu_probe(monkeypatch):
    probe = module()
    monkeypatch.delenv("TMUX", raising=False)
    try:
        probe.assert_tmux_and_idle_gpus(5, 1024)
    except SystemExit as exc:
        assert "inside tmux" in str(exc)
    else:
        raise AssertionError("non-tmux probe did not fail closed")


def test_probe_rejects_legacy_answer_only_model1(tmp_path):
    probe = module()
    selection = tmp_path / "selection.json"
    receipt = tmp_path / "receipt.json"
    selection.write_text(
        json.dumps({"selected_step": 20, "identity": {"model_path": "/models/format_cold_start_fraction/legacy"}})
    )
    receipt.write_text(
        json.dumps({"schema_version": 2, "overlap_policy": {"cold_start_vs_stage1_stage2_stage3": "pairwise_disjoint"}})
    )
    manifest = {
        "task": "code",
        "model1_selection_policy": {"selected_step": 20},
        "paths": {"model1_selection": str(selection), "dataset_receipt": str(receipt)},
    }
    try:
        probe.assert_cot_v3_probe_inputs(manifest)
    except SystemExit as exc:
        assert "legacy answer-only" in str(exc)
    else:
        raise AssertionError("legacy Model1 did not fail closed")


def test_probe_wrapper_is_dry_run_safe_and_writes_only_scratch():
    text = WRAPPER.read_text()
    assert "DRY_RUN" in text
    assert "/data-1/tmp/verl_agent_scratch/code_stage123_gpu_utilization_probe" in text
    assert "0.35,0.40,0.45,0.50,0.55" in text
    assert "run_code_stage123_gpu_utilization_probe.py" in text
    assert "code_qwen3_1p7b_stage123_cotmask_v3.yaml" in text


def test_probe_validation_contract_requires_real_n3_metrics_for_every_dataset_and_view():
    spec = importlib.util.spec_from_file_location("code_stage123_probe_phase", PHASE)
    phase = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(phase)
    single = {
        f"val-core/{dataset}/acc/{metric}": 0.5
        for dataset in ("HumanEval+", "MBPP+", "LiveCodeBench")
        for metric in ("mean@3", "pass@3")
    }
    assert phase.validation_contract(single, joint=False) == (True, [])
    joint = {
        f"val-core/{view}/{dataset}/acc/{metric}": 0.5
        for view in ("model1", "model2")
        for dataset in ("HumanEval+", "MBPP+", "LiveCodeBench")
        for metric in ("mean@3", "pass@3")
    }
    assert phase.validation_contract(joint, joint=True) == (True, [])
    joint.pop("val-core/model2/LiveCodeBench/acc/pass@3")
    passed, missing = phase.validation_contract(joint, joint=True)
    assert passed is False
    assert missing == ["val-core/model2/LiveCodeBench/acc/pass@3"]


def test_probe_runtime_contract_rejects_offload_mismatch():
    spec = importlib.util.spec_from_file_location("code_stage123_probe_phase_runtime", PHASE)
    phase = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(phase)
    passed, missing = phase.runtime_contract(
        "actor_rollout_ref.actor.fsdp_config.param_offload=False "
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False "
        "actor_rollout_ref.ref.fsdp_config.param_offload=False",
        joint=False,
    )
    assert passed is False
    assert missing == ["actor_param_offload", "actor_optimizer_offload", "reference_param_offload"]
    passed, missing = phase.runtime_contract(
        "actor_rollout_ref.actor.fsdp_config.param_offload=True "
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True "
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        joint=True,
    )
    assert passed is True
    assert missing == []


def test_probe_train_mode_disables_last_step_validation():
    text = PHASE.read_text()
    assert '"TEST_FREQ": "-1"' in text
    assert '"trainer.test_freq=-1"' in text


def test_probe_training_arm_uses_full_fixed_order_shard_and_dumps_rollouts():
    text = PHASE.read_text()
    assert '"TRAIN_MAX_SAMPLES": "-1"' in text
    assert '"ROLLOUT_DATA_DIR": str(args.output_root / "rollout_data")' in text
    assert "trainer.rollout_data_dir={env['ROLLOUT_DATA_DIR']}" in text


def test_probe_training_contract_requires_a_real_nonzero_update():
    spec = importlib.util.spec_from_file_location("code_stage123_probe_phase_training", PHASE)
    phase = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(phase)
    all_incorrect_beta_zero = {
        "training/global_step": 1,
        "actor/lr": 1e-6,
        "wdl_sft/n_correct": 0,
        "actor/wdl_sft_loss_positive": 0.0,
        "actor/grad_norm": 0.0,
        "timing_s/update_actor": 94.0,
    }
    passed, missing = phase.training_contract(all_incorrect_beta_zero)
    assert passed is False
    assert missing == ["positive_samples", "positive_loss", "nonzero_gradient"]
    all_incorrect_beta_point_one = dict(all_incorrect_beta_zero, **{"actor/grad_norm": 12.5})
    passed, missing = phase.training_contract(all_incorrect_beta_point_one)
    assert passed is False
    assert missing == ["positive_samples", "positive_loss"]
    positive_update = dict(
        all_incorrect_beta_point_one,
        **{"wdl_sft/n_correct": 1, "actor/wdl_sft_loss_positive": 123.0},
    )
    assert phase.training_contract(positive_update) == (True, [])


def test_admission_rejects_beta01_gradient_without_positive_samples():
    admission = admission_module()
    candidate = {
        "zero_step": [{"run_id": "b0-stage1"}],
        "one_step": [
            {
                "run_id": "b01-stage1",
                "training_contract_complete": True,
                "observed_training_metrics": {
                    "wdl_sft/n_correct": 0,
                    "actor/wdl_sft_loss_positive": 0.0,
                    "actor/grad_norm": 0.0,
                },
            },
            {
                "run_id": "b01-stage2-m2kl",
                "training_contract_complete": True,
                "observed_training_metrics": {
                    "wdl_sft/n_correct": 0,
                    "actor/wdl_sft_loss_positive": 0.0,
                    "actor/grad_norm": 12.5,
                },
            },
        ],
    }
    try:
        admission.validate_probe_training_arms(candidate)
    except SystemExit as exc:
        assert "no positive samples" in str(exc)
    else:
        raise AssertionError("zero-positive beta=0.1 gradient was admitted")


def test_admission_accepts_positive_reward_training_arms():
    admission = admission_module()
    candidate = {
        "zero_step": [{"run_id": "b0-stage1"}],
        "one_step": [
            {
                "run_id": run_id,
                "training_contract_complete": True,
                "observed_training_metrics": {
                    "wdl_sft/n_correct": 1,
                    "actor/wdl_sft_loss_positive": 123.0,
                    "actor/grad_norm": 4.0,
                },
            }
            for run_id in ("b01-stage1", "b01-stage2-m2kl")
        ],
    }
    admission.validate_probe_training_arms(candidate)


def test_nonfinite_probe_metrics_fail_closed():
    spec = importlib.util.spec_from_file_location("code_stage123_probe_phase_nonfinite", PHASE)
    phase = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(phase)
    observed = {
        "training/global_step": 1,
        "actor/lr": 1e-6,
        "wdl_sft/n_correct": 1,
        "actor/wdl_sft_loss_positive": float("inf"),
        "actor/grad_norm": float("inf"),
        "timing_s/update_actor": 1.0,
    }
    passed, missing = phase.training_contract(observed)
    assert passed is False
    assert missing == ["positive_loss", "nonzero_gradient"]


def test_frozen_zero_positive_probe_report_is_rejected():
    admission = admission_module()
    report_path = Path(
        "/data-1/tmp/verl_agent_scratch/code_stage123_gpu_utilization_probe/"
        "probe-20260723T170300Z/gpu-utilization-probe-report.json"
    )
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text())
    candidate = next(item for item in report["candidates"] if item["status"] == "passed")
    try:
        admission.validate_probe_training_arms(candidate)
    except SystemExit as exc:
        message = str(exc)
        assert "beta=0 Stage1" in message or "no positive samples" in message
    else:
        raise AssertionError("historical zero-positive probe report passed the new admission gate")
