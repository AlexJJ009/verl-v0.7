from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILY = REPO_ROOT / "recipe/on_policy_wdl_sft/rebuttal_rlvr"


def load_validator():
    path = FAMILY / "validate_inputs.py"
    spec = importlib.util.spec_from_file_location("rebuttal_rlvr_validate_inputs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_validator()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_model_catalog_separates_am1p4m_initialization_from_math_rlvr() -> None:
    catalog = (FAMILY / "model_paths.env").read_text()
    assert "R01_ORDINARY_SFT_4B_AM1P4M" in catalog
    assert "INIT_SFT_DATASET_ID=${INIT_SFT_DATASET_ID:-AM-1.4M}" in catalog
    assert "RLVR_TRAIN_DATASET_ID=${RLVR_TRAIN_DATASET_ID:-hendrycks_math_7500}" in catalog


def test_r01_named_placeholder_fails_before_launch(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(FAMILY / "run_experiment.sh"), "R01"],
        env={**os.environ, "ROOT": str(tmp_path), "MODEL_ROOT": str(tmp_path / "models")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "R01_ORDINARY_SFT_4B_AM1P4M" in result.stderr
    assert "pre-registered but unresolved" in result.stderr


def test_colleague_handoff_fails_with_one_actionable_missing_config() -> None:
    handoff = REPO_ROOT / "platform/hope_rebuttal_rlvr/run_handoff.sh"
    result = subprocess.run(
        ["bash", str(handoff)],
        env={**os.environ, "HANDOFF_ENV": "/definitely/missing/rebuttal-handoff.env"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "handoff config is missing" in result.stderr
    assert "handoff.env.example" in result.stderr


def test_direct_colleague_entry_runs_without_provenance_receipts_in_dry_run(tmp_path: Path) -> None:
    for model_name in ("R01_ORDINARY_SFT_4B_AM1P4M", "R02_WDL_SFT_4B_AM1P4M"):
        model = tmp_path / "models/rebuttal_rlvr/init" / model_name
        model.mkdir(parents=True)
        (model / "config.json").write_text(
            json.dumps({"model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36})
        )
        (model / "model.safetensors").write_bytes(b"placeholder")
    required_data = [
        "data/math/train_rl_format.parquet",
        "data/math7/AIME-2025/aime-2025_with_system_prompt.parquet",
        "data/math7/MATH-500/math500-test_with_system_prompt.parquet",
        "data/math7/AMC23/amc23-test_with_system_prompt.parquet",
        "data/math7/AQUA/aqua-test_with_system_prompt.parquet",
        "data/math7/gsm8k/gsm8k-test_with_system_prompt.parquet",
        "data/math7/MAWPS/mawps-test_with_system_prompt.parquet",
        "data/math7/SVAMP/svamp-test_with_system_prompt.parquet",
    ]
    for relative in required_data:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")
    for experiment in ("R01", "R02"):
        result = subprocess.run(
            ["bash", str(FAMILY / "run_colleague.sh"), experiment, "20260727"],
            env={**os.environ, "ROOT": str(tmp_path), "DRY_RUN": "1", "ATTEMPT_ID": f"test-{experiment}"},
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
    assumption_files = list((tmp_path / "verl-exp/receipts/rebuttal_rlvr").rglob("external_provenance_assumption.env"))
    assert len(assumption_files) == 2
    assumption = assumption_files[0].read_text()
    assert "policy=conditional_checkpoint_comparison" in assumption
    assert "claim_boundary=results_apply_only_to_the_two_supplied_checkpoints" in assumption


def test_direct_colleague_entry_rejects_unregistered_seed(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(FAMILY / "run_colleague.sh"), "R01", "7"],
        env={**os.environ, "ROOT": str(tmp_path)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "RLVR_SEED must be one of" in result.stderr


def test_direct_matrix_entry_covers_exact_six_cells() -> None:
    matrix = (REPO_ROOT / "platform/hope_rebuttal_rlvr/run_colleague_matrix.sh").read_text()
    assert "for rl_seed in 20260727 20260728 20260729" in matrix
    assert "for experiment in R01 R02" in matrix
    assert 'bash "${SCRIPT_DIR}/run_colleague.sh" "$experiment" "$rl_seed"' in matrix


def test_rebuttal_registry_import_requires_and_consumes_success_gate(tmp_path: Path) -> None:
    run_name = "rebuttal-rlvr-sft-I1-r20260727-test"
    gate_state = tmp_path / "gate.jsonl"
    registry_db = tmp_path / "registry.sqlite"
    init_model = tmp_path / "init"
    checkpoint_dir = tmp_path / "checkpoints"
    final_actor = checkpoint_dir / "global_step_115/actor"
    metrics = tmp_path / "metrics.jsonl"
    train_file = tmp_path / "train.parquet"
    init_model.mkdir()
    final_actor.mkdir(parents=True)
    metrics.write_text(json.dumps({"step": 115, "data": {"val-core/HuggingFaceH4/MATH-500/acc/mean@3": 0.5}}) + "\n")
    train_file.write_bytes(b"test-parquet-placeholder")

    common = [
        sys.executable,
        str(REPO_ROOT / "scripts/import_rebuttal_rlvr_registry.py"),
        "--run-name", run_name,
        "--arm", "sft",
        "--init-pair", "I1",
        "--rl-seed", "20260727",
        "--init-model-path", str(init_model),
        "--checkpoint-dir", str(checkpoint_dir),
        "--metrics-path", str(metrics),
        "--train-file", str(train_file),
        "--final-step", "115",
        "--db", str(registry_db),
    ]
    env = {**os.environ, "TRAINING_RELEASE_GATE_STATE": str(gate_state)}
    blocked = subprocess.run(common, env=env, text=True, capture_output=True, check=False)
    assert blocked.returncode != 0

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/training_result_release_gate.py"),
            "--state", str(gate_state),
            "record",
            "--run-name", run_name,
            "--family", "rebuttal_rlvr_sft",
            "--status", "success_complete",
            "--checkpoint", str(final_actor.parent),
            "--metrics", str(metrics),
            "--final-step", "115",
            "--observed-step", "115",
        ],
        check=True,
    )
    imported = subprocess.run(common, env=env, text=True, capture_output=True, check=False)
    assert imported.returncode == 0, imported.stderr
    assert "verl.rebuttal_rlvr.training" in imported.stdout


def test_release_hook_imports_registry_and_preserves_offline_wandb_after_terminal_success(tmp_path: Path) -> None:
    init_model = tmp_path / "init"
    checkpoints = tmp_path / "checkpoints"
    final_actor = checkpoints / "global_step_115/actor"
    log_dir = tmp_path / "logs"
    wandb_root = tmp_path / "wandb"
    offline_run = wandb_root / "offline-run-test"
    attempt = tmp_path / "attempt"
    train_file = tmp_path / "train.parquet"
    fake_bin = tmp_path / "bin"
    for directory in (init_model, final_actor, log_dir, offline_run, attempt, fake_bin):
        directory.mkdir(parents=True, exist_ok=True)
    (log_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 115, "data": {"val-core/HuggingFaceH4/MATH-500/acc/mean@3": 0.5}}) + "\n"
    )
    train_file.write_bytes(b"test-parquet-placeholder")
    (offline_run / "run-test.wandb").write_bytes(b"offline-wandb-data")
    fake_wandb = fake_bin / "wandb"
    fake_wandb.write_text('#!/usr/bin/env bash\necho called >"$WANDB_CALLED_MARKER"\nexit 99\n')
    fake_wandb.chmod(0o755)
    wandb_called_marker = tmp_path / "wandb-command-called"

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "ROOT": str(tmp_path),
        "ARM": "sft",
        "INIT_PAIR": "I1",
        "RLVR_SEED": "20260727",
        "INIT_MODEL_PATH": str(init_model),
        "TRAIN_FILE": str(train_file),
        "WANDB_RUN_NAME": "rebuttal-rlvr-sft-I1-r20260727-release-test",
        "CKPTS_DIR": str(checkpoints),
        "RUN_LOG_DIR": str(log_dir),
        "RUN_WANDB_DIR": str(wandb_root),
        "ATTEMPT_ROOT": str(attempt),
        "EXPERIMENT_REGISTRY_DB": str(tmp_path / "registry.sqlite"),
        "TRAINING_RELEASE_GATE_STATE": str(tmp_path / "gate.jsonl"),
        "WANDB_MODE": "offline",
        "WANDB_CALLED_MARKER": str(wandb_called_marker),
    }
    result = subprocess.run(
        ["bash", str(FAMILY / "release_after_success.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr, (attempt / "release.log").read_text())
    release_status = (attempt / "release_status.env").read_text()
    assert "release_status=local_complete" in release_status
    assert "wandb_mode=offline" in release_status
    assert "wandb_publication_status=deferred_manual_handoff" in release_status
    assert f"wandb_offline_dir={offline_run}" in release_status
    assert (attempt / "wandb_offline_run.sha256").is_file()
    assert "run-test.wandb" in (attempt / "wandb_offline_run.sha256").read_text()
    assert not wandb_called_marker.exists()

    launcher = (FAMILY / "_common_math_rlvr.sh").read_text()
    assert 'if [ "$RUN_MODE" = "formal" ] || [ "$RUN_MODE" = "external_checkpoint_assumption" ]; then' in launcher
    assert 'bash "${SCRIPT_DIR}/release_after_success.sh"' in launcher


def with_self_hash(value: dict, field: str) -> dict:
    value = dict(value)
    value[field] = hashlib.sha256(VALIDATOR.canonical_json(value)).hexdigest()
    return value


def h20_platform_resources() -> dict:
    return {
        "usergroup": "hadoop-ai-search",
        "queue": "root.shxs_training_cluster.hadoop-fridayagi.friday_h20_train",
        "workers": 1,
        "worker_memory_mb": 1920000,
        "worker_vcore": 128,
        "gpu_resource_key": "worker.gcoresh20-141g",
        "gpu_count": 8,
        "worker_script": "bash jupyter.sh",
        "shm_size_bytes": 549755813888,
        "max_retry": 0,
        "failover": False,
    }


def h20_selected() -> dict:
    return {
        "rollout_gpu_memory_utilization": 0.7,
        "generation_micro_batch_size": 32,
        "log_prob_micro_batch_size": 8,
        "actor_ppo_max_token_len": 18384,
    }


def h20_fixed() -> dict:
    return {
        "tensor_parallel_size": 1,
        "rollout_agent_num_workers": 4,
        "rollout_max_num_seqs": 256,
        "rollout_enforce_eager": True,
        "rollout_enable_chunked_prefill": True,
        "rollout_max_model_len": 4596,
        "log_prob_max_token_len_per_gpu": 4596,
        "actor_param_offload": False,
        "actor_optimizer_offload": False,
    }


def h20_runtime_versions() -> dict:
    return {
        "nvidia_driver": "590.48.01",
        "cuda_driver": "13.1",
        "cuda_runtime": "12.6",
        "pytorch": "2.9.1+cu126",
        "vllm": "0.12.0",
        "flashinfer": "0.5.3",
    }


def write_h20_run_hope(path: Path, image_digest: str) -> None:
    path.write_text(
        """[resource]
usergroup = hadoop-ai-search
queue = root.shxs_training_cluster.hadoop-fridayagi.friday_h20_train

[roles]
workers = 1
worker.memory = 1920000
worker.vcore = 128
worker.gcoresh20-141g = 8
worker.script = bash jupyter.sh

[docker]
afo.docker.image.name = registry.example/verl@{image_digest}

[failover]
afo.app.support.engine.failover = false

[others]
afo.app.env.YARN_CONTAINER_RUNTIME_DOCKER_SHM_SIZE_BYTES = 549755813888
afo.role.worker.task.attempt.max.retry = 0
""".format(image_digest=image_digest)
    )


def write_signed_h20_bundle(tmp_path: Path, image_digest: str) -> tuple[dict, Path, Path, Path, Path]:
    resources = h20_platform_resources()
    selected = h20_selected()
    fixed = h20_fixed()
    runtime_versions = h20_runtime_versions()
    arm_metrics = {
        "sft": {"peak_gpu_memory_fraction": 0.8, "median_step_seconds": 10.0, "step_time_cv": 0.1},
        "wdl": {"peak_gpu_memory_fraction": 0.85, "median_step_seconds": 11.0, "step_time_cv": 0.12},
    }
    profile = with_self_hash(
        {
            "schema_version": 1,
            "receipt_kind": "h20_system_profile",
            "profile_status": "formal_frozen",
            "image_digest": image_digest,
            "runtime_versions": runtime_versions,
            "gpu_count": 8,
            "platform_resources": resources,
            "selected": selected,
            "fixed": fixed,
            "arm_metrics": arm_metrics,
        },
        "receipt_sha256",
    )
    profile_path = tmp_path / "h20-profile.json"
    profile_path.write_text(json.dumps(profile, sort_keys=True))

    calibration_id = "calibration-test-1"
    terminals = {}
    for arm in ("sft", "wdl"):
        staged = tmp_path / f"{arm}.run.hope"
        write_h20_run_hope(staged, image_digest)
        status = tmp_path / f"{arm}.status.txt"
        status.write_text(f"job-{arm} SUCCEEDED\n")
        metrics = {
            **arm_metrics[arm],
            "finite_loss_and_grad_norm": True,
            "worker_retry_count": 0,
            "measured_steps": [3, 4, 5],
        }
        worker = tmp_path / f"{arm}.worker.json"
        worker_evidence = with_self_hash(
            {
                "schema_version": 1,
                "receipt_kind": "h20_calibration_worker_evidence",
                "calibration_id": calibration_id,
                "arm": arm,
                "job_id": f"job-{arm}",
                "image_digest": image_digest,
                "runtime_versions": runtime_versions,
                "runtime_versions_sha256": VALIDATOR.canonical_projection_hash(runtime_versions),
                "metrics": metrics,
            },
            "receipt_sha256",
        )
        worker.write_text(json.dumps(worker_evidence, sort_keys=True))
        terminal = with_self_hash(
            {
                "schema_version": 1,
                "receipt_kind": "h20_calibration_terminal",
                "calibration_id": calibration_id,
                "arm": arm,
                "state": "SUCCEEDED",
                "job_id": f"job-{arm}",
                "app_name": f"calibration-{arm}",
                "image_digest": image_digest,
                "runtime_versions": runtime_versions,
                "platform_resources": resources,
                "selected": selected,
                "fixed": fixed,
                "staged_run_hope": {"path": str(staged), "sha256": sha256(staged)},
                "status_evidence": {"path": str(status), "sha256": sha256(status)},
                "worker_evidence": {"path": str(worker), "sha256": sha256(worker)},
                "metrics": metrics,
            },
            "receipt_sha256",
        )
        terminal_path = tmp_path / f"{arm}-terminal.json"
        terminal_path.write_text(json.dumps(terminal, sort_keys=True))
        terminals[arm] = terminal_path

    key = tmp_path / "reviewer_ed25519"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    allowlist = tmp_path / "reviewer-keys.json"
    allowlist.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reviewers": [
                    {
                        "key_id": "h20-reviewer-1",
                        "principal": "h20-reviewer",
                        "owner_identity": "uid:99999:reviewer",
                        "public_key": (tmp_path / "reviewer_ed25519.pub").read_text().strip(),
                    }
                ],
            },
            sort_keys=True,
        )
    )
    review = tmp_path / "h20-review.md"
    review.write_text("Independent H20 calibration review: PASS\n")
    attestation = {
        "schema_version": 1,
        "gate": "G4",
        "status": "passed",
        "approval_scope": "rebuttal-h20-common-v1",
        "calibration_id": calibration_id,
        "h20_profile_path": str(profile_path),
        "h20_profile_sha256": sha256(profile_path),
        "sft_terminal_receipt_path": str(terminals["sft"]),
        "sft_terminal_receipt_sha256": sha256(terminals["sft"]),
        "wdl_terminal_receipt_path": str(terminals["wdl"]),
        "wdl_terminal_receipt_sha256": sha256(terminals["wdl"]),
        "image_digest": image_digest,
        "platform_resources_sha256": VALIDATOR.canonical_projection_hash(resources),
        "runtime_versions_sha256": VALIDATOR.canonical_projection_hash(runtime_versions),
        "selected_system_knobs_sha256": VALIDATOR.canonical_projection_hash(selected),
        "fixed_system_knobs_sha256": VALIDATOR.canonical_projection_hash(fixed),
        "selection_policy_version": "rebuttal-h20-common-selection-v1",
        "calibration_submitter_identity": "uid:1000:submitter",
        "reviewer_key_id": "h20-reviewer-1",
        "review_evidence_path": str(review),
        "review_evidence_sha256": sha256(review),
    }
    payload = tmp_path / "h20-attestation.json"
    payload.write_bytes(VALIDATOR.canonical_json(attestation) + b"\n")
    subprocess.run(
        ["ssh-keygen", "-Y", "sign", "-f", str(key), "-n", "rebuttal-rlvr-g4", str(payload)],
        check=True,
        capture_output=True,
    )
    signature = Path(str(payload) + ".sig")
    admission = with_self_hash(
        {
            "schema_version": 1,
            "receipt_kind": "h20_calibration_admission",
            "approved": True,
            "status": "passed",
            "approval_scope": "rebuttal-h20-common-v1",
            "calibration_id": calibration_id,
            "h20_profile_path": str(profile_path),
            "h20_profile_sha256": sha256(profile_path),
            "terminal_receipts": {
                arm: {"path": str(path), "sha256": sha256(path)} for arm, path in terminals.items()
            },
            "reviewer_key_id": "h20-reviewer-1",
            "attestation_payload_path": str(payload),
            "attestation_payload_sha256": sha256(payload),
            "attestation_signature_path": str(signature),
            "attestation_signature_sha256": sha256(signature),
        },
        "receipt_sha256",
    )
    admission_path = tmp_path / "h20-calibration.json"
    admission_path.write_text(json.dumps(admission, sort_keys=True))
    current_hope = tmp_path / "current.run.hope"
    write_h20_run_hope(current_hope, image_digest)
    return profile, profile_path, admission_path, allowlist, current_hope


def test_h20_runtime_schema_requires_exact_projection() -> None:
    base = {
        "schema_version": 1,
        "receipt_kind": "h20_system_profile",
        "profile_status": "smoke_candidate",
        "image_digest": "sha256:" + "a" * 64,
        "runtime_versions": h20_runtime_versions(),
        "gpu_count": 8,
        "platform_resources": h20_platform_resources(),
        "selected": h20_selected(),
        "fixed": h20_fixed(),
        "receipt_sha256": "b" * 64,
    }
    for mutation in ("missing", "extra"):
        candidate = json.loads(json.dumps(base))
        if mutation == "missing":
            candidate["runtime_versions"].pop("flashinfer")
        else:
            candidate["runtime_versions"]["nvcc"] = "12.6"
        with pytest.raises(VALIDATOR.ValidationError, match="schema validation failed"):
            VALIDATOR.validate_json_schema(candidate, VALIDATOR.H20_PROFILE_SCHEMA, "H20 profile")


def test_h20_runtime_guard_rejects_live_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_digest = "sha256:" + "4" * 64
    profile, profile_path, admission_path, allowlist, rendered_hope = write_signed_h20_bundle(
        tmp_path, image_digest
    )
    observed = {**profile["runtime_versions"], "pytorch": "2.9.2+cu126"}
    monkeypatch.setattr(VALIDATOR, "probe_runtime_versions", lambda expected_gpu_count: observed)
    with pytest.raises(VALIDATOR.ValidationError, match="live runtime_versions differ"):
        VALIDATOR.validate_h20_profile(
            profile,
            profile_path,
            admission_path,
            allowlist,
            rendered_hope,
            tmp_path,
            image_digest,
            "formal",
            tmp_path / "h20.env",
        )


@pytest.mark.parametrize("drift", ["terminal", "worker", "attestation"])
def test_h20_calibration_runtime_chain_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    image_digest = "sha256:" + "4" * 64
    profile, profile_path, admission_path, allowlist, rendered_hope = write_signed_h20_bundle(
        tmp_path, image_digest
    )
    monkeypatch.setattr(
        VALIDATOR,
        "probe_runtime_versions",
        lambda expected_gpu_count: profile["runtime_versions"],
    )
    admission = json.loads(admission_path.read_text())
    expected_message = ""
    if drift in {"terminal", "worker"}:
        terminal_path = Path(admission["terminal_receipts"]["sft"]["path"])
        terminal = json.loads(terminal_path.read_text())
        if drift == "terminal":
            terminal["runtime_versions"]["pytorch"] = "2.9.2+cu126"
            expected_message = "terminal receipt differs on runtime_versions"
        else:
            worker_path = Path(terminal["worker_evidence"]["path"])
            worker = json.loads(worker_path.read_text())
            worker.pop("receipt_sha256")
            worker["runtime_versions"]["pytorch"] = "2.9.2+cu126"
            worker["runtime_versions_sha256"] = VALIDATOR.canonical_projection_hash(
                worker["runtime_versions"]
            )
            worker = with_self_hash(worker, "receipt_sha256")
            worker_path.write_text(json.dumps(worker, sort_keys=True))
            terminal["worker_evidence"]["sha256"] = sha256(worker_path)
            expected_message = "worker runtime differs"
        terminal.pop("receipt_sha256")
        terminal = with_self_hash(terminal, "receipt_sha256")
        terminal_path.write_text(json.dumps(terminal, sort_keys=True))
        admission["terminal_receipts"]["sft"]["sha256"] = sha256(terminal_path)
    else:
        payload_path = Path(admission["attestation_payload_path"])
        payload = json.loads(payload_path.read_text())
        payload["runtime_versions_sha256"] = "0" * 64
        payload_path.write_bytes(VALIDATOR.canonical_json(payload) + b"\n")
        admission["attestation_payload_sha256"] = sha256(payload_path)
        expected_message = "attestation binding mismatch"
    admission.pop("receipt_sha256")
    admission = with_self_hash(admission, "receipt_sha256")
    admission_path.write_text(json.dumps(admission, sort_keys=True))
    with pytest.raises(VALIDATOR.ValidationError, match=expected_message):
        VALIDATOR.validate_h20_profile(
            profile,
            profile_path,
            admission_path,
            allowlist,
            rendered_hope,
            tmp_path,
            image_digest,
            "formal",
            tmp_path / "h20.env",
        )


def test_runtime_probe_rejects_inconsistent_driver_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command, **kwargs):
        if "--query-gpu=driver_version" in command:
            stdout = "590.48.01\n" * 7 + "591.00.00\n"
        else:
            stdout = "NVIDIA-SMI 590.48.01  Driver Version: 590.48.01  CUDA Version: 13.1\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(VALIDATOR.subprocess, "run", fake_run)
    with pytest.raises(VALIDATOR.ValidationError, match="one version on exactly 8 GPUs"):
        VALIDATOR.probe_runtime_versions(8)


def set_nested(root: dict, dotted: str, value) -> None:
    current = root
    parts = dotted.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def test_frozen_config_contains_reviewed_standard_grpo_v2_values() -> None:
    text = (FAMILY / "frozen_grpo_v2.env").read_text()
    expected = {
        'export REBUTTAL_GRPO_CONFIG_VERSION="rebuttal-standard-grpo-v2"',
        'export POLICY_LOSS_MODE="vanilla"',
        'export LR="5e-7"',
        'export CLIP_RATIO="0.2"',
        'export CLIP_RATIO_LOW="0.2"',
        'export CLIP_RATIO_HIGH="0.2"',
        'export CLIP_RATIO_C="3.0"',
        'export LOSS_AGG_MODE="token-mean"',
        'export OPTIMIZER_IMPL="torch.optim"',
        'export OPTIMIZER_BETAS="[0.9,0.999]"',
        'export OPTIMIZER_EPS="1e-8"',
        'export ZERO_INDEXED_STEP="true"',
        'export PPO_EPOCHS="1"',
        'export GRAD_CLIP="1.0"',
        'export DATA_SHUFFLE="false"',
        'export REWARD_MANAGER="naive"',
        'export USE_KL_LOSS="true"',
        'export KL_LOSS_COEF="0.001"',
        'export KL_LOSS_TYPE="low_var_kl"',
        'export ROLLOUT_IS="null"',
        'export ROLLOUT_IS_THRESHOLD="null"',
        'export ROLLOUT_IS_BATCH_NORMALIZE="false"',
        'export ROLLOUT_RS="null"',
        'export ALL_CORRECT_SFT_FALLBACK="false"',
        'export CALCULATE_ENTROPY="true"',
        'export VAL_N="3"',
        'export SAVE_FREQ="5"',
        'export TEST_FREQ="5"',
        'export VAL_BEFORE_TRAIN="true"',
        'export KEEP_BEST_CKPT="true"',
        'export MAX_ACTOR_CKPTS_TO_KEEP="1"',
    }
    assert expected.issubset(set(text.splitlines()))


def test_family_paths_are_root_derived_and_repo_launch_is_relative() -> None:
    common = (FAMILY / "_common_math_rlvr.sh").read_text()
    models = (FAMILY / "model_paths.env").read_text()
    entry = (FAMILY / "run_experiment.sh").read_text()

    assert "/data-1/dataset" not in common
    assert '${ROOT}/dataset/math/train_rl_format.parquet' in common
    assert 'HF_MODEL_CACHE_ROOT=${HF_MODEL_CACHE_ROOT:-"${ROOT}/.cache/huggingface"}' in models
    assert "WDL_4B_WEIGHT_BYTES=${WDL_4B_WEIGHT_BYTES:-8045067711}" in models
    assert "WDL_4B_WEIGHT_SHA256=${WDL_4B_WEIGHT_SHA256:-3267350" in models
    assert 'run_script="${SCRIPT_DIR}/run_math_wdl.sh"' in entry


def test_resolved_config_guard_rejects_rollout_is_drift(tmp_path: Path) -> None:
    seed = 20260727
    model = "/models/test"
    config: dict = {}
    for key, value in VALIDATOR.expected_resolved_values(seed, model).items():
        set_nested(config, key, value)
    path = tmp_path / "resolved.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=True))
    VALIDATOR.validate_resolved_config(path, seed, model)

    set_nested(config, "algorithm.rollout_correction.rollout_is", "token")
    path.write_text(yaml.safe_dump(config, sort_keys=True))
    with pytest.raises(VALIDATOR.ValidationError, match="rollout_is"):
        VALIDATOR.validate_resolved_config(path, seed, model)


def test_resolved_config_cli_defaults_to_five_step_save_cadence() -> None:
    args = VALIDATOR.build_parser().parse_args(
        ["resolved-config", "--config", "/tmp/resolved.yaml"]
    )

    assert args.total_training_steps == 115
    assert args.save_freq == 5


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("actor_rollout_ref.rollout.response_length", 2048),
        ("actor_rollout_ref.actor.entropy_coeff", 0.01),
        ("actor_rollout_ref.actor.calculate_entropy", False),
        ("actor_rollout_ref.actor.clip_ratio", 0.3),
        ("actor_rollout_ref.actor.optim.optimizer_impl", "torchao.optim"),
        ("actor_rollout_ref.actor.optim.betas", [0.9, 0.95]),
        ("actor_rollout_ref.actor.optim.zero_indexed_step", False),
        ("actor_rollout_ref.actor.optim.override_optimizer_config", {"eps": 1e-6}),
        ("data.shuffle", True),
        ("actor_rollout_ref.actor.shuffle", True),
        ("algorithm.rollout_correction.bypass_mode", True),
        ("algorithm.rollout_correction.loss_type", "bypass"),
        ("trainer.total_training_steps", 114),
        ("trainer.save_freq", 25),
        ("trainer.max_actor_ckpt_to_keep", 5),
    ],
)
def test_resolved_config_guard_covers_frozen_surface(tmp_path: Path, key: str, bad_value) -> None:
    config: dict = {}
    for dotted, value in VALIDATOR.expected_resolved_values(20260727, "/models/test").items():
        set_nested(config, dotted, value)
    set_nested(config, key, bad_value)
    path = tmp_path / "resolved.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=True))
    with pytest.raises(VALIDATOR.ValidationError, match=key.split(".")[-1]):
        VALIDATOR.validate_resolved_config(path, 20260727, "/models/test")


def test_checkpoint_and_pair_receipts_bind_model_files(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    (model / "tokenizer.json").write_text("{}\n")
    (model / "model.safetensors").write_bytes(b"small-test-weight")
    files = {name: sha256(model / name) for name in ("config.json", "tokenizer.json", "model.safetensors")}

    receipt = with_self_hash(
        {
            "schema_version": 1,
            "classifier": "ordinary_sft",
            "model_path": str(model),
            "files": files,
            "base_model_revision": "Qwen/Qwen3-4B-Base@revision",
            "architecture": "Qwen3ForCausalLM",
            "tokenizer_hash": "1" * 64,
            "dataset_receipt_sha256": "2" * 64,
            "prompt_template_hash": "3" * 64,
            "initialization_seed": 7,
            "optimizer": "AdamW",
            "optimizer_updates": 10,
            "target_supervised_tokens": 100,
            "learning_rate_schedule": "constant",
            "checkpoint_selection_rule": "final_predeclared_update",
            "training_code_commit": "4" * 40,
            "post_checkpoint_rl": False,
        },
        "receipt_sha256",
    )
    receipt_path = tmp_path / "sft-receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    pair = with_self_hash(
        {
            "schema_version": 1,
            "status": "admitted",
            "admission_scope": "fixed_pair_pilot",
            "init_pair": "I1",
            "ordinary_sft_receipt": {"path": str(receipt_path), "sha256": sha256(receipt_path)},
            "offline_wdl_sft_receipt": {"path": "/unused/wdl.json", "sha256": "5" * 64},
            "matched_fields": {field: True for field in VALIDATOR.MATCHED_FIELDS},
            "candidate_universe_hash": "6" * 64,
            "selection_timestamp": "2026-07-27T00:00:00Z",
            "historical_math7_access_disclosed": True,
        },
        "manifest_sha256",
    )
    pair_path = tmp_path / "pair.json"
    pair_path.write_text(json.dumps(pair, sort_keys=True))

    VALIDATOR.validate_checkpoint_receipt(receipt, receipt_path, "sft", model)
    VALIDATOR.validate_pair_manifest(pair, pair_path, "sft", receipt_path)
    (model / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(VALIDATOR.ValidationError, match="hash mismatch"):
        VALIDATOR.validate_checkpoint_receipt(receipt, receipt_path, "sft", model)


def test_hf_model_validator_accepts_hash_bound_pytorch_bin(tmp_path: Path) -> None:
    model = tmp_path / "public-wdl-4b"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}\n')
    (model / "tokenizer.json").write_text("{}\n")
    (model / "pytorch_model.bin").write_bytes(b"test-only-pytorch-weights")

    assert VALIDATOR.validate_hf_model(model) == {
        "config.json",
        "tokenizer.json",
        "pytorch_model.bin",
    }


def smoke_environment(tmp_path: Path) -> dict[str, str]:
    model = tmp_path / "base-model"
    model.mkdir(exist_ok=True)
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    paths = {name: inputs / f"{name}.parquet" for name in ("train", "aime", "math500", "amc23", "aqua", "gsm8k", "mawps", "svamp")}
    for path in paths.values():
        path.touch()
    return {
        **os.environ,
        "RUN_MODE": "smoke",
        "ALLOW_BASE_PLACEHOLDER": "1",
        "DRY_RUN": "1",
        "BASE_PLACEHOLDER_MODEL_PATH": str(model),
        "ROOT": str(tmp_path),
        "OUTPUT_ROOT": str(tmp_path / "output"),
        "RAY_TMPDIR": str(tmp_path / "ray"),
        "TMPDIR": str(tmp_path / "tmp"),
        "TRAIN_FILE": str(paths["train"]),
        "MATH7_AIME_FILE": str(paths["aime"]),
        "MATH7_MATH500_FILE": str(paths["math500"]),
        "MATH7_AMC23_FILE": str(paths["amc23"]),
        "MATH7_AQUA_FILE": str(paths["aqua"]),
        "MATH7_GSM8K_FILE": str(paths["gsm8k"]),
        "MATH7_MAWPS_FILE": str(paths["mawps"]),
        "MATH7_SVAMP_FILE": str(paths["svamp"]),
    }


def test_base_placeholder_renders_smoke_but_fails_formal(tmp_path: Path) -> None:
    wrapper = FAMILY / "run_math_wdl.sh"
    smoke = subprocess.run(["bash", str(wrapper)], env=smoke_environment(tmp_path), text=True, capture_output=True)
    assert smoke.returncode == 0, smoke.stderr
    assert "policy_loss.loss_mode=vanilla" in smoke.stdout
    assert "rollout_correction.rollout_is=null" in smoke.stdout
    assert "model.path=" + str(tmp_path / "base-model") in smoke.stdout
    assert "SMOKE-wdl-I1-r20260727" in smoke.stdout

    formal_env = smoke_environment(tmp_path)
    formal_env["RUN_MODE"] = "formal"
    formal = subprocess.run(["bash", str(wrapper)], env=formal_env, text=True, capture_output=True)
    assert formal.returncode == 2
    assert "SMOKE_ONLY" in formal.stderr


def test_unified_entrypoint_maps_r02_to_portable_wdl_4b_path(tmp_path: Path) -> None:
    env = smoke_environment(tmp_path)
    model = tmp_path / "portable-cache" / "wdl-4b"
    model.mkdir(parents=True)
    env.update(
        {
            "ALLOW_BASE_PLACEHOLDER": "0",
            "WDL_4B_MODEL_PATH": str(model),
        }
    )

    result = subprocess.run(
        ["bash", str(FAMILY / "run_experiment.sh"), "R02"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"model.path={model}" in result.stdout
    assert "REBUTTAL-RLVR-MATH-OFFLINE-WDL-SFT" in result.stdout


def test_unified_entrypoint_fails_closed_for_unpublished_r03(tmp_path: Path) -> None:
    env = smoke_environment(tmp_path)
    env["ALLOW_BASE_PLACEHOLDER"] = "0"

    result = subprocess.run(
        ["bash", str(FAMILY / "run_experiment.sh"), "R03"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "public 8B model ID" in result.stderr


def test_platform_mode_ignores_inherited_grader_override(tmp_path: Path) -> None:
    env = smoke_environment(tmp_path)
    env.update(
        {
            "REQUIRE_PLATFORM_RECEIPTS": "1",
            "ALGORITHM_CONFIG_HASH": sha256(FAMILY / "frozen_grpo_v2.env"),
            "CUSTOM_REWARD_FN_PATH": "/tmp/evil_reward.py",
            "CUSTOM_REWARD_FN_NAME": "evil_score",
        }
    )
    result = subprocess.run(
        ["bash", str(FAMILY / "run_math_sft.sh")],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    expected = REPO_ROOT / "recipe/joint_training/custom_reward_function_latex_verify.py"
    assert f"custom_reward_function.path={expected}" in result.stdout
    assert "custom_reward_function.name=compute_score_latex_verify" in result.stdout
    assert "/tmp/evil_reward.py" not in result.stdout


def test_live_train_math7_grader_and_h20_receipts_are_content_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = tmp_path / "train.parquet"
    train.write_bytes(b"train")
    row_counts: dict[Path, int] = {train: 7500}
    monkeypatch.setattr(VALIDATOR, "parquet_row_count", lambda path: row_counts[path])
    monkeypatch.setattr(
        VALIDATOR,
        "filtered_train_projection",
        lambda path, tokenizer_model: (7405, "1" * 64, "3" * 64),
    )

    train_receipt = with_self_hash(
        {
            "schema_version": 1,
            "receipt_kind": "rlvr_train_dataset",
            "path": str(train),
            "sha256": sha256(train),
            "source_data": "ck46/hendrycks_math",
            "source_row_count": 7500,
            "filtered_row_count": 7405,
            "filtered_row_ids_sha256": "1" * 64,
            "prompt_template_sha256": "3" * 64,
        },
        "receipt_sha256",
    )
    VALIDATOR.validate_train_receipt(train_receipt, train, tmp_path / "model")
    bad_train_receipt = with_self_hash(
        {**{key: value for key, value in train_receipt.items() if key != "receipt_sha256"}, "filtered_row_ids_sha256": "9" * 64},
        "receipt_sha256",
    )
    with pytest.raises(VALIDATOR.ValidationError, match="row-ID"):
        VALIDATOR.validate_train_receipt(bad_train_receipt, train, tmp_path / "model")
    train.write_bytes(b"tampered")
    with pytest.raises(VALIDATOR.ValidationError, match="hash mismatch"):
        VALIDATOR.validate_train_receipt(train_receipt, train, tmp_path / "model")

    repo = tmp_path / "repo"
    grader = repo / "recipe/joint_training/custom_reward_function_latex_verify.py"
    grader.parent.mkdir(parents=True)
    grader.write_text("def compute_score_latex_verify():\n    return 1\n")
    image_digest = "sha256:" + "4" * 64

    def make_grader_receipt() -> dict:
        return with_self_hash(
            {
                "schema_version": 1,
                "receipt_kind": "latex_verify_grader",
                "path": str(grader),
                "sha256": sha256(grader),
                "function_name": "compute_score_latex_verify",
                "recipe_commit": "5" * 40,
                "image_digest": image_digest,
            },
            "receipt_sha256",
        )

    grader_receipt = make_grader_receipt()
    with monkeypatch.context() as grader_patch:
        grader_patch.setattr(
            VALIDATOR.subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "5" * 40 + "\n", ""),
        )
        VALIDATOR.validate_grader_receipt(grader_receipt, grader, repo, image_digest)
        grader.write_text("def other_function():\n    return 1\n")
        with pytest.raises(VALIDATOR.ValidationError, match="hash mismatch"):
            VALIDATOR.validate_grader_receipt(grader_receipt, grader, repo, image_digest)

    grader.write_text("def compute_score_latex_verify():\n    return 1\n")
    grader_receipt = make_grader_receipt()
    grader_receipt_path = tmp_path / "grader.json"
    grader_receipt_path.write_text(json.dumps(grader_receipt, sort_keys=True))

    rows = {
        "aime_2025": 30,
        "math_500": 500,
        "amc23": 40,
        "aqua": 254,
        "gsm8k": 1319,
        "mawps": 355,
        "svamp": 300,
    }
    datasets = {}
    expected_paths = {}
    prompt_projections = {}
    for index, (key, count) in enumerate(rows.items(), start=1):
        path = tmp_path / f"{key}.parquet"
        path.write_bytes(key.encode())
        row_counts[path] = count
        expected_paths[key] = path
        prompts = [[{"role": "user", "content": key}]]
        prompt_ids = [index]
        prompt_projections[path] = (prompts, prompt_ids)
        datasets[key] = {
            "path": str(path),
            "sha256": sha256(path),
            "row_count": count,
            "ordered_prompt_ids_sha256": VALIDATOR.canonical_projection_hash(prompt_ids),
            "prompt_template_sha256": VALIDATOR.canonical_projection_hash(prompts),
        }
    monkeypatch.setattr(VALIDATOR, "parquet_prompt_projection", lambda path: prompt_projections[path])
    math7_receipt = with_self_hash(
        {
            "schema_version": 1,
            "receipt_kind": "math7_bundle",
            "datasets": datasets,
            "grader_receipt_sha256": sha256(grader_receipt_path),
        },
        "receipt_sha256",
    )
    VALIDATOR.validate_math7_receipt(math7_receipt, expected_paths, grader_receipt_path)
    bad_math7 = json.loads(json.dumps(math7_receipt))
    bad_math7.pop("receipt_sha256")
    bad_math7["datasets"]["aime_2025"]["ordered_prompt_ids_sha256"] = "9" * 64
    bad_math7 = with_self_hash(bad_math7, "receipt_sha256")
    with pytest.raises(VALIDATOR.ValidationError, match="prompt-ID"):
        VALIDATOR.validate_math7_receipt(bad_math7, expected_paths, grader_receipt_path)
    expected_paths["svamp"].write_bytes(b"changed")
    with pytest.raises(VALIDATOR.ValidationError, match="hash mismatch"):
        VALIDATOR.validate_math7_receipt(math7_receipt, expected_paths, grader_receipt_path)

    profile, profile_path, admission_path, allowlist_path, rendered_hope = write_signed_h20_bundle(
        tmp_path, image_digest
    )
    monkeypatch.setattr(
        VALIDATOR,
        "probe_runtime_versions",
        lambda expected_gpu_count: profile["runtime_versions"],
    )
    env_output = tmp_path / "h20.env"
    VALIDATOR.validate_h20_profile(
        profile,
        profile_path,
        admission_path,
        allowlist_path,
        rendered_hope,
        tmp_path,
        image_digest,
        "formal",
        env_output,
    )
    text = env_output.read_text()
    assert "ROLLOUT_GPU_MEMORY_UTILIZATION='0.7'" in text
    assert "ACTOR_PPO_MAX_TOKEN_LEN='18384'" in text
    with pytest.raises(VALIDATOR.ValidationError, match="signed H20 calibration"):
        VALIDATOR.validate_h20_profile(
            profile,
            profile_path,
            None,
            allowlist_path,
            rendered_hope,
            tmp_path,
            image_digest,
            "formal",
            env_output,
        )

    root = tmp_path / "parent"
    repo_subpath = "repos/verl-rebuttal-rlvr"
    init_model = root / "models/rebuttal_rlvr/init/sft/I1"
    output_root = root / "verl-exp"
    runtime_paths = {
        "output_root": output_root,
        "checkpoint_root": output_root / "checkpoints/rebuttal_rlvr",
        "eval_root": output_root / "eval/rebuttal_rlvr",
        "log_root": output_root / "logs/rebuttal_rlvr",
        "wandb_root": output_root / "wandb_runs/rebuttal_rlvr",
        "receipt_root": output_root / "receipts/rebuttal_rlvr",
        "hf_home": output_root / "cache/hf",
        "huggingface_hub_cache": output_root / "cache/hf/hub",
        "hf_datasets_cache": output_root / "cache/datasets",
        "xdg_cache_home": output_root / "cache/xdg",
        "ray_tmpdir": Path("/tmp/rebuttal_rlvr/ray"),
        "tmpdir": Path("/tmp/rebuttal_rlvr/tmp"),
        "vllm_config_root": Path("/tmp/rebuttal_rlvr/vllm"),
        "zmq_ipc_dir": Path("/tmp/rebuttal_rlvr/zmq"),
    }
    path_receipt = with_self_hash(
        {
            "schema_version": 1,
            "receipt_kind": "rebuttal_parent_root_layout",
            "root": str(root),
            "repo_subpath": repo_subpath,
            "repo_root": str(root / repo_subpath),
            "model_root": str(root / "models/rebuttal_rlvr/init"),
            "init_model_path": str(init_model),
            "train_file": str(root / "data/math/train_rl_format.parquet"),
            "math7_root": str(root / "data/math7"),
            **{key: str(value) for key, value in runtime_paths.items()},
        },
        "receipt_sha256",
    )
    VALIDATOR.validate_path_override_receipt(path_receipt, root, repo_subpath, init_model, "formal", runtime_paths)
    path_receipt["output_root"] = "/outside/output"
    path_receipt = with_self_hash(
        {k: v for k, v in path_receipt.items() if k != "receipt_sha256"},
        "receipt_sha256",
    )
    with pytest.raises(VALIDATOR.ValidationError, match="one-root"):
        VALIDATOR.validate_path_override_receipt(path_receipt, root, repo_subpath, init_model, "formal", runtime_paths)
    outside_model = root / "arbitrary/other-model"
    path_receipt["output_root"] = str(output_root)
    path_receipt["init_model_path"] = str(outside_model)
    path_receipt = with_self_hash(
        {k: v for k, v in path_receipt.items() if k != "receipt_sha256"},
        "receipt_sha256",
    )
    with pytest.raises(VALIDATOR.ValidationError, match="below the receipt model_root"):
        VALIDATOR.validate_path_override_receipt(
            path_receipt,
            root,
            repo_subpath,
            outside_model,
            "formal",
            runtime_paths,
        )


def test_platform_env_overwrites_inherited_paths(tmp_path: Path) -> None:
    root = tmp_path / "root"
    model = root / "models/init"
    model.mkdir(parents=True)
    env = {
        **os.environ,
        "ROOT": str(root),
        "REQUIRE_PLATFORM_RECEIPTS": "1",
        "INIT_MODEL_PATH": str(model),
        "TRAIN_FILE": "/outside/train.parquet",
        "OUTPUT_ROOT": "/outside/output",
        "HF_HOME": "/outside/hf",
        "RAY_TMPDIR": "/outside/ray",
    }
    script = FAMILY / "meituan/env.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\n%s\n%s\n%s\n" "$TRAIN_FILE" "$OUTPUT_ROOT" "$HF_HOME" "$RAY_TMPDIR"',
            "bash",
            str(script),
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        str(root / "data/math/train_rl_format.parquet"),
        str(root / "verl-exp"),
        str(root / "verl-exp/cache/hf"),
        "/tmp/rebuttal_rlvr/ray",
    ]


def test_worker_rejects_receipt_hash_drift_before_recipe_launch(tmp_path: Path) -> None:
    root = tmp_path / "root"
    repo = root / "repo"
    submitter = repo / "platform/hope_rebuttal_rlvr/submit_manifest.py"
    frozen = repo / "recipe/on_policy_wdl_sft/rebuttal_rlvr/frozen_grpo_v2.env"
    submitter.parent.mkdir(parents=True)
    frozen.parent.mkdir(parents=True)
    shutil.copyfile(REPO_ROOT / "platform/hope_rebuttal_rlvr/submit_manifest.py", submitter)
    shutil.copyfile(FAMILY / "frozen_grpo_v2.env", frozen)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    receipts = root / "receipts"
    receipts.mkdir()
    paths = {}
    for name in ("pair", "checkpoint", "train", "math7", "grader", "h20", "h20cal", "paths"):
        path = receipts / f"{name}.json"
        path.write_text(f'{{"name":"{name}"}}\n')
        paths[name] = path
    submodules = receipts / "submodules.json"
    submodules.write_text(json.dumps({"status_sha256": hashlib.sha256(b"").hexdigest()}))

    file_hash = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    env = {
        **os.environ,
        "ROOT": str(root),
        "REPO_SUBPATH": "repo",
        "REPO_COMMIT": commit,
        "REPO_SUBMODULE_RECEIPT": str(submodules),
        "REPO_SUBMODULE_RECEIPT_HASH": file_hash(submodules),
        "SUBMITTER_SOURCE_HASH": file_hash(submitter),
        "IMAGE_DIGEST": "sha256:" + "9" * 64,
        "ARM": "sft",
        "EXPERIMENT": "R01",
        "INIT_MODEL_PATH": str(root / "models/sft"),
        "PAIRED_INIT_MANIFEST": str(paths["pair"]),
        "PAIRED_INIT_MANIFEST_HASH": file_hash(paths["pair"]),
        "CHECKPOINT_RECEIPT": str(paths["checkpoint"]),
        "CHECKPOINT_RECEIPT_HASH": file_hash(paths["checkpoint"]),
        "TRAIN_RECEIPT": str(paths["train"]),
        "TRAIN_RECEIPT_HASH": file_hash(paths["train"]),
        "MATH7_RECEIPT": str(paths["math7"]),
        "MATH7_RECEIPT_HASH": "0" * 64,
        "GRADER_RECEIPT": str(paths["grader"]),
        "GRADER_RECEIPT_HASH": file_hash(paths["grader"]),
        "JOB_TAG": "sft-I1-r20260727-test",
        "CELL_HASH": "1" * 64,
        "ATTEMPT_ID": "a1",
        "ALGORITHM_CONFIG_HASH": file_hash(frozen),
        "H20_PROFILE_PATH": str(paths["h20"]),
        "H20_PROFILE_HASH": file_hash(paths["h20"]),
        "H20_CALIBRATION_RECEIPT": str(paths["h20cal"]),
        "H20_CALIBRATION_RECEIPT_HASH": file_hash(paths["h20cal"]),
        "PATH_OVERRIDE_RECEIPT": str(paths["paths"]),
        "PATH_OVERRIDE_RECEIPT_HASH": file_hash(paths["paths"]),
        "RUN_MODE": "formal",
    }
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "platform/hope_rebuttal_rlvr/jupyter.sh")],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "Math-7 receipt hash mismatch" in result.stderr
