from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/code_qwen3_1p7b_stage123_cotmask_v3.yaml"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_code_stage123_manifest_is_step20_and_launch_ready_after_gpu_probe():
    manifest = yaml.safe_load(MANIFEST.read_text())
    assert manifest["task"] == "code"
    assert manifest["launch_allowed"] is True
    assert manifest["status"] == "launch_ready_gpu_probe_passed"
    assert manifest["resources"]["rollout_gpu_memory_utilization"] == 0.35
    assert "author_signature_v2_step20" in manifest["paths"]["artifact_root"]
    assert manifest["paths"]["state_root"].endswith("/state")
    assert manifest["paths"]["event_log"].endswith("/state/events.jsonl")
    assert "author_signature_v2_gpu_utilization_probe/step20" in manifest["paths"]["gpu_probe_report"]
    assert manifest["model1_selection_policy"] == {
        "selected_step": 20,
        "allow_below_format_threshold": True,
        "rationale": "user_authorized_best_observed_format_checkpoint_at_step20",
    }
    assert "author_signature_v2" in manifest["paths"]["source_train_file"]
    assert "author_signature_v2" in manifest["paths"]["model1_selection"]
    assert "author_signature_v2" in manifest["paths"]["dataset_receipt"]
    assert len(manifest["runs"]) == 16
    assert {run["beta"] for run in manifest["runs"]} == {0.0, 0.1}
    assert manifest["validation"] == {
        "datasets": ["HumanEval+", "MBPP+", "LiveCodeBench"],
        "n": 3,
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": -1,
        "do_sample": True,
        "primary_metric": "val-core/code3_macro/acc/mean@3",
        "joint_primary_metric": "val-core/model2/code3_macro/acc/mean@3",
    }
    assert manifest["training"] == {
        "learning_rate": 1e-6,
        "warmup_steps": 0,
        "train_prompt_batch_size": 64,
        "rollout_n": 8,
        "data_shuffle": False,
        "validation_interval_steps": 5,
    }
    assert "format_cold_start_fraction/qwen3-1p7b" not in MANIFEST.read_text()


def test_code_model1_selection_exists_after_new_cold_start_finishes():
    manifest = yaml.safe_load(MANIFEST.read_text())
    assert manifest["launch_allowed"] is True
    selection = json.loads(Path(manifest["paths"]["model1_selection"]).read_text())
    assert selection["selected_step"] == 20
    assert selection["candidate"]["step"] == 20
    assert selection["identity"]["model_path"].endswith("/candidates/step_20")
    model_config = json.loads((Path(selection["identity"]["model_path"]) / "config.json").read_text())
    assert model_config["model_type"] == "qwen3"
    assert model_config["transformers_version"] == "4.51.0"


def test_code_stage123_new_dataset_receipt_must_be_generated():
    manifest = yaml.safe_load(MANIFEST.read_text())
    module = load_module(
        ROOT / "recipe/on_policy_wdl_sft/code_task/prepare_qwen3_1p7b_code_stage123_data.py"
    )
    receipt_path = Path(manifest["paths"]["dataset_receipt"])
    if receipt_path.exists():
        receipt = module.verify_receipt(
            receipt_path.parent,
            expected_source=Path(manifest["paths"]["source_train_file"]),
            expected_seed=manifest["seed"],
        )
        assert receipt["cold_start_steps"] == 20
        assert receipt["cold_start_batch_size"] == 64
        assert receipt["cold_start_rows_consumed"] == 1280
    else:
        with __import__("pytest").raises(FileNotFoundError):
            module.verify_receipt(
                receipt_path.parent,
                expected_source=Path(manifest["paths"]["source_train_file"]),
                expected_seed=manifest["seed"],
            )


def test_code_stage123_excludes_only_rows_consumed_by_step20_cold_start():
    module = load_module(
        ROOT / "recipe/on_policy_wdl_sft/code_task/prepare_qwen3_1p7b_code_stage123_data.py"
    )
    manifest = yaml.safe_load(MANIFEST.read_text())
    receipt = module.verify_receipt(
        Path(manifest["paths"]["dataset_receipt"]).parent,
        expected_source=Path(manifest["paths"]["source_train_file"]),
        expected_seed=manifest["seed"],
    )
    consumed = module.consumed_cold_start_source_indices(
        Path(receipt["cold_start_file"]),
        steps=20,
        batch_size=64,
    )
    assert len(consumed) == 1280
    stage_indices = set()
    for name in ("stage1", "stage2", "stage3"):
        import pandas as pd

        indices = set(
            int(value)
            for value in pd.read_parquet(receipt["shards"][name]["path"])["stage123_source_index"]
        )
        assert not consumed.intersection(indices)
        assert not stage_indices.intersection(indices)
        stage_indices.update(indices)
    assert len(stage_indices) == 6400


def test_code_stage123_launch_surface_uses_new_manifest_and_monitor():
    queue = (ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_qwen3_1p7b_stage123_cotmask_v3_queue.sh").read_text()
    monitor = (ROOT / "recipe/on_policy_wdl_sft/code_task/monitor_code_qwen3_1p7b_stage123_cotmask_v3.sh").read_text()
    profile = (ROOT / "recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh").read_text()
    gate = (ROOT / "recipe/on_policy_wdl_sft/code_task/stage123_manifest_gate.sh").read_text()
    assert "code_qwen3_1p7b_stage123_cotmask_v3.yaml" in queue
    assert "math_stage123_queue.py" in queue
    assert "code_stage123_monitor.py" in monitor
    assert '["paths"]["event_log"]' in queue
    assert '["paths"]["event_log"]' in monitor
    assert "CODE_STAGE123_ADMISSION" in queue
    assert 'CODE_STAGE123_MANIFEST_SHA256="$manifest_sha"' in queue
    assert 'CODE_STAGE123_MODEL1_SELECTION_SHA256="$model1_selection_sha"' in queue
    assert 'CODE_STAGE123_DATASET_RECEIPT_SHA256="$dataset_receipt_sha"' in queue
    assert "code_stage123_admission.py" in gate
    for setting in (
        "ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}",
        "ACTOR_CALCULATE_ENTROPY=${ACTOR_CALCULATE_ENTROPY:-False}",
        "CALCULATE_ENTROPY=${CALCULATE_ENTROPY:-False}",
    ):
        assert setting in profile
    single_common = (ROOT / "recipe/on_policy_wdl_sft/ablation_single_model/_common_ablation.sh").read_text()
    assert "offload=${FSDP_OFFLOAD:-False}" in single_common
    assert "optimizer_offload=${FSDP_OPTIMIZER_OFFLOAD:-${offload}}" in single_common
    assert "ref_offload=${REF_FSDP_OFFLOAD:-${offload}}" in single_common
    queue_module = (ROOT / "scripts/math_stage123_queue.py").read_text()
    assert '"DATA_SEED": str(manifest["seed"])' in queue_module
    assert '"DATA_SHUFFLE": "False"' in queue_module
    assert 'env["STAGE2_SUBMODEL"] = submodel' in queue_module
    assert 'env["STAGE2_PROVENANCE_FILE"]' in queue_module
    assert '"CODE_TRAIN_FILE": env["TRAIN_FILE"]' in queue_module
    assert '"EXPECTED_MODEL1_PATH"' in queue_module
    assert '"STAGE1_MODEL2_PROVENANCE_FILE"' in queue_module
    assert '"CODE_VAL_FILES": str(code_validation_files)' in queue_module
    assert '"TEST_FILES": str(code_validation_files)' in queue_module
    assert 'str(manifest["resources"]["rollout_gpu_memory_utilization"])' in queue_module


def test_code_stage123_dry_run_renders_full_step20_matrix_without_events(tmp_path):
    event_log = tmp_path / "events.jsonl"
    container_root = Path("/workspace/verl") if Path("/workspace/verl/scripts/math_stage123_queue.py").is_file() else ROOT
    result = subprocess.run(
        [
            __import__("sys").executable,
            str(container_root / "scripts/math_stage123_queue.py"),
            "--manifest",
            str(container_root / "recipe/on_policy_wdl_sft/experiment_manifest/code_qwen3_1p7b_stage123_cotmask_v3.yaml"),
            "--dry-run",
        ],
        cwd=container_root,
        env={
            **__import__("os").environ,
            "STAGE123_EVENT_LOG": str(event_log),
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(commands) == 16
    assert all(
        item["environment"]["EXPECTED_MODEL1_PATH"].endswith("/candidates/step_20")
        for item in commands
    )
    assert all(item["environment"]["LR"] == "1e-6" for item in commands)
    assert all(item["environment"]["LR_WARMUP_STEPS"] == "0" for item in commands)
    assert not event_log.exists()


def test_gpu_probe_has_separate_non_launchable_admission_path():
    gate = (ROOT / "recipe/on_policy_wdl_sft/code_task/stage123_manifest_gate.sh").read_text()
    probe_phase = (ROOT / "scripts/code_stage123_probe_phase.py").read_text()
    assert "CODE_STAGE123_GPU_PROBE_ADMITTED" in gate
    assert "blocked_pending_gpu_utilization_probe" in gate
    assert "manifest.get(\"launch_allowed\") is not False" in gate
    assert "/data-1/tmp/verl_agent_scratch/code_stage123_gpu_utilization_probe" in gate
    assert '"CODE_STAGE123_GPU_PROBE_ADMITTED": "1"' in probe_phase


def test_code_stage2_and_stage3_handoff_contracts_are_compatible():
    stage2 = (ROOT / "recipe/on_policy_wdl_sft/code_task/run_s2_code_qwen3_1p7b_stage123_common.sh").read_text()
    stage2_base = (ROOT / "recipe/on_policy_wdl_sft/code_task/run_s2_code_model2_rollout_common.sh").read_text()
    stage3 = (ROOT / "recipe/on_policy_wdl_sft/code_task/run_s3_code_qwen3_1p7b_stage123_common.sh").read_text()
    queue = (ROOT / "scripts/math_stage123_queue.py").read_text()
    assert "run_s2_code_model2_rollout_common.sh" in stage2
    assert "code_stage123_macro_overrides" in stage2
    assert "ALLOW_EXTERNAL_MODEL2" in stage2_base
    assert 'provenance.get("identity", {}).get("model_path")' in stage2_base
    assert 'source.get(f"extracted_{submodel}")' in stage3
    assert 'provenance["source"]' in queue
    assert '"extracted_model1"' in queue
    assert '"extracted_model2"' in queue


def test_code_stage2_dry_run_requires_new_model1_selection(tmp_path):
    manifest = yaml.safe_load(MANIFEST.read_text())
    selection_path = Path(manifest["paths"]["model1_selection"])
    assert manifest["launch_allowed"] is True
    assert selection_path.exists()
