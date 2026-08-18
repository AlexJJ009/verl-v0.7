import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pandas as pd
import pytest
import yaml

from recipe.joint_training.custom_reward_function_latex_verify import compute_format_telemetry
from recipe.joint_training.offline_eval import compute_shared_metrics
from recipe.joint_training.custom_reward_function_latex_verify import compute_score_latex_verify
from verl.trainer.ppo.ray_trainer import _add_validation_macro_average


ROOT = Path(__file__).resolve().parents[2]


def load_data_module():
    path = ROOT / "recipe/on_policy_wdl_sft/math_task/prepare_qwen3_1p7b_math_stage123_data.py"
    spec = importlib.util.spec_from_file_location("math_stage123_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_math7_validation_module():
    path = ROOT / "recipe/on_policy_wdl_sft/math_task/prepare_math7_validation_data.py"
    spec = importlib.util.spec_from_file_location("math7_validation_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_stage123_queue_module():
    path = ROOT / "scripts/math_stage123_queue.py"
    spec = importlib.util.spec_from_file_location("math_stage123_queue", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_format_telemetry_requires_one_ordered_pair_per_tag():
    assert compute_format_telemetry("<think>x</think><answer>\\boxed{1}</answer>") == {
        "think_nonempty": True,
        "think_complete": True,
        "answer_complete": True,
        "format_ordered": True,
    }


def test_complete_format_contract_is_the_intersection():
    result = compute_score_latex_verify(
        "test",
        "<think>work</think><answer>\\boxed{1}</answer>",
        "1",
        {"valid_response_length": 10, "max_resp_len": 20},
    )
    assert result["format_contract_success"] is True
    truncated = compute_score_latex_verify(
        "test",
        "<think>work</think><answer>\\boxed{1}</answer>",
        "1",
        {"valid_response_length": 20, "max_resp_len": 20},
    )
    assert truncated["format_contract_success"] is False
    assert compute_format_telemetry("<think>x</think><answer>missing close") == {
        "think_nonempty": True,
        "think_complete": True,
        "answer_complete": False,
        "format_ordered": False,
    }


def test_offline_format_metrics_are_response_rates():
    metrics = compute_shared_metrics(
        [
            {
                "results": [
                    {"acc": True, "think_complete": True, "answer_complete": True, "boxed_extraction_success": True, "reward_grader_success": True, "format_contract_success": True, "has_eos": True, "truncated": False},
                    {"acc": False, "think_complete": True, "answer_complete": False, "boxed_extraction_success": False, "reward_grader_success": True, "format_contract_success": False, "has_eos": False, "truncated": True},
                    {"acc": False, "think_complete": False, "answer_complete": False, "boxed_extraction_success": False, "reward_grader_success": False, "format_contract_success": False, "has_eos": True, "truncated": False},
                ]
            }
        ],
        3,
    )
    assert metrics["mean@3"] == pytest.approx(1 / 3)
    assert metrics["think_complete_rate"] == pytest.approx(2 / 3)
    assert metrics["boxed_extraction_success_rate"] == pytest.approx(1 / 3)
    assert metrics["format_contract_success_rate"] == pytest.approx(1 / 3)
    assert metrics["truncation_rate"] == pytest.approx(1 / 3)


def test_math7_macro_average_uses_equal_dataset_weighting():
    sources = ["a", "b", "c"]
    metrics = {f"val-core/{source}/acc/mean@3": value for source, value in zip(sources, [0.3, 0.6, 0.9])}
    config = {
        "validation_macro_average_sources": sources,
        "validation_macro_average_name": "math7_macro",
        "validation_macro_average_metric": "acc/mean@3",
    }
    _add_validation_macro_average(metrics, config)
    assert metrics["val-core/math7_macro/acc/mean@3"] == pytest.approx(0.6)


def test_disjoint_split_receipt_and_control_order(tmp_path):
    module = load_data_module()
    source = tmp_path / "source.parquet"
    output = tmp_path / "output"
    pd.DataFrame({"row": list(range(7500))}).to_parquet(source, index=False)
    old_argv = module.parse_args
    module.parse_args = lambda: type("Args", (), {"source": source, "output_root": output, "seed": 20260719, "verify_only": False})()
    try:
        module.main()
    finally:
        module.parse_args = old_argv
    receipt = module.verify_receipt(output)
    assert [receipt["shards"][name]["rows"] for name in ("cold_start", "stage1", "stage2", "stage3")] == [1100, 2560, 1280, 2560]
    stage2 = pd.read_parquet(output / "stage2.parquet")["stage123_source_index"].tolist()
    stage3 = pd.read_parquet(output / "stage3.parquet")["stage123_source_index"].tolist()
    control = pd.read_parquet(output / "stage1_control_stage2_then_stage3.parquet")["stage123_source_index"].tolist()
    assert control == stage2 + stage3
    assert len(control) == 60 * 64


def test_math7_validation_schema_alignment_supports_real_concatenation(tmp_path):
    module = load_math7_validation_module()
    sources = []
    for index, extra_info in enumerate(
        [
            {"index": "a", "level": 2},
            {"index": 3, "answer": "x"},
            {"index": 4, "options": ["A", "B"]},
        ]
    ):
        source = tmp_path / f"source_{index}.parquet"
        pd.DataFrame(
            {
                "data_source": [f"source-{index}"],
                "ability": ["Math"],
                "reward_model": [{"ground_truth": "1", "style": "rule"}],
                "prompt": [[{"role": "user", "content": "1+0?"}]],
                "split": ["test"],
                "extra_info": [extra_info],
            }
        ).to_parquet(source, index=False)
        sources.append(source)
    normalized_paths = []
    for source in sources:
        output = tmp_path / module.normalized_name(source)
        module.normalize_frame(pd.read_parquet(source)).to_parquet(output, index=False)
        normalized_paths.append(str(output))
    loaded = [module.datasets.load_dataset("parquet", data_files=path, split="train") for path in normalized_paths]
    combined = module.datasets.concatenate_datasets(loaded)
    assert len(combined) == 3
    assert all(isinstance(value, str) for value in combined["extra_info"])


def test_math7_validation_receipt_cannot_escape_output_root(tmp_path):
    module = load_math7_validation_module()
    source = tmp_path / "source.parquet"
    pd.DataFrame(
        {
            "data_source": ["source"],
            "ability": ["Math"],
            "reward_model": [{"ground_truth": "1", "style": "rule"}],
            "prompt": [[{"role": "user", "content": "1+0?"}]],
            "split": ["test"],
            "extra_info": [{"index": 1}],
        }
    ).to_parquet(source, index=False)
    output_root = tmp_path / "output"
    output_root.mkdir()
    escaped_output = tmp_path / module.normalized_name(source)
    module.normalize_frame(pd.read_parquet(source)).to_parquet(escaped_output, index=False)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "schema_version": 1,
        "sources": [str(source)],
        "outputs": [
            {
                "source": str(source),
                "source_sha256": digest(source),
                "path": str(escaped_output),
                "sha256": digest(escaped_output),
                "rows": 1,
            }
        ],
        "total_rows": 1,
    }
    (output_root / "dataset_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="output path mismatch"):
        module.verify_receipt(output_root, sources=(source,))


def test_math7_resource_profile_rejects_validation_file_override():
    profile = ROOT / "recipe/on_policy_wdl_sft/math_task/qwen3_1p7b_math_stage123_resource_profile.sh"
    result = subprocess.run(
        ["bash", "-c", f"MATH7_VAL_FILES=stale source {profile!s}"],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "must match the verified Math-7 validation root" in result.stderr


def test_manifests_freeze_full_math7_and_block_invalidated_launches():
    manifest_root = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest"
    cold = yaml.safe_load((manifest_root / "math_qwen3_1p7b_cold_start_cotmask_v3.yaml").read_text())
    stage = yaml.safe_load((manifest_root / "math_qwen3_1p7b_stage123_cotmask_v3.yaml").read_text())
    assert cold["training"]["step_interval"] == 5
    assert cold["paths"]["source_train_file"] == stage["paths"]["source_train_file"]
    assert len(cold["validation"]["datasets"]) == 7
    assert cold["validation"]["n"] == 1
    assert cold["admission_thresholds"]["format_contract_success_rate"] == 0.95
    assert cold["execution"]["launch_allowed"] is False
    assert cold["execution"]["requires_whole_message_loss_mask"] is True
    assert "loss_mask_preflight_receipt" in cold["paths"]
    assert cold["execution"]["auto_select_first_passing_checkpoint"] is True
    assert stage["launch_allowed"] is False
    assert stage["validation"]["n"] == 3
    assert stage["validation"]["primary_metric"] == "val-core/math7_macro/acc/mean@3"
    assert len(stage["runs"]) == 16
    assert {run["beta"] for run in stage["runs"]} == {0.0, 0.1}

    invalidated = [
        manifest_root / "math_qwen3_1p7b_cold_start.yaml",
        manifest_root / "math_qwen3_1p7b_cold_start_lr5e6_v2.yaml",
        manifest_root / "math_qwen3_1p7b_stage123_lr5e6_v2.yaml",
    ]
    for path in invalidated:
        manifest = yaml.safe_load(path.read_text())
        launch_allowed = manifest.get("launch_allowed", manifest.get("execution", {}).get("launch_allowed"))
        assert "invalidated" in manifest["status"]
        assert launch_allowed is False


def test_math_queue_defaults_only_to_cotmask_v3_manifests():
    cold_queue = (ROOT / "recipe/on_policy_wdl_sft/math_task/run_math_qwen3_1p7b_cold_start_queue.sh").read_text()
    stage_queue = (ROOT / "recipe/on_policy_wdl_sft/math_task/run_math_qwen3_1p7b_stage123_queue.sh").read_text()
    cold_python = (ROOT / "scripts/math_cold_start_queue.py").read_text()
    stage_python = (ROOT / "scripts/math_stage123_queue.py").read_text()
    assert "math_qwen3_1p7b_cold_start_cotmask_v3.yaml" in cold_queue
    assert "math_qwen3_1p7b_stage123_cotmask_v3.yaml" in stage_queue
    assert "math_qwen3_1p7b_cold_start_cotmask_v3.yaml" in cold_python
    assert "math_qwen3_1p7b_stage123_cotmask_v3.yaml" in stage_python
    assert "prepare_math7_validation_data.py" in stage_queue
    resource_profile = (ROOT / "recipe/on_policy_wdl_sft/math_task/qwen3_1p7b_math_stage123_resource_profile.sh").read_text()
    assert "qwen3_1p7b_math7_validation_v1" in resource_profile
    assert "_schema_aligned.parquet" in resource_profile


def test_math_stage123_learning_rate_matches_code_stage123():
    math_root = ROOT / "recipe/on_policy_wdl_sft/math_task"
    code_root = ROOT / "recipe/on_policy_wdl_sft/code_task"
    pairs = [
        ("run_s1_math_qwen3_1p7b_stage123_common.sh", "run_s1_code_qwen3_1p7b_stage123_common.sh"),
        ("run_s2_math_qwen3_1p7b_stage123_common.sh", "run_s2_code_qwen3_1p7b_stage123_common.sh"),
        ("run_s3_math_qwen3_1p7b_stage123_common.sh", "run_s3_code_qwen3_1p7b_stage123_common.sh"),
    ]
    for math_name, code_name in pairs:
        math_script = (math_root / math_name).read_text()
        code_script = (code_root / code_name).read_text()
        for setting in ("export LR=${LR:-1e-6}", "export LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-0}"):
            assert setting in math_script
            assert setting in code_script
    queue = (ROOT / "scripts/math_stage123_queue.py").read_text()
    assert '"LR": "1e-6"' in queue
    assert '"LR_WARMUP_STEPS": "0"' in queue
    assert '"ROLLOUT_GPU_MEMORY_UTILIZATION": str(manifest["resources"]["rollout_gpu_memory_utilization"])' in queue
    assert '"ACTOR_CALCULATE_ENTROPY": "False"' in queue
    assert '"CALCULATE_ENTROPY": "False"' in queue


def test_math_stage123_stage2_cache_paths_are_short_and_run_unique():
    module = load_stage123_queue_module()
    artifact_root = Path(
        "/data-2/model_weights/math_task/qwen3_1p7b_stage123_cotmask_v3/launches/20260720T091917Z/artifacts"
    )
    paths = {
        module.stage2_joint_cache_path(artifact_root, run_id)
        for run_id in ("b0-stage2-nokl", "b0-stage2-m2kl", "b01-stage2-nokl", "b01-stage2-m2kl")
    }
    assert len(paths) == 4
    for path in paths:
        assert path.parent == Path("/data-1/.cache/huggingface")
        assert len(path.name.replace("-", "_hyphen_")) <= 180


def test_math_stage123_continuation_requires_matching_completed_provenance(tmp_path: Path):
    module = load_stage123_queue_module()
    runs = [
        {"id": "stage1", "phase": "stage1", "beta": 0.0, "train_shard": "stage1", "final_step": 40},
        {
            "id": "stage2",
            "phase": "stage2",
            "beta": 0.0,
            "source_run": "stage1",
            "train_shard": "stage2",
            "final_step": 20,
            "kl": "nokl",
        },
    ]
    output_model = tmp_path / "stage1-model"
    output_model.mkdir()
    provenance_dir = tmp_path / "stage1"
    provenance_dir.mkdir()
    (provenance_dir / "provenance.json").write_text(
        json.dumps({"schema_version": 1, "run": runs[0], "outputs": {"model": str(output_model)}})
    )

    outputs, remaining = module.continuation_state({"runs": runs}, tmp_path, "stage2")

    assert outputs == {"stage1": {"model": str(output_model)}}
    assert remaining == [runs[1]]


def test_math_stage123_checkpoint_selection_ignores_incomplete_retry_root(tmp_path, monkeypatch):
    module = load_stage123_queue_module()
    checkpoint_root = tmp_path / "checkpoints"
    prefix = "MATH-B01_STAGE2_M2KL-QWEN3-1P7B-V1"
    incomplete = checkpoint_root / f"{prefix}_100"
    complete = checkpoint_root / f"{prefix}_200"
    incomplete_actor = incomplete / "global_step_20" / "actor"
    incomplete_actor.mkdir(parents=True)
    (incomplete_actor / "fsdp_config.json").write_text(json.dumps({"world_size": 2}))
    (incomplete_actor / "huggingface").mkdir()
    (incomplete_actor / "huggingface/config.json").write_text("{}")
    (incomplete_actor / "model_world_size_2_rank_0.pt").touch()
    actor = complete / "global_step_20" / "actor"
    actor.mkdir(parents=True)
    (actor / "fsdp_config.json").write_text(json.dumps({"world_size": 2}))
    (actor / "huggingface").mkdir()
    (actor / "huggingface/config.json").write_text("{}")
    for rank in range(2):
        (actor / f"model_world_size_2_rank_{rank}.pt").touch()
    started_at = min(incomplete.stat().st_mtime, complete.stat().st_mtime)
    original_path = module.Path

    def redirected_path(value):
        if value == "/data-1/checkpoints":
            return checkpoint_root
        return original_path(value)

    monkeypatch.setattr(module, "Path", redirected_path)

    assert module.checkpoint_after(prefix, started_at, 20) == actor


def test_math_stage123_retries_only_vllm_tcpstore_port_collisions(tmp_path, monkeypatch):
    module = load_stage123_queue_module()
    state = tmp_path / "attempted"
    monkeypatch.setenv("MATH_RUN_ATTEMPT_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("MATH_TRANSIENT_PORT_RETRIES", "1")
    monkeypatch.setenv("MATH_TRANSIENT_PORT_RETRY_DELAY_SEC", "0")
    command = [
        "bash",
        "-c",
        (
            f"if [ ! -e {state!s} ]; then "
            f"touch {state!s}; "
            "echo 'vLLMHttpServer torch.distributed.DistNetworkError TCPStore EADDRINUSE address already in use'; "
            "exit 1; "
            "fi; echo success"
        ),
    ]

    module.execute(command, dict(os.environ), False, "stage2-m2kl")

    assert (tmp_path / "logs/stage2-m2kl.attempt-1.log").is_file()
    assert "success" in (tmp_path / "logs/stage2-m2kl.attempt-2.log").read_text()


def test_math_stage123_does_not_retry_unrelated_failures(tmp_path, monkeypatch):
    module = load_stage123_queue_module()
    monkeypatch.setenv("MATH_RUN_ATTEMPT_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("MATH_TRANSIENT_PORT_RETRIES", "2")
    monkeypatch.setenv("MATH_TRANSIENT_PORT_RETRY_DELAY_SEC", "0")

    with pytest.raises(subprocess.CalledProcessError):
        module.execute(["bash", "-c", "echo CUDA OOM; exit 1"], dict(os.environ), False, "stage2-m2kl")

    assert (tmp_path / "logs/stage2-m2kl.attempt-1.log").is_file()
    assert not (tmp_path / "logs/stage2-m2kl.attempt-2.log").exists()


def test_math_stage123_queue_passes_explicit_start_run():
    queue = (ROOT / "recipe/on_policy_wdl_sft/math_task/run_math_qwen3_1p7b_stage123_queue.sh").read_text()
    assert 'args+=(--start-run "$MATH_STAGE123_START_RUN")' in queue


def test_math_and_code_stage123_disable_entropy_and_use_admitted_rollout_memory():
    math_profile = (ROOT / "recipe/on_policy_wdl_sft/math_task/qwen3_1p7b_math_stage123_resource_profile.sh").read_text()
    code_profile = (ROOT / "recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh").read_text()
    assert "ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}" in math_profile
    assert "ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}" in code_profile
    for profile in (math_profile, code_profile):
        assert "ACTOR_CALCULATE_ENTROPY=${ACTOR_CALCULATE_ENTROPY:-False}" in profile
        assert "CALCULATE_ENTROPY=${CALCULATE_ENTROPY:-False}" in profile
    assert '"$ACTOR_CALCULATE_ENTROPY" = False' in code_profile
    assert '"$CALCULATE_ENTROPY" = False' in code_profile

    code_stage1 = (ROOT / "recipe/on_policy_wdl_sft/code_task/run_s1_code_kodcode_qwen3_1p7b_instruct_ctx8k_beta_0.sh").read_text()
    code_stage2 = (ROOT / "recipe/on_policy_wdl_sft/code_task/run_s2_code_kodcode_qwen3_1p7b_instruct_ctx8k_p40_common.sh").read_text()
    assert "ACTOR_CALCULATE_ENTROPY=${ACTOR_CALCULATE_ENTROPY:-False}" in code_stage1
    assert "ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}" in code_stage1
    assert "CALCULATE_ENTROPY=${CALCULATE_ENTROPY:-False}" in code_stage2
    assert "ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}" in code_stage2
