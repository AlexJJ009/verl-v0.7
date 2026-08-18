import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gate = load_script("code_wdl_stage1_reuse_gate", ROOT / "scripts/code_wdl_stage1_reuse_gate.py")
queue = load_script("code_wdl_acd0_queue", ROOT / "scripts/code_wdl_acd0_queue.py")


def row(*, source="HumanEval+", formatted=True, correct=True, finish="stop", score=None):
    return {
        "data_source": source,
        "format_contract_success": formatted,
        "acc": 1.0 if correct else 0.0,
        "score": (1.0 if correct else -1.0) if score is None else score,
        "response_finish_reason": finish,
        "response_eos_present": finish == "stop",
        "code_reward_dependency_error": 0,
        "code_reward_timeout": 0,
    }


def test_native_finish_reason_overrides_stale_reward_truncation_field():
    sample = row(formatted=False, correct=False, finish="length")
    sample.update({"has_eos": True, "truncated": False})
    summary = gate.summarize([sample])
    assert summary["micro"]["native_truncation_rate"] == 1.0
    assert summary["micro"]["format_failures_due_to_truncation_rate"] == 1.0


def test_signal_loss_is_reported_as_response_throughput_not_accuracy_causality():
    summary = gate.summarize(
        [row(formatted=True, correct=True), row(formatted=False, correct=False, finish="length")]
    )
    assert summary["micro"]["estimated_positive_signal_loss_fraction"] == 0.5
    assert summary["micro"]["expected_positive_responses_per_group_n8"] == 4.0


def test_manifest_freezes_beta0_acd0_p60_contract():
    manifest = yaml.safe_load(
        (ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/code_qwen3_1p7b_wdl_acd0_p60_beta0.yaml").read_text()
    )
    assert manifest["status"] == "admitted_for_launch"
    assert manifest["launch_allowed"] is True
    assert manifest["blockers"] == []
    assert manifest["training_contract"]["beta"] == 0.0
    assert manifest["training_contract"]["negative_loss_gradient_contribution"] == 0.0
    assert manifest["training_contract"]["final_step"] == 60
    assert manifest["training_contract"]["stage3"] == "omitted"
    runs = {item["id"]: item for item in manifest["runs"]}
    assert (runs["arm-c-mixture"]["fusion_lambda"], runs["arm-c-mixture"]["fusion_mode"]) == (0.8, "mixture")
    assert (
        runs["arm-d0-matched-scale-no-weak"]["fusion_lambda"],
        runs["arm-d0-matched-scale-no-weak"]["fusion_mode"],
    ) == (0.8, "strong_scaled")
    assert manifest["queue_order"] == [
        "arm-a-stage1-continuation",
        "arm-d0-matched-scale-no-weak",
        "arm-c-mixture",
    ]


def test_queue_refuses_implicit_checkpoint_resume(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    artifact_root = tmp_path / "artifacts"
    checkpoint_root.mkdir()
    run = {"id": "arm-c-mixture", "run_prefix": "CODE-WDL-ACD0-P60-ARM-C-QWEN3-1P7B"}
    (checkpoint_root / f"{run['run_prefix']}_123").mkdir()
    manifest = {"paths": {"checkpoint_root": str(checkpoint_root), "artifact_root": str(artifact_root)}}
    try:
        queue.require_clean_targets(manifest, run)
    except RuntimeError as exc:
        assert "refusing implicit overwrite/resume" in str(exc)
    else:
        raise AssertionError("checkpoint collision must fail closed")


def test_queue_can_select_only_d0_without_reordering_manifest():
    manifest = {
        "queue_order": [
            "arm-a-stage1-continuation",
            "arm-d0-matched-scale-no-weak",
            "arm-c-mixture",
        ],
        "runs": [
            {"id": "arm-a-stage1-continuation", "model_kind": "single"},
            {"id": "arm-d0-matched-scale-no-weak", "model_kind": "joint"},
            {"id": "arm-c-mixture", "model_kind": "joint"},
        ],
    }

    selected = queue.select_runs(manifest, "arm-d0-matched-scale-no-weak")

    assert [run["id"] for run in selected] == ["arm-d0-matched-scale-no-weak"]


def test_joint_arm_cache_path_is_short_run_unique_and_arm_specific(tmp_path):
    cache_root = tmp_path / "huggingface"
    d0 = {"id": "arm-d0-matched-scale-no-weak", "model_kind": "joint"}
    c = {"id": "arm-c-mixture", "model_kind": "joint"}

    d0_path = queue.joint_model_cache_path(d0, 1800000004, {"JOINT_MODEL_CACHE_ROOT": str(cache_root)})
    c_path = queue.joint_model_cache_path(c, 1800000004, {"JOINT_MODEL_CACHE_ROOT": str(cache_root)})

    assert d0_path.parent == cache_root
    assert d0_path.name == "code-acd0-d0-1800000004"
    assert c_path.name == "code-acd0-c-1800000004"
    assert len(d0_path.name.replace("-", "_hyphen_")) <= 180
    assert d0_path != c_path


def test_joint_wrapper_has_short_cache_fallback_for_direct_launches():
    wrapper = (
        ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_qwen3_1p7b_wdl_acd0_joint_common.sh"
    ).read_text()

    assert "JOINT_MODEL_CACHE_ROOT" in wrapper
    assert 'export MODEL_PATH="${JOINT_MODEL_CACHE_ROOT}/code-acd0-${arm_tag}-${run_timestamp}"' in wrapper


def test_reward_prefers_explicit_native_eos_over_length_fallback():
    from recipe.on_policy_wdl_sft.code_task.official_aligned_reward import compute_score_code_official_aligned

    response = "<think>short</think><answer>```python\nprint('OK')\n```</answer>"
    ground_truth = {"verification_method": "stdin_stdout_exec", "tests": [{"input": "", "output": "OK\n"}]}
    result = compute_score_code_official_aligned(
        "deepcoder_preview_train",
        response,
        ground_truth,
        extra_info={"valid_response_length": 128, "max_resp_len": 128, "response_eos_present": True},
    )
    assert result["has_eos"] is True
    assert result["truncated"] is False

    result = compute_score_code_official_aligned(
        "deepcoder_preview_train",
        response,
        ground_truth,
        extra_info={"valid_response_length": 100, "max_resp_len": 128, "response_eos_present": False},
    )
    assert result["score"] == -1.0
    assert result["has_eos"] is False
    assert result["truncated"] is True


def test_dapo_manager_injects_native_eos_and_lengths():
    from omegaconf import OmegaConf

    from verl import DataProto
    from verl.workers.reward_manager.dapo import DAPORewardManager

    seen = {}

    def compute_score(*, extra_info, **_):
        seen.update(extra_info)
        return {"score": -1.0, "acc": 0.0}

    class Tokenizer:
        eos_token_id = 99
        eos_token = "<eos>"

        @staticmethod
        def decode(ids, skip_special_tokens=True):
            return "decoded"

    manager = DAPORewardManager(
        tokenizer=Tokenizer(),
        num_examine=0,
        compute_score=compute_score,
        max_resp_len=3,
        overlong_buffer_cfg=OmegaConf.create({"enable": False, "len": 0, "penalty_factor": 0.0, "log": False}),
    )
    batch = DataProto.from_single_dict(
        {
            "prompts": torch.tensor([[11, 12]]),
            "responses": torch.tensor([[21, 22, 23]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "reward_model": np.array([{"ground_truth": {}}], dtype=object),
            "data_source": np.array(["HumanEval+"], dtype=object),
            "extra_info": np.array([{}], dtype=object),
        }
    )
    manager(batch, return_dict=True)
    assert seen["valid_response_length"] == 3
    assert seen["max_resp_len"] == 3
    assert seen["response_eos_present"] is False


@pytest.mark.asyncio
async def test_async_dapo_manager_injects_native_eos_and_lengths():
    from omegaconf import OmegaConf

    from verl import DataProto
    from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager as AsyncDAPORewardManager

    seen = {}

    async def compute_score(*, extra_info, **_):
        seen.update(extra_info)
        return {"score": -1.0, "acc": 0.0}

    class Tokenizer:
        eos_token_id = 99

        @staticmethod
        def decode(ids, skip_special_tokens=True):
            return "decoded"

    manager = AsyncDAPORewardManager(
        config=OmegaConf.create({"reward": {"timeout": 5, "reward_kwargs": {"max_resp_len": 3}}}),
        tokenizer=Tokenizer(),
        compute_score=compute_score,
    )
    batch = DataProto.from_single_dict(
        {
            "responses": torch.tensor([[21, 99, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
            "reward_model": np.array([{"ground_truth": {}}], dtype=object),
            "data_source": np.array(["HumanEval+"], dtype=object),
            "extra_info": np.array([{}], dtype=object),
        }
    )

    await manager.run_single(batch)

    assert seen["valid_response_length"] == 2
    assert seen["max_resp_len"] == 3
    assert seen["response_eos_present"] is True


def _terminal_manifest(tmp_path: Path) -> dict:
    return {
        "experiment_id": "code_qwen3_1p7b_wdl_acd0_p60_beta0",
        "paths": {"checkpoint_root": str(tmp_path / "checkpoints")},
        "training_contract": {"final_step": 60},
    }


def _terminal_env(tmp_path: Path) -> dict[str, str]:
    return {
        "TRAINING_RELEASE_GATE_STATE": str(tmp_path / "release_gate.jsonl"),
        "VERL_FILE_LOGGER_ROOT": str(tmp_path / "metrics"),
        "WANDB_PROJECT": "OnPolicyWDLSFT-CodeTask",
    }


def _gate_check(tmp_path: Path, run_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/training_result_release_gate.py"),
            "--state",
            str(tmp_path / "release_gate.jsonl"),
            "check",
            "--run-name",
            run_name,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_success_terminal_event_uses_exact_timestamped_name_and_passes_gate(tmp_path):
    manifest = _terminal_manifest(tmp_path)
    env = _terminal_env(tmp_path)
    run = {"id": "arm-c-mixture", "run_prefix": "CODE-WDL-ACD0-P60-ARM-C-QWEN3-1P7B"}
    run_name = "CODE-WDL-ACD0-P60-ARM-C-QWEN3-1P7B_1800000000"
    checkpoint = tmp_path / "checkpoints" / run_name
    (checkpoint / "global_step_60").mkdir(parents=True)
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("60\n")
    metrics = tmp_path / "metrics" / env["WANDB_PROJECT"] / f"{run_name}.jsonl"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(json.dumps({"step": 60, "data": {"training/global_step": 60}}) + "\n")

    queue.record_terminal_outcome(manifest, run, run_name, completed=True, env=env)

    event = json.loads((tmp_path / "release_gate.jsonl").read_text().splitlines()[-1])
    assert event["run_name"] == run_name
    assert event["status"] == "success_complete"
    assert event["checkpoint"] == str(checkpoint)
    assert event["metrics"] == str(metrics)
    assert event["observed_step"] == event["final_step"] == 60
    assert _gate_check(tmp_path, run_name).returncode == 0


def test_failed_terminal_event_uses_exact_timestamped_name_and_blocks_gate(tmp_path):
    manifest = _terminal_manifest(tmp_path)
    env = _terminal_env(tmp_path)
    run = {"id": "arm-a-stage1-continuation", "run_prefix": "CODE-WDL-ACD0-P60-ARM-A-QWEN3-1P7B"}
    run_name = "CODE-WDL-ACD0-P60-ARM-A-QWEN3-1P7B_1800000001"

    queue.record_terminal_outcome(manifest, run, run_name, completed=False, env=env, notes="wrapper failed")

    event = json.loads((tmp_path / "release_gate.jsonl").read_text().splitlines()[-1])
    assert event["run_name"] == run_name
    assert event["status"] == "failed"
    assert _gate_check(tmp_path, run_name).returncode == 1


@pytest.mark.parametrize("missing", ["checkpoint", "metrics"])
def test_missing_final_evidence_cannot_record_success_complete(tmp_path, missing):
    manifest = _terminal_manifest(tmp_path)
    env = _terminal_env(tmp_path)
    run = {"id": "arm-d0-matched-scale-no-weak", "run_prefix": "CODE-WDL-ACD0-P60-ARM-D0-QWEN3-1P7B"}
    run_name = f"CODE-WDL-ACD0-P60-ARM-D0-QWEN3-1P7B_180000000{2 if missing == 'checkpoint' else 3}"
    checkpoint = tmp_path / "checkpoints" / run_name
    metrics = tmp_path / "metrics" / env["WANDB_PROJECT"] / f"{run_name}.jsonl"
    if missing != "checkpoint":
        (checkpoint / "global_step_60").mkdir(parents=True)
        (checkpoint / "latest_checkpointed_iteration.txt").write_text("60\n")
    if missing != "metrics":
        metrics.parent.mkdir(parents=True)
        metrics.write_text(json.dumps({"step": 60}) + "\n")

    with pytest.raises(RuntimeError, match="terminal success evidence incomplete"):
        queue.record_terminal_outcome(manifest, run, run_name, completed=True, env=env)

    event = json.loads((tmp_path / "release_gate.jsonl").read_text().splitlines()[-1])
    assert event["status"] == "failed"
    assert _gate_check(tmp_path, run_name).returncode == 1


@pytest.mark.parametrize("wrapper_fails", [False, True])
def test_queue_main_wires_exact_run_name_to_terminal_outcome(tmp_path, monkeypatch, wrapper_fails):
    run = {
        "id": "arm-c-mixture",
        "run_prefix": "CODE-WDL-ACD0-P60-ARM-C-QWEN3-1P7B",
        "model_kind": "joint",
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "queue_order": [run["id"]],
                "runs": [run],
                "paths": {
                    "model1": str(tmp_path / "model1"),
                    "model2": str(tmp_path / "model2"),
                    "model2_provenance": str(tmp_path / "provenance.json"),
                    "train_file": str(tmp_path / "train.parquet"),
                    "checkpoint_root": str(tmp_path / "checkpoints"),
                    "artifact_root": str(tmp_path / "artifacts"),
                },
            }
        )
    )
    recorded = []

    monkeypatch.setattr(queue, "validate_manifest", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(queue, "require_receipts", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(queue, "require_clean_targets", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(queue.time, "time", lambda: 1800000004)
    monkeypatch.setattr(queue, "record_terminal_outcome", lambda *args, **kwargs: recorded.append((args, kwargs)))
    monkeypatch.setattr(queue, "WRAPPERS", {run["id"]: tmp_path / "wrapper.sh"})
    monkeypatch.setenv("TMUX", "test")
    monkeypatch.setattr(queue.sys, "argv", ["code_wdl_acd0_queue.py", "--manifest", str(manifest_path)])

    def fake_run(*_args, **kwargs):
        assert kwargs["env"]["WANDB_RUN_NAME"] == f"{run['run_prefix']}_1800000004"
        assert Path(kwargs["env"]["MODEL_PATH"]).name == "code-acd0-c-1800000004"
        assert kwargs["env"]["LOG_DIR"] == str(tmp_path / "artifacts" / "logs")
        assert kwargs["env"]["VERL_FILE_LOGGER_ROOT"] == str(
            tmp_path / "artifacts" / "logs" / "metrics"
        )
        if wrapper_fails:
            raise subprocess.CalledProcessError(1, "wrapper")
        return subprocess.CompletedProcess("wrapper", 0)

    monkeypatch.setattr(queue.subprocess, "run", fake_run)
    if wrapper_fails:
        with pytest.raises(subprocess.CalledProcessError):
            queue.main()
    else:
        assert queue.main() == 0

    assert len(recorded) == 1
    args, kwargs = recorded[0]
    assert args[2] == f"{run['run_prefix']}_1800000004"
    assert kwargs["completed"] is (not wrapper_fails)
    assert kwargs["env"]["WANDB_RUN_NAME"] == args[2]
