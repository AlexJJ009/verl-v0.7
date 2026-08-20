from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_environment_distinguishes_no_kl_and_model2_kl():
    module = load_module("stage123_matrix_memory_probe", ROOT / "scripts/run_stage123_matrix_memory_probe.py")
    base = {
        "phase": "stage2",
        "train_file": "/train.parquet",
        "source": {
            "checkpoint_root": "/checkpoints/source",
            "model1_path": "/models/frac25",
            "model1_config_sha256": "a" * 64,
            "model1_tokenizer_config_sha256": "b" * 64,
            "model1_chat_template_sha256": "c" * 64,
            "model1_provenance_path": "/models/frac25/source.json",
            "model1_provenance_sha256": "d" * 64,
            "model2_path": "/models/model2",
            "run_prefix": "SOURCE",
            "handoff_step": 40,
        },
    }
    profile = {
        "sha256": "p" * 64,
        "rollout_gpu_memory_utilization": 0.4,
        "rollout_max_num_batched_tokens": 32768,
        "rollout_free_cache_engine": False,
        "rollout_enable_sleep_mode": False,
        "ref_fsdp_offload": True,
        "actor_param_offload": True,
        "actor_optimizer_offload": True,
        "ref_log_prob_micro_batch_size": 1,
        "ref_log_prob_max_token_len_per_gpu": 9216,
    }
    no_kl = module.run_environment(
        {
            **base,
            "submodel_kl": {
                "enabled": False,
                "model1_enabled": False,
                "model1_coef": 0.0,
                "model2_enabled": False,
                "model2_coef": 0.0,
            },
        },
        profile,
    )
    model2_kl = module.run_environment(
        {
            **base,
            "submodel_kl": {
                "enabled": True,
                "model1_enabled": False,
                "model1_coef": 0.0,
                "model2_enabled": True,
                "model2_coef": 0.01,
                "model2_ref_path": "/models/ref",
            },
        },
        profile,
    )
    assert no_kl["SUBMODEL_KL_MODEL2_ENABLED"] == "false"
    assert model2_kl["SUBMODEL_KL_MODEL2_ENABLED"] == "true"
    assert model2_kl["SUBMODEL_KL_MODEL2_REF_PATH"] == "/models/ref"
    assert model2_kl["STAGE123_EXPECTED_VAL_N"] == "3"
    assert no_kl["BASE_MODEL_PATH"] == "/models/frac25"
    assert no_kl["ROLLOUT_GPU_MEMORY_UTILIZATION"] == "0.40"
    assert no_kl["ROLLOUT_MAX_NUM_BATCHED_TOKENS"] == "32768"
    assert no_kl["STAGE123_EXPECTED_PROFILE_HASH"] == "p" * 64
    assert no_kl["FSDP_OFFLOAD"] == "True"


def test_summary_requires_auditable_gpu_headroom():
    module = load_module("stage123_matrix_memory_probe_summary", ROOT / "scripts/run_stage123_matrix_memory_probe.py")
    run = {"id": "arm", "submodel_kl": {"enabled": True}}
    repetition = {
        "status": "passed",
        "resources": {"peak_gpu_memory_used_mib": 43000, "per_gpu_memory": [{"index": 0, "total_memory_mib": 46068}]},
    }
    assert module.summarize(run, [repetition], 4096)["status"] == "failed"
    assert module.summarize(run, [repetition], 2048)["status"] == "passed"


def test_matrix_qualification_accepts_namespaced_n3_evidence():
    module = load_module(
        "stage123_matrix_memory_probe_qualification", ROOT / "scripts/run_stage123_matrix_memory_probe.py"
    )
    result = {
        "returncode": 0,
        "timed_out": False,
        "generation_count": 384,
        "validation_generation_files": ["/validation/model1/0.jsonl", "/validation/model2/0.jsonl"],
        "formal_checkpoint_files": [],
        "cleanup": {"resources_released": True},
        "resources": {"peak_gpu_memory_used_mib": 23867},
    }
    qualified = module.qualify_matrix_repetition(result)
    assert qualified["status"] == "passed"
    assert qualified["matrix_score_complete"] is True


def test_host_workload_split_uses_harness_without_host_pandas(tmp_path, monkeypatch):
    module = load_module("calibration_probe_split", ROOT / "scripts/run_calibration_probe_zero_step.py")
    monkeypatch.setattr(module.Path, "exists", lambda self: False if str(self) == "/.dockerenv" else Path.exists(self))
    monkeypatch.setattr(module, "sha256", lambda path: "source-hash")

    def fake_run(command, cwd, env, check):
        assert command[:3] == [
            "/data-1/verl07/run_train.sh",
            "python",
            "/workspace/verl/scripts/split_calibration_workload.py",
        ]
        output_root = Path(command[command.index("--output-root") + 1])
        receipt = Path(command[command.index("--receipt") + 1])
        outputs = {}
        for name in ("HumanEval+", "MBPP+", "LiveCodeBench"):
            path = output_root / f"{name.lower().replace('+', '_plus')}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"parquet")
            outputs[name] = {"path": str(path), "sha256": "x"}
        receipt.write_text(json.dumps({"source_sha256": "source-hash", "outputs": outputs}))

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    outputs = module.split_workload(tmp_path)
    assert set(outputs) == {"HumanEval+", "MBPP+", "LiveCodeBench"}
    assert all(path.is_file() for path in outputs.values())


def test_calibration_stage2_preserves_admitted_model1_identity():
    script = (ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_operational_calibration_phase.sh").read_text()
    stage2 = script.split(" stage2)", 1)[1].split(" ;;", 1)[0]
    assert 'BASE_MODEL_PATH="${BASE_MODEL_PATH:?}"' in stage2
    assert 'BASE_MODEL_PATH="${QWEN3_1P7B_MODEL_PATH:?}"' not in stage2
