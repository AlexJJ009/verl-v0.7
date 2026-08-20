from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "stage123_matrix_throughput_probe", ROOT / "scripts/run_stage123_matrix_throughput_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_environment_binds_model1_and_throughput_profile(tmp_path: Path):
    module = load_module()
    run = {
        "id": "arm",
        "train_file": "/train.parquet",
        "source": {
            "checkpoint_root": "/checkpoints/source",
            "run_prefix": "SOURCE",
            "handoff_step": 40,
            "model2_path": "/models/model2",
            "model1_path": "/models/frac25",
            "model1_config_sha256": "a" * 64,
            "model1_tokenizer_config_sha256": "b" * 64,
            "model1_chat_template_sha256": "c" * 64,
            "model1_provenance_path": "/models/frac25/source.json",
            "model1_provenance_sha256": "d" * 64,
        },
        "submodel_kl": {
            "enabled": False,
            "model1_enabled": False,
            "model1_coef": 0.0,
            "model2_enabled": False,
            "model2_coef": 0.0,
        },
    }
    profile = {
        "sha256": "p" * 64,
        "rollout_gpu_memory_utilization": 0.4,
        "rollout_max_num_batched_tokens": 32768,
        "ref_log_prob_micro_batch_size": 1,
        "ref_log_prob_max_token_len_per_gpu": 9216,
    }
    environment = module.run_environment(run, tmp_path, profile, "probe")
    assert environment["BASE_MODEL_PATH"] == "/models/frac25"
    assert environment["EXPECTED_MODEL1_CONFIG_SHA256"] == "a" * 64
    assert environment["ROLLOUT_GPU_MEMORY_UTILIZATION"] == "0.40"
    assert environment["ROLLOUT_MAX_NUM_BATCHED_TOKENS"] == "32768"
    assert environment["REF_LOG_PROB_MICRO_BATCH_SIZE"] == "1"
    assert environment["REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU"] == "9216"


def test_peak_resources_reports_headroom():
    module = load_module()
    summary = module.peak_resources(
        [[{"index": 0, "memory_used_mib": 32000, "memory_total_mib": 46068, "utilization_gpu_percent": 98}]]
    )
    assert summary["peak_gpu_memory_used_mib"] == 32000
    assert summary["minimum_gpu_headroom_mib"] == 14068
    assert summary["peak_gpu_utilization_percent"] == 98


def test_qualification_accepts_dual_mode_metadata(tmp_path: Path):
    module = load_module()
    log = tmp_path / "probe.log"
    log.write_text("completed\n")
    run = {
        "source": {"model1_path": "/models/frac25", "model2_path": "/models/model2"},
    }
    result = {
        "returncode": 0,
        "timed_out": False,
        "optimizer_steps": 1,
        "metrics": {"step_time_seconds": 70.0, "rollout_tokens_per_second": 4000.0, "actor_grad_norm": 8.0},
        "resources": {"peak_gpu_memory_used_mib": 43000, "minimum_gpu_headroom_mib": 3000},
        "cleanup": {"resources_released": True},
        "formal_checkpoint_files": [],
        "joint_model_sources": {"mode": "dual", "model1": "/models/frac25", "model2": "/models/model2"},
        "log": str(log),
    }
    assert module.qualify_result(result, run)["status"] == "passed"
