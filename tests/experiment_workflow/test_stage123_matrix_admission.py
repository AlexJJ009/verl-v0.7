from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "stage123_matrix_admission", ROOT / "scripts/stage123_matrix_admission.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gpu_facts_are_structured(monkeypatch):
    module = load_module()
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda *args, **kwargs: "0, NVIDIA L40S, 46068\n1, NVIDIA L40S, 46068\n"
    )
    assert module.gpu_facts() == [
        {"index": 0, "name": "NVIDIA L40S", "memory_total_mib": 46068},
        {"index": 1, "name": "NVIDIA L40S", "memory_total_mib": 46068},
    ]


def test_throughput_probe_requires_optimizer_and_all_stage2_arms():
    module = load_module()
    identity = {
        "model1_path": "/models/frac25",
        "model1_config_sha256": "a" * 64,
        "model1_tokenizer_config_sha256": "b" * 64,
        "model1_chat_template_sha256": "c" * 64,
        "model1_provenance_path": "/models/frac25/source.json",
        "model1_provenance_sha256": "d" * 64,
    }
    manifest = {
        "manifest_sha256": "m",
        "resource_profile": {
            "rollout_gpu_memory_utilization": 0.4,
            "rollout_max_num_batched_tokens": 32768,
            "rollout_free_cache_engine": False,
            "rollout_enable_sleep_mode": False,
            "ref_fsdp_offload": True,
            "actor_optimizer_offload": True,
            "actor_param_offload": True,
            "minimum_gpu_headroom_mib": 1024,
            "ref_log_prob_micro_batch_size": 1,
            "ref_log_prob_max_token_len_per_gpu": 9216,
        },
        "runs": [
            {"id": "no-kl", "phase": "stage2", "source": identity},
            {"id": "m2kl", "phase": "stage2", "source": identity},
        ],
    }
    run = {
        "status": "passed",
        "optimizer_steps": 1,
        "metrics": {"step_time_seconds": 60.0, "rollout_tokens_per_second": 1000.0},
        "cleanup": {"resources_released": True},
        "formal_checkpoint_files": [],
    }
    probe = {
        "result_type": "stage123_matrix_throughput_probe",
        "status": "passed",
        "manifest_sha256": "m",
        "training_steps": 1,
        "optimizer_enabled": True,
        "rollout_gpu_memory_utilization": 0.4,
        "rollout_max_num_batched_tokens": 32768,
        "rollout_free_cache_engine": False,
        "rollout_enable_sleep_mode": False,
        "ref_fsdp_offload": True,
        "actor_optimizer_offload": True,
        "actor_param_offload": True,
        "minimum_gpu_headroom_mib": 1024,
        "ref_log_prob_micro_batch_size": 1,
        "ref_log_prob_max_token_len_per_gpu": 9216,
        "model1_identity": identity,
        "runs": [{"run_id": "no-kl", **run}, {"run_id": "m2kl", **run}],
    }
    module.validate_throughput_probe(probe, manifest, identity)

    no_optimizer = copy.deepcopy(probe)
    no_optimizer["optimizer_enabled"] = False
    try:
        module.validate_throughput_probe(no_optimizer, manifest, identity)
    except SystemExit as exc:
        assert "optimizer step" in str(exc)
    else:
        raise AssertionError("zero-optimizer throughput evidence did not fail closed")

    missing_arm = copy.deepcopy(probe)
    missing_arm["runs"].pop()
    try:
        module.validate_throughput_probe(missing_arm, manifest, identity)
    except SystemExit as exc:
        assert "every Stage2 arm" in str(exc)
    else:
        raise AssertionError("partial-arm throughput evidence did not fail closed")
