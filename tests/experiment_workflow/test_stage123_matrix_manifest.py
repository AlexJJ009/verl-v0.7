from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/stage123_matrix_manifest.py"
MANIFEST = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/stage123_model2_kl_split_stage3.yaml"


def load_tool():
    spec = importlib.util.spec_from_file_location("stage123_matrix_manifest", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_matrix_manifest_validates_six_runs_in_dependency_order() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOL), "render", str(MANIFEST)],
        check=True,
        text=True,
        capture_output=True,
    )
    manifest = json.loads(result.stdout)

    assert [run["id"] for run in manifest["runs"]] == [
        "frac25-stage2-nokl",
        "frac25-stage2-m2kl",
        "frac25-stage3-nokl-model1",
        "frac25-stage3-m2kl-model1",
        "frac25-stage3-nokl-model2",
        "frac25-stage3-m2kl-model2",
    ]
    assert manifest["validation"]["n"] == 3
    assert manifest["validation"]["primary_metric"] == "val-core/model2/HumanEval+/acc/mean@3"
    assert manifest["hypotheses"]["primary"].startswith("model2_only_kl")
    assert manifest["decision_policy"]["minimum_effect_pp"] == 1.0
    stage2 = [run for run in manifest["runs"] if run["phase"] == "stage2"]
    assert all(run["source"]["model1_path"].endswith("qwen3-1p7b-kodcode-format-sft-frac25") for run in stage2)
    assert manifest["resource_profile"]["rollout_gpu_memory_utilization"] == 0.4
    assert manifest["resource_profile"]["rollout_max_num_batched_tokens"] == 32768
    assert manifest["resource_profile"]["rollout_free_cache_engine"] is False
    assert manifest["resource_profile"]["rollout_enable_sleep_mode"] is False
    assert manifest["resource_profile"]["ref_fsdp_offload"] is True
    assert manifest["resource_profile"]["actor_optimizer_offload"] is True
    assert manifest["resource_profile"]["actor_param_offload"] is True
    assert manifest["resource_profile"]["minimum_gpu_headroom_mib"] == 1024
    assert manifest["resource_profile"]["ref_log_prob_micro_batch_size"] == 1
    assert manifest["resource_profile"]["ref_log_prob_max_token_len_per_gpu"] == 9216
    assert manifest["resource_profile"]["submodel_kl_reference_mode"] == "standalone_enabled_submodel"


def test_matrix_manifest_rejects_missing_or_wrong_model1_identity() -> None:
    tool = load_tool()
    manifest = tool.load(MANIFEST)
    missing = copy.deepcopy(manifest)
    del missing["runs"][0]["source"]["model1_config_sha256"]
    try:
        tool.validate(missing)
    except ValueError as exc:
        assert "missing Model1 identity" in str(exc)
    else:
        raise AssertionError("missing Model1 identity did not fail closed")

    wrong = copy.deepcopy(manifest)
    wrong["runs"][0]["source"]["model1_path"] = "/models/Qwen3-1.7B-Base"
    try:
        tool.validate(wrong)
    except ValueError as exc:
        assert "FRAC25" in str(exc)
    else:
        raise AssertionError("wrong Model1 did not fail closed")


def test_matrix_manifest_rejects_safety_only_rollout_profile() -> None:
    tool = load_tool()
    manifest = tool.load(MANIFEST)
    manifest["resource_profile"]["rollout_gpu_memory_utilization"] = 0.24
    try:
        tool.validate(manifest)
    except ValueError as exc:
        assert "throughput-qualified" in str(exc)
    else:
        raise AssertionError("safety-only rollout profile did not fail closed")
