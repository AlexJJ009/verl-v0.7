# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the frozen Stage123 online validation decoder."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE = REPO_ROOT / "recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh"
COMMON_LAUNCHER = REPO_ROOT / "recipe/on_policy_wdl_sft/ablation_single_model/_common_ablation.sh"
JOINT_LAUNCHER = REPO_ROOT / "recipe/on_policy_wdl_sft/_common_wdl_sft_is_joint.sh"
AUDIT = REPO_ROOT / "scripts/stage123_phase_contract_audit.py"
MANIFEST = REPO_ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml"
PHASE_WRAPPERS = [
    REPO_ROOT / "recipe/on_policy_wdl_sft/code_task/run_s1_code_qwen3_1p7b_stage123_common.sh",
    REPO_ROOT / "recipe/on_policy_wdl_sft/code_task/run_s2_code_qwen3_1p7b_stage123_common.sh",
    REPO_ROOT / "recipe/on_policy_wdl_sft/code_task/run_s3_code_qwen3_1p7b_stage123_common.sh",
]


def _profile_environment(tmp_path: Path) -> dict[str, str]:
    index = tmp_path / "release_v5_input_output.sqlite"
    index.write_bytes(b"stage123-validation-protocol-test")
    index_sha256 = hashlib.sha256(index.read_bytes()).hexdigest()
    receipt = tmp_path / "release_v5_input_output.receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_version": "release_v5",
                "row_count": 880,
                "sha256": index_sha256,
                "size_bytes": index.stat().st_size,
            }
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "LCB_INPUT_OUTPUT_INDEX": str(index),
            "LCB_INPUT_OUTPUT_INDEX_SHA256": index_sha256,
            "LCB_INPUT_OUTPUT_INDEX_RECEIPT": str(receipt),
            "RAY_object_spilling_directory": str(tmp_path / "spill"),
            "STAGE123_RAY_TMPDIR": str(tmp_path / "ray"),
            "STAGE123_TMPDIR": str(tmp_path / "tmp"),
        }
    )
    return environment


def _profile_snapshot(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    environment = _profile_environment(tmp_path)
    environment.update(overrides)
    return subprocess.run(
        ["bash", "-c", f"source {PROFILE}; stage123_profile_snapshot"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_stage123_profile_serializes_frozen_decoder_contract(tmp_path: Path) -> None:
    result = _profile_snapshot(tmp_path)

    assert result.returncode == 0, result.stderr
    snapshot = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert {
        "TEMPERATURE": "1.0",
        "TOP_P": "1.0",
        "ROLLOUT_DO_SAMPLE": "True",
        "VAL_TEMPERATURE": "0.2",
        "VAL_TOP_P": "0.95",
        "VAL_DO_SAMPLE": "True",
        "VAL_N": "3",
    }.items() <= snapshot.items()


def test_stage123_profile_rejects_validation_decoder_drift(tmp_path: Path) -> None:
    result = _profile_snapshot(tmp_path, VAL_TEMPERATURE="1.0")

    assert result.returncode != 0
    assert "Stage123 validation temperature must equal 0.2" in result.stderr


def test_all_stage123_phases_source_the_frozen_profile() -> None:
    for wrapper in PHASE_WRAPPERS:
        assert 'source "${SCRIPT_DIR}/qwen3_1p7b_stage123_resource_profile.sh"' in wrapper.read_text(encoding="utf-8")


def test_stage123_launchers_consume_profile_owned_decoder_fields() -> None:
    for launcher_path in (COMMON_LAUNCHER, JOINT_LAUNCHER):
        launcher = launcher_path.read_text(encoding="utf-8")

        assert "rollout_do_sample=${ROLLOUT_DO_SAMPLE:-True}" in launcher
        assert "val_temperature=${VAL_TEMPERATURE:-$temperature}" in launcher
        assert "val_top_p=${VAL_TOP_P:-0.95}" in launcher
        assert "val_do_sample=${VAL_DO_SAMPLE:-True}" in launcher
        assert "actor_rollout_ref.rollout.temperature=${temperature}" in launcher
        assert "actor_rollout_ref.rollout.top_p=${top_p}" in launcher
        assert "actor_rollout_ref.rollout.do_sample=${rollout_do_sample}" in launcher
        assert "actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature}" in launcher
        assert "actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}" in launcher
        assert "actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample}" in launcher


def test_phase_contract_audit_rejects_the_invalidated_manifest() -> None:
    result = subprocess.run(
        ["python", str(AUDIT), "--manifest", str(MANIFEST)],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode != 0
    assert "FAIL: manifest render failed" in result.stderr


def test_phase_contract_audit_rejects_a_decoder_hash_mutation(tmp_path: Path) -> None:
    mutated_manifest = tmp_path / "stage123.yaml"
    mutated_manifest.write_text(
        MANIFEST.read_text(encoding="utf-8").replace(
            "sampled_decoding_semantic_hash: 8860e3bbcbe940ddf8d1e3dfce1a66fc584e2307eeca136f0788730d8d74b89e",
            "sampled_decoding_semantic_hash: " + "0" * 64,
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["python", str(AUDIT), "--manifest", str(mutated_manifest)],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode != 0
    assert "FAIL: manifest render failed" in result.stderr
