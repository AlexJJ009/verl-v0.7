#!/usr/bin/env python3
"""Fail-closed CPU audit for the frozen Stage123 execution contract."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh"
WRAPPERS = [
    ROOT / "recipe/on_policy_wdl_sft/code_task/run_s1_code_qwen3_1p7b_stage123_common.sh",
    ROOT / "recipe/on_policy_wdl_sft/code_task/run_s2_code_qwen3_1p7b_stage123_common.sh",
    ROOT / "recipe/on_policy_wdl_sft/code_task/run_s3_code_qwen3_1p7b_stage123_common.sh",
]
LAUNCHERS = [
    ROOT / "recipe/on_policy_wdl_sft/ablation_single_model/_common_ablation.sh",
    ROOT / "recipe/on_policy_wdl_sft/_common_wdl_sft_is_joint.sh",
]
EXPECTED_DECODER = {
    "TEMPERATURE": "1.0",
    "TOP_P": "1.0",
    "ROLLOUT_DO_SAMPLE": "True",
    "VAL_TEMPERATURE": "0.2",
    "VAL_TOP_P": "0.95",
    "VAL_DO_SAMPLE": "True",
    "VAL_N": "3",
}


def semantic_hash() -> str:
    contract = {
        "main_validation": {"do_sample": True, "n": 1, "temperature": 0.2, "top_p": 0.95},
        "training_rollout": {"do_sample": True, "temperature": 1.0, "top_p": 1.0},
    }
    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def rendered_manifest(manifest_path: Path) -> dict:
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/experiment_manifest.py"),
                "render",
                str(manifest_path),
                "--format",
                "json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "manifest render failed"
        raise SystemExit(f"FAIL: manifest render failed: {detail}") from error
    return json.loads(result.stdout)


def profile_snapshot() -> tuple[dict[str, str], str]:
    result = subprocess.run(
        [
            "bash",
            "-lc",
            f"source {PROFILE}; stage123_profile_snapshot; printf '__HASH__=%s\\n' \"$(stage123_profile_hash)\"",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    snapshot: dict[str, str] = {}
    profile_hash = ""
    for line in result.stdout.splitlines():
        key, value = line.split("=", 1)
        if key == "__HASH__":
            profile_hash = value
        else:
            snapshot[key] = value
    return snapshot, profile_hash


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def audit_static_execution_paths() -> None:
    profile_source = 'source "${SCRIPT_DIR}/qwen3_1p7b_stage123_resource_profile.sh"'
    for wrapper in WRAPPERS:
        require(
            profile_source in wrapper.read_text(encoding="utf-8"), f"{wrapper} does not source the Stage123 profile"
        )
    for launcher in LAUNCHERS:
        text = launcher.read_text(encoding="utf-8")
        for required in (
            "rollout_do_sample=${ROLLOUT_DO_SAMPLE:-True}",
            "val_temperature=${VAL_TEMPERATURE:-$temperature}",
            "val_top_p=${VAL_TOP_P:-0.95}",
            "val_do_sample=${VAL_DO_SAMPLE:-True}",
            "actor_rollout_ref.rollout.do_sample=${rollout_do_sample}",
            "actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample}",
        ):
            require(required in text, f"{launcher} does not consume {required}")
    adapter = (ROOT / "scripts/stage123_phase_adapter.py").read_text(encoding="utf-8")
    require('"DATA_SHUFFLE": "False"' in adapter, "Stage123 adapter does not force DATA_SHUFFLE=False")


def audit_manifest(manifest: dict, snapshot: dict[str, str], profile_hash: str) -> None:
    require(
        {key: snapshot.get(key) for key in EXPECTED_DECODER} == EXPECTED_DECODER,
        "profile decoder differs from the frozen contract",
    )
    profile = manifest["resource_profile"]
    require(profile["name"] == snapshot["STAGE123_RESOURCE_PROFILE_NAME"], "manifest profile name drift")
    require(profile["sha256"] == profile_hash, "manifest profile hash drift")
    require(
        manifest["semantics"]["sampled_decoding_semantic_hash"] == semantic_hash(),
        "manifest decoder semantic hash drift",
    )
    runs = manifest["runs"]
    require(
        [run["id"] for run in runs] == ["frac25-stage1-control", "frac25-stage2", "frac25-stage3"],
        "fresh phase order drift",
    )
    serialized_runs = json.dumps(runs, sort_keys=True)
    for forbidden in ("v13", "v14", "treatment-reuse", "stage3-handoff-reuse", "certified-control"):
        require(forbidden not in serialized_runs.lower(), f"legacy reuse reference in fresh manifest: {forbidden}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    manifest = rendered_manifest(args.manifest)
    snapshot, profile_hash = profile_snapshot()
    audit_static_execution_paths()
    audit_manifest(manifest, snapshot, profile_hash)
    print(
        json.dumps(
            {
                "ok": True,
                "manifest_sha256": manifest["manifest_sha256"],
                "profile_sha256": profile_hash,
                "decoder_semantic_sha256": semantic_hash(),
                "run_ids": [run["id"] for run in manifest["runs"]],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
