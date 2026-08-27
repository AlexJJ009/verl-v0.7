#!/usr/bin/env python3
"""Render the external, candidate-bound GON-35 GRPO admission envelope."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RECIPE_CANDIDATE = "4cd5f36a7183da2027ba0665040c7cb5a51be156"
ROOT_BASE = "3a553e73549d5a2bc8f03defb0fffdfbe7249443"
RECIPE_BASE = "a752d9edb8e2c4582d95ba2c507ba173ed8d7d11"
IMAGE = "ghcr.io/alexjj009/verl-harness@sha256:d380888dc8a10796c7f841e341bd775c2d6500ede539f4ea16bb7bf0de92665d"
IMAGE_DIGEST = IMAGE.rsplit("@", 1)[1]
PARITY_IMAGE_ID = "sha256:126f9a69cd42c2f38688bc4e20daa3676dbb4b6f92f624299289c316634fc1c1"
PARITY_LAUNCHER = "/data_storage/yl_test/lgx/home/.local/bin/verl-dev-run"
PARITY_LAUNCHER_SHA256 = "0fac6c447d2ec8244bc8fccc49b2a8a0dbeaf132335512def7bb4df807b4c460"
PARITY_PAYLOAD = "python -m pytest -q tests/experiment_workflow tests/joint_training/regression/test_validation_generation_logging.py"
PARITY_REPOSITORY_MOUNT = "/workspace/verl:ro"
ENTRY = "on_policy_wdl_sft/standard_grpo/run_math_stage1_grpo.sh"
ARTIFACT_OUTPUT_PREFIX = Path("/data_storage/yl_test/lgx/artifacts/verl/outputs")
CONTAINER_OUTPUT_PREFIX = Path("/data-1/outputs")
MODEL_SHA256 = "ff8ff12d311bcc862247bd1d13f4380ec53f8af87095b183cf393147222d94b0"
DATA_SHA256 = "88d3accf25f54933b5776bfb0a4c07f5719a25199abc0ed800ccfc68eae15d66"
SCORER_SHA256 = "6fc2364da021bc5d14e1e3e8788d52cd49a3036088cacbb96d4eb5535e4473e5"


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def canonical_external(path: Path, prefix: Path, label: str) -> Path:
    value = path.expanduser().resolve()
    try:
        value.relative_to(prefix)
    except ValueError:
        fail(f"{label} must stay below {prefix}: {value}")
    return value


def load_evidence(path: Path, label: str, expected: dict[str, Any]) -> tuple[dict[str, Any], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {label} evidence {path}: {exc}")
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        fail(f"{label} evidence mismatch: {json.dumps(mismatches, sort_keys=True)}")
    return payload, sha256(path)


def load_ci_admission_evidence(
    path: Path,
    *,
    root_candidate: str,
    recipe_candidate: str,
) -> tuple[dict[str, Any], str, str]:
    """Accept a passing full CI or an exact, no-regression Base/Candidate comparison."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read CI admission evidence {path}: {exc}")

    evidence_kind = payload.get("evidence_kind")
    if evidence_kind in (None, "root_full_ci_pass"):
        expected = {
            "root_candidate_sha": root_candidate,
            "recipe_candidate_sha": recipe_candidate,
            "status": "passed",
        }
        mismatches = {
            key: {"expected": value, "actual": payload.get(key)}
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            fail(f"full CI evidence mismatch: {json.dumps(mismatches, sort_keys=True)}")
        return payload, sha256(path), "full_ci_pass"

    if evidence_kind != "root_full_ci_base_candidate_parity":
        fail(f"unsupported CI admission evidence kind: {evidence_kind!r}")

    expected_paths = {
        "schema_version": 1,
        "repository_full_name": "AlexJJ009/verl-v0.7",
        "base.root_sha": ROOT_BASE,
        "base.recipe_sha": RECIPE_BASE,
        "candidate.root_sha": root_candidate,
        "candidate.recipe_sha": recipe_candidate,
        "runtime.image": IMAGE,
        "runtime.image_id": PARITY_IMAGE_ID,
        "runtime.launcher": PARITY_LAUNCHER,
        "runtime.launcher_sha256": PARITY_LAUNCHER_SHA256,
        "runtime.payload": PARITY_PAYLOAD,
        "runtime.repository_mount": PARITY_REPOSITORY_MOUNT,
    }

    def nested_get(dotted: str) -> Any:
        value: Any = payload
        for part in dotted.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return value

    mismatches = {
        key: {"expected": value, "actual": nested_get(key)}
        for key, value in expected_paths.items()
        if nested_get(key) != value
    }
    if mismatches:
        fail(f"base-relative parity evidence mismatch: {json.dumps(mismatches, sort_keys=True)}")

    for profile_name in ("default_profile", "a800_dev_profile"):
        profile = payload.get(profile_name)
        if not isinstance(profile, dict):
            fail(f"base-relative parity evidence is missing {profile_name}")
        for side in ("base", "candidate"):
            result = profile.get(side)
            count_names = ("tests", "passed", "failed", "skipped")
            if not isinstance(result, dict) or any(
                not isinstance(result.get(name), int) for name in count_names
            ):
                fail(f"{profile_name}.{side} must contain integer test counts")
            for digest_name in ("log_sha256", "junit_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", str(result.get(digest_name, ""))):
                    fail(f"{profile_name}.{side}.{digest_name} must be a SHA-256 digest")
        required_zeroes = (
            "candidate_new_failures",
            "base_only_tests",
            "shared_failure_detail_changes",
        )
        nonzero = {name: profile.get(name) for name in required_zeroes if profile.get(name) != 0}
        if nonzero:
            fail(f"{profile_name} is not a no-regression comparison: {json.dumps(nonzero, sort_keys=True)}")
        if profile["base"]["failed"] != profile["candidate"]["failed"]:
            fail(f"{profile_name} Base/Candidate failure counts differ")
        if not isinstance(profile.get("candidate_only_tests"), int):
            fail(f"{profile_name}.candidate_only_tests must be an integer")
        for digest_name in ("failure_set_sha256",):
            if not re.fullmatch(r"[0-9a-f]{64}", str(profile.get(digest_name, ""))):
                fail(f"{profile_name}.{digest_name} must be a SHA-256 digest")

    return payload, sha256(path), "base_relative_parity"


def atomic_write(path: Path, content: str, mode: int = 0o600) -> None:
    if path.exists():
        fail(f"refusing to overwrite existing admission artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def inspect_image() -> str:
    output = subprocess.check_output(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}\n{{json .RepoDigests}}"],
        text=True,
    ).splitlines()
    if len(output) != 2:
        fail("unexpected image-inspect response")
    image_id = output[0]
    repo_digests = json.loads(output[1])
    if IMAGE not in repo_digests:
        fail(f"local image is not bound to the admitted RepoDigest: {IMAGE}")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        fail(f"invalid local image ID: {image_id}")
    return image_id


def export_line(name: str, value: str) -> str:
    return f"export {name}={shlex.quote(value)}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--p0-evidence", type=Path, required=True)
    parser.add_argument("--p1-evidence", type=Path, required=True)
    parser.add_argument("--full-ci-evidence", type=Path, required=True)
    parser.add_argument("--review-evidence", type=Path, required=True)
    args = parser.parse_args()

    if not re.fullmatch(r"GON35-MATH-QWEN3-1P7B-STAGE1-GRPO-[0-9]{8}T[0-9]{6}Z", args.run_name):
        fail("run name must be the timestamped GON-35 Math Stage1 GRPO namespace")
    repo = args.repo_root.resolve()
    recipe = repo / "recipe"
    output_root = canonical_external(args.output_root, ARTIFACT_OUTPUT_PREFIX, "output root")
    receipt_root = canonical_external(args.receipt_root, output_root, "receipt root")
    if output_root.name != args.run_name:
        fail("output root leaf must equal the exact run name")

    root_candidate = git(repo, "rev-parse", "HEAD")
    recipe_candidate = git(recipe, "rev-parse", "HEAD")
    if recipe_candidate != RECIPE_CANDIDATE:
        fail(f"recipe candidate mismatch: {recipe_candidate}")
    if git(repo, "ls-tree", "HEAD", "recipe").split()[2] != recipe_candidate:
        fail("committed Recipe gitlink does not match the checked-out candidate")
    if git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        fail("root candidate must be clean before admission rendering")
    if git(recipe, "status", "--porcelain=v1", "--untracked-files=all"):
        fail("Recipe candidate must be clean before admission rendering")

    baseline_path = recipe / "on_policy_wdl_sft/standard_grpo/scheduler/job_130_baseline.json"
    audit_path = recipe / "on_policy_wdl_sft/standard_grpo/scheduler/scheduler_audit.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    entry_blob = git(recipe, "rev-parse", f"{recipe_candidate}:{ENTRY}")
    if entry_blob != baseline["entry"]["git_blob"]:
        fail("unchanged Standard GRPO entry blob differs from the frozen Job 130 baseline")

    common = {"root_candidate_sha": root_candidate, "recipe_candidate_sha": recipe_candidate}
    _, p0_digest = load_evidence(
        args.p0_evidence,
        "P0",
        {
            **common,
            "status": "passed",
            "p0_config_match": True,
            "entry_blob": entry_blob,
            "model_sha256": MODEL_SHA256,
            "data_sha256": DATA_SHA256,
            "scorer_sha256": SCORER_SHA256,
        },
    )
    _, p1_digest = load_evidence(
        args.p1_evidence,
        "P1",
        {
            **common,
            "status": "passed",
            "p1_review_complete": True,
            "image_digest": IMAGE_DIGEST,
            "gpu_count": 8,
            "gpu_model": "NVIDIA A800-SXM4-80GB",
            "doctor_default": "passed",
            "doctor_a800": "passed",
        },
    )
    _, ci_admission_digest, ci_admission_mode = load_ci_admission_evidence(
        args.full_ci_evidence,
        root_candidate=root_candidate,
        recipe_candidate=recipe_candidate,
    )
    review, review_digest = load_evidence(
        args.review_evidence,
        "independent review",
        {**common, "status": "passed", "findings": []},
    )
    if review.get("reviewer") in (None, "", "delivery-agent"):
        fail("independent review evidence must identify a separate reviewer")

    image_id = inspect_image()
    container_output = CONTAINER_OUTPUT_PREFIX / args.run_name
    source_snapshot = {
        "schema_version": 1,
        "batch_id": "GON-35",
        "root_candidate_sha": root_candidate,
        "recipe_candidate_sha": recipe_candidate,
        "recipe_gitlink_sha": recipe_candidate,
        "entry_path": ENTRY,
        "entry_blob": entry_blob,
        "entry_sha256": baseline["entry"]["sha256"],
        "scheduler_audit_sha256": sha256(audit_path),
        "baseline_sha256": sha256(baseline_path),
        "evidence": {
            "p0_config_sha256": p0_digest,
            "p1_review_sha256": p1_digest,
            "ci_admission_mode": ci_admission_mode,
            "ci_admission_sha256": ci_admission_digest,
            "full_ci_sha256": ci_admission_digest,
            "independent_review_sha256": review_digest,
        },
    }
    snapshot_path = receipt_root / "source-snapshot.json"
    atomic_write(snapshot_path, json.dumps(source_snapshot, indent=2, sort_keys=True) + "\n")
    snapshot_digest = sha256(snapshot_path)

    runtime_env_path = receipt_root / "runtime.env"
    runtime_values = {
        "GON35_HOST_OUTPUT_ROOT": str(output_root),
        "GON35_CONTAINER_OUTPUT_ROOT": str(container_output),
        "VERL_DEV_RUN_REAL": "/data_storage/yl_test/lgx/home/.local/bin/verl-dev-run",
        "LGX_ROOT": "/data_storage/yl_test/lgx",
        "DOCKER_IMAGE": IMAGE,
        "EXPECTED_IMAGE_ID": image_id,
        "GRPO_RUNTIME_IMAGE_DIGEST": IMAGE_DIGEST,
        "GRPO_EXPECTED_IMAGE_DIGEST": IMAGE_DIGEST,
        "GRPO_ROOT_COMMIT": root_candidate,
        "GRPO_RECIPE_COMMIT": recipe_candidate,
        "GRPO_SNAPSHOT_DIGEST": snapshot_digest,
        "TOTAL_TRAINING_STEPS": "160",
        "TOTAL_EPOCHS": "3",
        "STAGE1_MODEL_PATH": baseline["p0"]["model"]["path"],
        "RUN_PREFIX": args.run_name,
        "WANDB_MODE": "offline",
    }
    runtime_text = f"export PATH={shlex.quote(str(repo / 'scripts/a800/gon35-bin'))}:$PATH\n"
    runtime_text += "".join(export_line(key, value) for key, value in runtime_values.items())
    atomic_write(runtime_env_path, runtime_text)

    admission = {
        "schema_version": 1,
        "batch_id": "GON-35",
        "root_candidate_sha": root_candidate,
        "recipe_candidate_sha": recipe_candidate,
        "scheduler": "pueue",
        "group": "gpu8",
        "group_concurrency": 1,
        "host_launcher": "verl-dev-run --a800-dev-profile",
        "host_launcher_sha256": PARITY_LAUNCHER_SHA256,
        "image": IMAGE,
        "image_digest": IMAGE_DIGEST,
        "local_image_id": image_id,
        "p0_config_match": True,
        "p1_review_complete": True,
        "model_sha256": MODEL_SHA256,
        "data_sha256": DATA_SHA256,
        "scorer_sha256": SCORER_SHA256,
        "p0_config_evidence_sha256": p0_digest,
        "p1_review_evidence_sha256": p1_digest,
        "ci_admission_mode": ci_admission_mode,
        "ci_admission_evidence_sha256": ci_admission_digest,
        "full_ci_evidence_sha256": ci_admission_digest,
        "independent_review_evidence_sha256": review_digest,
        "full_gpu_submission_allowed": True,
        "runtime_env_sha256": sha256(runtime_env_path),
        "source_snapshot_sha256": snapshot_digest,
        "run_name": args.run_name,
        "root_entry_blob": entry_blob,
        "exact_command": ["bash", "recipe/on_policy_wdl_sft/standard_grpo/run_math_stage1_grpo.sh"],
        "scheduler_audit_sha256": source_snapshot["scheduler_audit_sha256"],
        "output_root": str(output_root),
        "receipt_root": str(receipt_root),
        "runtime_env_file": str(runtime_env_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    admission_path = receipt_root / "admission.json"
    atomic_write(admission_path, json.dumps(admission, indent=2, sort_keys=True) + "\n")
    print(admission_path)


if __name__ == "__main__":
    main()
