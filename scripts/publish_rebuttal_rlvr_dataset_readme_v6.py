#!/usr/bin/env python3
"""Append the reader-tested RLdataset README without changing payload bytes or history."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import publish_rebuttal_rlvr_full_dataset_v4 as full_publisher
import publish_rebuttal_rlvr_public_dataset as private_handoff
import validate_rebuttal_rlvr_dataset as dataset_validator
import verify_rebuttal_rlvr_public_release as public_verifier

REPO_ID = "AlexGeek/RLdataset"
REPO_TYPE = "dataset"
EXPECTED_PARENT = public_verifier.READER_FIXED_PARENT
EXPECTED_ANCESTORS = (
    public_verifier.README_SIMPLIFIED_PARENT,
    public_verifier.EXPECTED_PRESERVED_PARENT,
)
EXPECTED_OLD_MANIFEST_SHA256 = public_verifier.README_SIMPLIFIED_MANIFEST_SHA256
EXPECTED_NEW_MANIFEST_SHA256 = public_verifier.READER_FIXED_MANIFEST_SHA256
EXPECTED_INVENTORY_SHA256 = public_verifier.EXPECTED_INVENTORY_SHA256
EXPECTED_FILE_COUNT = public_verifier.EXPECTED_FILE_COUNT
EXPECTED_PAYLOAD_COUNT = public_verifier.EXPECTED_PAYLOAD_COUNT
EXPECTED_PAYLOAD_ROWS = public_verifier.EXPECTED_PAYLOAD_ROWS
RELEASE_PROFILE = "readme-reader-fixed-v6"
CHANGED_PATHS = ("README.md", dataset_validator.MANIFEST_PATH)
CONFIRMATION = "PUBLISH_AlexGeek/RLdataset_README_READER_FIX_PRESERVE_HISTORY"


class ReadmePublicationError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_phase(path: Path, phase: str, **details: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": now(), "phase": phase, **details}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_args(args: argparse.Namespace) -> None:
    for label, path in (
        ("--old-bundle", args.old_bundle),
        ("--new-bundle", args.new_bundle),
    ):
        if not path.is_absolute() or path.is_symlink() or not path.is_dir():
            raise ReadmePublicationError(f"{label} must be an absolute existing non-symlink directory")
    for label, path in (
        ("--receipt", args.receipt),
        ("--state-log", args.state_log),
        ("--anonymous-root", args.anonymous_root),
    ):
        if not path.is_absolute() or path.is_symlink() or path.exists():
            raise ReadmePublicationError(f"{label} must be a new absolute non-symlink path")
    if args.apply and args.confirm_publish != CONFIRMATION:
        raise ReadmePublicationError(f"--apply requires --confirm-publish {CONFIRMATION}")


def reviewed_bundle(root: Path, expected_manifest: str) -> tuple[dict[str, str], dict[str, Any]]:
    result = dataset_validator.validate(root)
    if (
        result["inventory_sha256"] != EXPECTED_INVENTORY_SHA256
        or result["manifest_sha256"] != expected_manifest
        or result["file_count"] != EXPECTED_FILE_COUNT
        or result["payload_count"] != EXPECTED_PAYLOAD_COUNT
        or result["payload_rows"] != EXPECTED_PAYLOAD_ROWS
    ):
        raise ReadmePublicationError(f"reviewed bundle pins drifted: {result}")
    checksums = dataset_validator.parse_checksum_manifest(root / dataset_validator.MANIFEST_PATH)
    checksums[dataset_validator.MANIFEST_PATH] = expected_manifest
    if len(checksums) != EXPECTED_FILE_COUNT:
        raise ReadmePublicationError("reviewed bundle exact file count drifted")
    return checksums, result


def prove_readme_only_change(
    old_checksums: dict[str, str],
    new_checksums: dict[str, str],
) -> None:
    if set(old_checksums) != set(new_checksums):
        raise ReadmePublicationError("README update must not add or remove repository paths")
    changed = tuple(sorted(path for path in old_checksums if old_checksums[path] != new_checksums[path]))
    if changed != tuple(sorted(CHANGED_PATHS)):
        raise ReadmePublicationError(f"README update changed unexpected paths: {changed}")
    if any(
        old_checksums[path] != new_checksums[path]
        for path in old_checksums
        if path.endswith(".parquet")
    ):
        raise ReadmePublicationError("README update changed a Parquet payload")


def verify_remote_state(
    api: private_handoff.GuardedHfApi,
    revision: str,
    checksums: dict[str, str],
    expected_history: list[str],
) -> dict[str, Any]:
    return full_publisher.verify_exact_release_state(
        api,
        revision,
        checksums,
        expected_private=False,
        expected_history=expected_history,
    )


def append_readme_commit(
    api: private_handoff.GuardedHfApi,
    new_bundle: Path,
    new_checksums: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    from huggingface_hub import CommitOperationAdd

    operations = [
        CommitOperationAdd(path_in_repo=path, path_or_fileobj=new_bundle / path)
        for path in CHANGED_PATHS
    ]
    try:
        commit = api.create_commit(
            REPO_ID,
            operations=operations,
            repo_type=REPO_TYPE,
            revision="main",
            parent_commit=EXPECTED_PARENT,
            create_pr=False,
            commit_message="Clarify dataset validation and launcher path instructions",
        )
        revision = getattr(commit, "oid", None)
    except Exception as commit_error:
        observed = api.repo_info(REPO_ID, repo_type=REPO_TYPE, files_metadata=False)
        revision = getattr(observed, "sha", None)
        if not public_verifier.is_commit_sha(revision) or revision == EXPECTED_PARENT:
            raise ReadmePublicationError(
                f"README commit failed without a recoverable new HEAD: {commit_error}"
            ) from commit_error
        try:
            state = verify_remote_state(
                api,
                revision,
                new_checksums,
                [revision, EXPECTED_PARENT, *EXPECTED_ANCESTORS],
            )
        except Exception as verification_error:
            raise ReadmePublicationError(
                "README commit failed and observed HEAD is not the exact reviewed tree: "
                f"commit={commit_error}; verify={verification_error}"
            ) from verification_error
        state["recovered_after_commit_error"] = f"{type(commit_error).__name__}: {commit_error}"
        return revision, state

    if not public_verifier.is_commit_sha(revision):
        raise ReadmePublicationError(f"README commit returned invalid SHA: {revision!r}")
    state = verify_remote_state(
        api,
        revision,
        new_checksums,
        [revision, EXPECTED_PARENT, *EXPECTED_ANCESTORS],
    )
    return revision, state


def run_anonymous_verifier(
    args: argparse.Namespace,
    revision: str,
) -> tuple[dict[str, Any], Path, Path]:
    args.anonymous_root.mkdir(mode=0o700, parents=True)
    home = args.anonymous_root / "home"
    hf_home = args.anonymous_root / "hf-home"
    home.mkdir(mode=0o700)
    hf_home.mkdir(mode=0o700)
    local_dir = args.anonymous_root / "RLdataset"
    receipt = args.anonymous_root / "anonymous-readme-v6-receipt.json"
    log = args.anonymous_root / "anonymous-readme-v6-verifier.log"

    environment = dict(os.environ)
    for name in public_verifier.TOKEN_ENV_VARS:
        environment.pop(name, None)
    environment.pop("HF_HUB_OFFLINE", None)
    environment.update(
        {
            "HOME": str(home),
            "HF_HOME": str(hf_home),
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_ENDPOINT": private_handoff.HF_ENDPOINT,
        }
    )
    command = [
        sys.executable,
        str(Path(public_verifier.__file__).resolve()),
        "--release-profile",
        RELEASE_PROFILE,
        "--bundle",
        str(args.new_bundle),
        "--revision",
        revision,
        "--local-dir",
        str(local_dir),
        "--receipt",
        str(receipt),
    ]
    for preserved_revision in (EXPECTED_PARENT, *EXPECTED_ANCESTORS):
        command.extend(("--preserved-revision", preserved_revision))
    completed = subprocess.run(
        command,
        env=environment,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
        check=False,
    )
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0 or not receipt.is_file():
        raise ReadmePublicationError(f"anonymous verifier failed with status {completed.returncode}; see {log}")
    result = json.loads(receipt.read_text(encoding="utf-8"))
    if (
        result.get("ok") is not True
        or result.get("revision") != revision
        or result.get("release_profile") != RELEASE_PROFILE
        or result.get("anonymous") is not True
        or result.get("api_private") is not False
        or result.get("api_gated") is not False
        or result.get("manifest_sha256") != EXPECTED_NEW_MANIFEST_SHA256
        or result.get("inventory_sha256") != EXPECTED_INVENTORY_SHA256
        or result.get("commit_ids") != [revision, EXPECTED_PARENT, *EXPECTED_ANCESTORS]
    ):
        raise ReadmePublicationError("anonymous README verifier receipt does not satisfy the release gate")
    return result, receipt, log


def preflight(
    api: private_handoff.GuardedHfApi,
    old_checksums: dict[str, str],
) -> dict[str, Any]:
    state = verify_remote_state(
        api,
        EXPECTED_PARENT,
        old_checksums,
        [EXPECTED_PARENT, *EXPECTED_ANCESTORS],
    )
    return {
        **state,
        "history_policy": "append_only_preserve_existing_history",
        "allowed_changed_paths": list(CHANGED_PATHS),
    }


def publish(
    api: private_handoff.GuardedHfApi,
    args: argparse.Namespace,
    old_checksums: dict[str, str],
    new_checksums: dict[str, str],
    new_result: dict[str, Any],
) -> dict[str, Any]:
    parent_state = preflight(api, old_checksums)
    append_phase(args.state_log, "public_parent_verified", head=EXPECTED_PARENT)
    current = api.repo_info(REPO_ID, repo_type=REPO_TYPE, files_metadata=False)
    if current.sha != EXPECTED_PARENT or current.private is not False or getattr(current, "gated", None) is not False:
        raise ReadmePublicationError("remote state drifted immediately before README append")

    revision, authenticated_state = append_readme_commit(api, args.new_bundle, new_checksums)
    append_phase(args.state_log, "readme_commit_authenticated_gate_passed", revision=revision)
    try:
        anonymous_result, anonymous_receipt, anonymous_log = run_anonymous_verifier(args, revision)
    except Exception as exc:
        append_phase(
            args.state_log,
            "readme_commit_created_anonymous_gate_failed",
            revision=revision,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    append_phase(args.state_log, "readme_commit_anonymous_gate_passed", revision=revision)

    result = {
        "schema_version": 1,
        "receipt_kind": "hf_dataset_readme_reader_fix_release",
        "completed_at": now(),
        "ok": True,
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "revision": revision,
        "preserved_history": [revision, EXPECTED_PARENT, *EXPECTED_ANCESTORS],
        "changed_paths": list(CHANGED_PATHS),
        "parquet_payloads_changed": False,
        "bundle": {
            "file_count": new_result["file_count"],
            "payload_count": new_result["payload_count"],
            "payload_rows": new_result["payload_rows"],
            "inventory_sha256": EXPECTED_INVENTORY_SHA256,
            "manifest_sha256": EXPECTED_NEW_MANIFEST_SHA256,
        },
        "parent_state": parent_state,
        "authenticated_state": authenticated_state,
        "anonymous_verification": anonymous_result,
        "anonymous_receipt": str(anonymous_receipt),
        "anonymous_log": str(anonymous_log),
    }
    full_publisher.write_json_new(args.receipt, result)
    append_phase(args.state_log, "release_receipt_written", revision=revision)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-bundle", type=Path, required=True)
    parser.add_argument("--new-bundle", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--state-log", type=Path, required=True)
    parser.add_argument("--anonymous-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-publish")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        old_checksums, _ = reviewed_bundle(args.old_bundle, EXPECTED_OLD_MANIFEST_SHA256)
        new_checksums, new_result = reviewed_bundle(args.new_bundle, EXPECTED_NEW_MANIFEST_SHA256)
        prove_readme_only_change(old_checksums, new_checksums)
        admission = private_handoff.validate_network_admission(private_handoff.admit_hf_network())
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ReadmePublicationError("huggingface_hub is required") from exc
        api = private_handoff.GuardedHfApi(HfApi(endpoint=private_handoff.HF_ENDPOINT), admission)
        full_publisher.identity_name(api)
        if not args.apply:
            result = {
                "ok": True,
                "mode": "read_only_preflight",
                "mutation_enabled": False,
                "remote": preflight(api, old_checksums),
                "changed_paths": list(CHANGED_PATHS),
                "parquet_payloads_changed": False,
                "new_bundle": new_result,
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        args.state_log.parent.mkdir(parents=True, exist_ok=True)
        args.state_log.touch(mode=0o600, exist_ok=False)
        append_phase(args.state_log, "readme_publication_started", expected_parent=EXPECTED_PARENT)
        result = publish(api, args, old_checksums, new_checksums, new_result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        ReadmePublicationError,
        full_publisher.FullPublicationError,
        public_verifier.PublicVerificationError,
        private_handoff.PublishError,
        dataset_validator.ValidationError,
        OSError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
