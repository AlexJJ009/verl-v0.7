#!/usr/bin/env python3
"""Publish the full-13 RLdataset release while preserving repository history."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import publish_rebuttal_rlvr_public_dataset as private_handoff
import validate_rebuttal_rlvr_dataset as dataset_validator
import verify_rebuttal_rlvr_public_release as public_verifier

REPO_ID = "AlexGeek/RLdataset"
REPO_TYPE = "dataset"
EXPECTED_PRIVATE_PARENT = "da622cf077ca3f0eaf0ebc55dd4e115d0ebc0b9c"
EXPECTED_INVENTORY_SHA256 = public_verifier.EXPECTED_INVENTORY_SHA256
EXPECTED_MANIFEST_SHA256 = public_verifier.EXPECTED_MANIFEST_SHA256
EXPECTED_FILE_COUNT = public_verifier.EXPECTED_FILE_COUNT
EXPECTED_PAYLOAD_COUNT = public_verifier.EXPECTED_PAYLOAD_COUNT
EXPECTED_PAYLOAD_ROWS = public_verifier.EXPECTED_PAYLOAD_ROWS
CONFIRMATION = "PUBLISH_AlexGeek/RLdataset_FULL13_PRESERVE_HISTORY"


class FullPublicationError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_commit_sha(value: Any) -> bool:
    return public_verifier.is_commit_sha(value)


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FullPublicationError(f"refusing to replace JSON artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_phase(path: Path, phase: str, **details: Any) -> None:
    entry = {"at": now(), "phase": phase, **details}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_artifact_paths(args: argparse.Namespace) -> None:
    for label, path in (
        ("--receipt", args.receipt),
        ("--state-log", args.state_log),
        ("--anonymous-root", args.anonymous_root),
    ):
        if not path.is_absolute() or path.is_symlink() or path.exists():
            raise FullPublicationError(f"{label} must be a new absolute non-symlink path")
    if args.receipt == args.state_log:
        raise FullPublicationError("--receipt and --state-log must be distinct")
    if args.expected_parent != EXPECTED_PRIVATE_PARENT:
        raise FullPublicationError(f"--expected-parent must equal the reviewed private HEAD {EXPECTED_PRIVATE_PARENT}")
    if args.apply and args.confirm_publish != CONFIRMATION:
        raise FullPublicationError(f"--apply requires --confirm-publish {CONFIRMATION}")


def reviewed_public_bundle(bundle: Path) -> tuple[dict[str, str], dict[str, Any]]:
    result = dataset_validator.validate(bundle)
    if (
        result["inventory_sha256"] != EXPECTED_INVENTORY_SHA256
        or result["manifest_sha256"] != EXPECTED_MANIFEST_SHA256
        or result["file_count"] != EXPECTED_FILE_COUNT
        or result["payload_count"] != EXPECTED_PAYLOAD_COUNT
        or result["payload_rows"] != EXPECTED_PAYLOAD_ROWS
    ):
        raise FullPublicationError(f"reviewed public bundle pins drifted: {result}")
    inventory = json.loads((bundle / dataset_validator.INVENTORY_PATH).read_text(encoding="utf-8"))
    if (
        inventory.get("publication_status") != "owner_approved_for_public_release"
        or inventory.get("publication_decision", {}).get("decision_id") != "rebuttal-rlvr-full13-public-20260729"
        or inventory.get("publication_decision", {}).get("decision") != "publish_full_13_payload_collection"
        or inventory.get("packaging", {}).get("publication_format_conversion") is not False
    ):
        raise FullPublicationError("bundle is not bound to the owner-approved byte-preserving release")
    checksums = dataset_validator.parse_checksum_manifest(bundle / dataset_validator.MANIFEST_PATH)
    checksums[dataset_validator.MANIFEST_PATH] = EXPECTED_MANIFEST_SHA256
    if len(checksums) != EXPECTED_FILE_COUNT:
        raise FullPublicationError("public bundle exact file count drifted")
    return checksums, result


def identity_name(api: private_handoff.GuardedHfApi) -> str:
    identity = api.whoami()
    name = identity.get("name") if isinstance(identity, dict) else None
    if name != "AlexGeek":
        raise FullPublicationError(f"credential identity must be AlexGeek, observed {name!r}")
    return name


def ref_snapshot(
    api: private_handoff.GuardedHfApi,
    expected_main: str,
) -> dict[str, Any]:
    refs = api.list_repo_refs(
        REPO_ID,
        repo_type=REPO_TYPE,
        include_pull_requests=True,
    )

    def describe(values: Any) -> list[dict[str, Any]]:
        return [
            {
                "name": getattr(item, "name", None),
                "ref": getattr(item, "ref", None),
                "target_commit": getattr(item, "target_commit", None),
            }
            for item in (values or [])
        ]

    branches = describe(getattr(refs, "branches", None))
    converts = describe(getattr(refs, "converts", None))
    tags = describe(getattr(refs, "tags", None))
    pull_requests = describe(getattr(refs, "pull_requests", None))
    if branches != [
        {
            "name": "main",
            "ref": "refs/heads/main",
            "target_commit": expected_main,
        }
    ]:
        raise FullPublicationError(f"main ref does not target the expected commit: {branches}")
    if tags or pull_requests:
        raise FullPublicationError(f"unexpected tags or pull-request refs: tags={tags}, prs={pull_requests}")
    if len(converts) > 1:
        raise FullPublicationError(f"unexpected conversion refs: {converts}")
    if converts and (
        converts[0]["name"] != "parquet"
        or converts[0]["ref"] != "refs/convert/parquet"
        or not is_commit_sha(converts[0]["target_commit"])
    ):
        raise FullPublicationError(f"unexpected conversion ref: {converts}")
    return {
        "branches": branches,
        "converts": converts,
        "tags": tags,
        "pull_requests": pull_requests,
    }


def read_only_private_preflight(
    api: private_handoff.GuardedHfApi,
    args: argparse.Namespace,
    initial_admission: dict[str, Any],
) -> dict[str, Any]:
    old_checksums = private_handoff.load_and_verify_bundle(args.old_bundle)
    private_handoff.verify_reviewed_bundle(args.old_bundle)
    info = api.repo_info(
        REPO_ID,
        repo_type=REPO_TYPE,
        revision=args.expected_parent,
        files_metadata=True,
    )
    if (
        info.id != REPO_ID
        or info.sha != args.expected_parent
        or info.private is not True
        or getattr(info, "gated", None) is not False
    ):
        raise FullPublicationError("existing repository is not the reviewed private ungated parent")
    private_handoff.verify_remote(
        api,
        REPO_ID,
        args.expected_parent,
        old_checksums,
    )
    commits = private_handoff.list_commit_ids(api, REPO_ID)
    if commits != [args.expected_parent]:
        raise FullPublicationError(f"reviewed parent history drifted before append-only publication: {commits}")
    refs = ref_snapshot(api, args.expected_parent)
    final_admission = api._admit_same_route()
    if final_admission["selected_leaf_sha256"] != initial_admission["selected_leaf_sha256"]:
        raise FullPublicationError("network route changed during private preflight")
    return {
        "repo": private_handoff.repo_snapshot(info),
        "commit_ids": commits,
        "refs": refs,
        "verified_file_sha256": old_checksums,
        "history_policy": "preserve_existing_history",
        "network_admission": final_admission,
    }


def verify_exact_release_state(
    api: private_handoff.GuardedHfApi,
    revision: str,
    checksums: dict[str, str],
    *,
    expected_private: bool,
    expected_history: list[str],
) -> dict[str, Any]:
    info = api.repo_info(
        REPO_ID,
        repo_type=REPO_TYPE,
        revision=revision,
        files_metadata=True,
    )
    head = api.repo_info(REPO_ID, repo_type=REPO_TYPE, files_metadata=False)
    if (
        info.id != REPO_ID
        or info.sha != revision
        or head.sha != revision
        or info.private is not expected_private
        or head.private is not expected_private
        or getattr(info, "gated", None) is not False
        or getattr(head, "gated", None) is not False
    ):
        raise FullPublicationError("repository state does not match the expected immutable release")
    private_handoff.verify_remote(api, REPO_ID, revision, checksums)
    commits = private_handoff.list_commit_ids(api, REPO_ID)
    if commits != expected_history:
        raise FullPublicationError(f"repository history was not preserved exactly: {commits}")
    refs = ref_snapshot(api, revision)
    return {
        "repo": private_handoff.repo_snapshot(info),
        "commit_ids": commits,
        "refs": refs,
    }


def upload_or_recover_exact_commit(
    api: private_handoff.GuardedHfApi,
    args: argparse.Namespace,
    checksums: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    from huggingface_hub import CommitOperationAdd

    operations = [
        CommitOperationAdd(
            path_in_repo=relative,
            path_or_fileobj=args.bundle / relative,
        )
        for relative in sorted(checksums)
    ]
    try:
        commit = api.create_commit(
            REPO_ID,
            operations=operations,
            repo_type=REPO_TYPE,
            revision="main",
            parent_commit=args.expected_parent,
            create_pr=False,
            commit_message=("Publish full RLVR runtime datasets with upstream attribution"),
        )
        uploaded_sha = getattr(commit, "oid", None)
    except Exception as upload_error:
        observed = api.repo_info(
            REPO_ID,
            repo_type=REPO_TYPE,
            files_metadata=False,
        )
        uploaded_sha = getattr(observed, "sha", None)
        if not is_commit_sha(uploaded_sha) or uploaded_sha == args.expected_parent:
            raise FullPublicationError(
                f"upload failed without a recoverable new HEAD: {upload_error}"
            ) from upload_error
        try:
            state = verify_exact_release_state(
                api,
                uploaded_sha,
                checksums,
                expected_private=True,
                expected_history=[uploaded_sha, args.expected_parent],
            )
        except Exception as verification_error:
            raise FullPublicationError(
                "upload failed and the observed private HEAD is not the exact "
                f"reviewed release: upload={upload_error}; verify={verification_error}"
            ) from verification_error
        state["recovered_after_upload_error"] = f"{type(upload_error).__name__}: {upload_error}"
        return uploaded_sha, state

    if not is_commit_sha(uploaded_sha):
        raise FullPublicationError(f"upload returned an invalid commit SHA: {uploaded_sha!r}")
    state = verify_exact_release_state(
        api,
        uploaded_sha,
        checksums,
        expected_private=True,
        expected_history=[uploaded_sha, args.expected_parent],
    )
    return uploaded_sha, state


def run_anonymous_verifier(
    args: argparse.Namespace,
    public_sha: str,
) -> tuple[dict[str, Any], Path, Path]:
    args.anonymous_root.mkdir(mode=0o700, parents=True)
    home = args.anonymous_root / "home"
    hf_home = args.anonymous_root / "hf-home"
    home.mkdir(mode=0o700)
    hf_home.mkdir(mode=0o700)
    local_dir = args.anonymous_root / "RLdataset"
    receipt = args.anonymous_root / "anonymous-public-receipt.json"
    log = args.anonymous_root / "anonymous-public-verifier.log"

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
        "--bundle",
        str(args.bundle),
        "--revision",
        public_sha,
        "--local-dir",
        str(local_dir),
        "--receipt",
        str(receipt),
        "--preserved-revision",
        args.expected_parent,
    ]
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
        raise FullPublicationError(f"anonymous verifier failed with status {completed.returncode}; see {log}")
    result = json.loads(receipt.read_text(encoding="utf-8"))
    if (
        result.get("ok") is not True
        or result.get("revision") != public_sha
        or result.get("anonymous") is not True
        or result.get("api_private") is not False
        or result.get("api_gated") is not False
        or result.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or result.get("inventory_sha256") != EXPECTED_INVENTORY_SHA256
        or [item.get("revision") for item in result.get("preserved_history", [])] != [args.expected_parent]
    ):
        raise FullPublicationError("anonymous verifier receipt does not satisfy the public gate")
    return result, receipt, log


def apply_publication(
    api: private_handoff.GuardedHfApi,
    args: argparse.Namespace,
    initial_admission: dict[str, Any],
    http_observer: Any,
    checksums: dict[str, str],
    bundle_result: dict[str, Any],
) -> dict[str, Any]:
    preflight = read_only_private_preflight(api, args, initial_admission)
    old_paths = set(preflight["verified_file_sha256"])
    if set(checksums) != old_paths | {"validate_dataset.py"}:
        raise FullPublicationError("v4 must replace every reviewed v3 path and add only validate_dataset.py")
    append_phase(
        args.state_log,
        "private_parent_verified",
        head=args.expected_parent,
        history_policy="preserve_existing_history",
    )

    current = api.repo_info(REPO_ID, repo_type=REPO_TYPE, files_metadata=False)
    if (
        current.sha != args.expected_parent
        or current.private is not True
        or getattr(current, "gated", None) is not False
    ):
        raise FullPublicationError("remote HEAD drifted immediately before append-only upload")
    uploaded_sha, uploaded_private = upload_or_recover_exact_commit(
        api,
        args,
        checksums,
    )
    append_phase(
        args.state_log,
        "private_append_commit_verified",
        uploaded_sha=uploaded_sha,
        preserved_parent=args.expected_parent,
    )

    public_mutation_attempted = False
    try:
        current = api.repo_info(
            REPO_ID,
            repo_type=REPO_TYPE,
            files_metadata=False,
        )
        if current.sha != uploaded_sha or current.private is not True:
            raise FullPublicationError("private release drifted immediately before public transition")
        public_mutation_attempted = True
        api.update_repo_settings(
            REPO_ID,
            repo_type=REPO_TYPE,
            private=False,
            gated=False,
        )
        authenticated_public = verify_exact_release_state(
            api,
            uploaded_sha,
            checksums,
            expected_private=False,
            expected_history=[uploaded_sha, args.expected_parent],
        )
        append_phase(
            args.state_log,
            "authenticated_public_gate_passed",
            public_sha=uploaded_sha,
        )
        observed_hosts = http_observer.observed_hosts
        final_admission = api._admit_same_route(tuple(observed_hosts))
        if final_admission["connection_hosts_verified"] != observed_hosts:
            raise FullPublicationError("final authenticated route receipt does not bind every connection host")
        anonymous_result, anonymous_receipt, anonymous_log = run_anonymous_verifier(args, uploaded_sha)
        append_phase(
            args.state_log,
            "anonymous_public_gate_passed",
            public_sha=uploaded_sha,
        )
        append_phase(
            args.state_log,
            "remote_release_verified",
            public_sha=uploaded_sha,
        )
    except Exception as publication_error:
        if public_mutation_attempted:
            try:
                api.update_repo_settings(
                    REPO_ID,
                    repo_type=REPO_TYPE,
                    private=True,
                    gated=False,
                )
                rollback = verify_exact_release_state(
                    api,
                    uploaded_sha,
                    checksums,
                    expected_private=True,
                    expected_history=[uploaded_sha, args.expected_parent],
                )
                append_phase(
                    args.state_log,
                    "public_gate_failed_rolled_back_private",
                    head=rollback["repo"]["sha"],
                    error=(f"{type(publication_error).__name__}: {publication_error}"),
                )
            except Exception as rollback_error:
                append_phase(
                    args.state_log,
                    "public_gate_failed_rollback_failed",
                    publication_error=(f"{type(publication_error).__name__}: {publication_error}"),
                    rollback_error=(f"{type(rollback_error).__name__}: {rollback_error}"),
                )
        raise

    result = {
        "schema_version": 2,
        "receipt_kind": "hf_dataset_full13_public_release",
        "completed_at": now(),
        "ok": True,
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "endpoint": private_handoff.HF_ENDPOINT,
        "credential_identity": "AlexGeek",
        "owner_decision_id": "rebuttal-rlvr-full13-public-20260729",
        "history_policy": "preserve_existing_history",
        "preserved_parent": args.expected_parent,
        "public_sha": uploaded_sha,
        "uploaded_private_state": uploaded_private,
        "authenticated_public_state": authenticated_public,
        "anonymous_public_verification": anonymous_result,
        "anonymous_receipt": str(anonymous_receipt),
        "anonymous_receipt_sha256": dataset_validator.sha256_file(anonymous_receipt),
        "anonymous_log": str(anonymous_log),
        "bundle": {
            "path": str(args.bundle),
            "file_count": bundle_result["file_count"],
            "payload_count": bundle_result["payload_count"],
            "payload_rows": bundle_result["payload_rows"],
            "inventory_sha256": EXPECTED_INVENTORY_SHA256,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "publication_format_conversion": False,
        },
        "private_preflight": preflight,
        "route_admission": final_admission,
        "observed_connection_hosts": observed_hosts,
    }
    try:
        write_json_new(args.receipt, result)
    except Exception as receipt_error:
        append_phase(
            args.state_log,
            "release_receipt_write_failed_remote_public_verified",
            public_sha=uploaded_sha,
            error=f"{type(receipt_error).__name__}: {receipt_error}",
        )
        raise FullPublicationError(
            "remote release passed authenticated and anonymous verification, "
            f"but the local receipt could not be written: {receipt_error}"
        ) from receipt_error
    append_phase(
        args.state_log,
        "release_receipt_written",
        public_sha=uploaded_sha,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--old-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-parent",
        default=EXPECTED_PRIVATE_PARENT,
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--state-log", type=Path, required=True)
    parser.add_argument("--anonymous-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-publish")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_artifact_paths(args)
        checksums, bundle_result = reviewed_public_bundle(args.bundle)
        initial_admission = private_handoff.validate_network_admission(private_handoff.admit_hf_network())
        http_observer = private_handoff.configure_hf_http_observer(initial_admission)
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise FullPublicationError("huggingface_hub is required") from exc
        api = private_handoff.GuardedHfApi(
            HfApi(endpoint=private_handoff.HF_ENDPOINT),
            initial_admission,
        )
        identity_name(api)
        if not args.apply:
            result = {
                "ok": True,
                "mode": "read_only_preflight",
                "credential_identity": "AlexGeek",
                "bundle": bundle_result,
                "remote": read_only_private_preflight(
                    api,
                    args,
                    initial_admission,
                ),
                "mutation_enabled": False,
                "history_policy": "preserve_existing_history",
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        args.state_log.parent.mkdir(parents=True, exist_ok=True)
        args.state_log.touch(mode=0o600, exist_ok=False)
        append_phase(
            args.state_log,
            "publication_started",
            expected_parent=args.expected_parent,
            history_policy="preserve_existing_history",
        )
        result = apply_publication(
            api,
            args,
            initial_admission,
            http_observer,
            checksums,
            bundle_result,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        FullPublicationError,
        public_verifier.PublicVerificationError,
        dataset_validator.ValidationError,
        private_handoff.PublishError,
        OSError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
