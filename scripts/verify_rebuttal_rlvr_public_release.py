#!/usr/bin/env python3
"""Verify the immutable RLdataset release anonymously through the audited local route."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import publish_rebuttal_rlvr_public_dataset as private_handoff
import validate_rebuttal_rlvr_dataset as dataset_validator

REPO_ID = "AlexGeek/RLdataset"
REPO_TYPE = "dataset"
EXPECTED_INVENTORY_SHA256 = "fe90ad41b1abbf08c3bbd17f9638954ba9b15b0dcf916b3edcfa62d24b95d130"
EXPECTED_MANIFEST_SHA256 = "26cc2d7395e3aceb1f71ea44e150e3a458d285591766c9ad688c44efa604d394"
EXPECTED_FILE_COUNT = 18
EXPECTED_PAYLOAD_COUNT = 13
EXPECTED_PAYLOAD_ROWS = 22860
EXPECTED_PRESERVED_PARENT = "da622cf077ca3f0eaf0ebc55dd4e115d0ebc0b9c"
README_SIMPLIFIED_PARENT = "b1c264a92ace36dace52babdda651e415d9e9f82"
README_SIMPLIFIED_MANIFEST_SHA256 = "5f702e98e6cce21949d4ca901be189fce4e3e9b556ad6827023d91b411a3b9ad"
READER_FIXED_PARENT = "3d4d0e5f1be6dad9de2613d6caf88f197ec78044"
READER_FIXED_MANIFEST_SHA256 = "62bfaed9b1530af3f504e846ef84454cf771ad9673598a9e1bbf6e8e8c8b64cd"
DEFAULT_RELEASE_PROFILE = "full13-v4"
RELEASE_PROFILES = {
    DEFAULT_RELEASE_PROFILE: {
        "inventory_sha256": EXPECTED_INVENTORY_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "preserved_revisions": [EXPECTED_PRESERVED_PARENT],
    },
    "readme-simplified-v5": {
        "inventory_sha256": EXPECTED_INVENTORY_SHA256,
        "manifest_sha256": README_SIMPLIFIED_MANIFEST_SHA256,
        "preserved_revisions": [README_SIMPLIFIED_PARENT, EXPECTED_PRESERVED_PARENT],
    },
    "readme-reader-fixed-v6": {
        "inventory_sha256": EXPECTED_INVENTORY_SHA256,
        "manifest_sha256": READER_FIXED_MANIFEST_SHA256,
        "preserved_revisions": [
            READER_FIXED_PARENT,
            README_SIMPLIFIED_PARENT,
            EXPECTED_PRESERVED_PARENT,
        ],
    },
}
TOKEN_ENV_VARS = (
    "HF_TOKEN",
    "HF_TOKEN_PATH",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
)


class PublicVerificationError(RuntimeError):
    pass


def is_commit_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def require_private_empty_home(variable: str) -> Path:
    raw = os.environ.get(variable)
    if not raw:
        raise PublicVerificationError(f"{variable} must point to a fresh private directory")
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise PublicVerificationError(f"{variable} must be an absolute existing non-symlink directory")
    metadata = path.stat()
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise PublicVerificationError(f"{variable} must be operator-owned and inaccessible to group/other")
    return path


def validate_anonymous_environment() -> dict[str, str]:
    present = [name for name in TOKEN_ENV_VARS if os.environ.get(name)]
    if present:
        raise PublicVerificationError(f"anonymous verification forbids token variables: {present}")
    if os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") != "1":
        raise PublicVerificationError("HF_HUB_DISABLE_IMPLICIT_TOKEN must equal 1")
    if os.environ.get("HF_HUB_OFFLINE") not in {None, "", "0"}:
        raise PublicVerificationError("HF_HUB_OFFLINE must be unset for public verification")
    home = require_private_empty_home("HOME")
    hf_home = require_private_empty_home("HF_HOME")
    if home == hf_home:
        raise PublicVerificationError("HOME and HF_HOME must be distinct fresh directories")
    forbidden_files = (
        hf_home / "token",
        hf_home / "stored_tokens",
        home / ".cache/huggingface/token",
        home / ".cache/huggingface/stored_tokens",
    )
    if any(path.exists() or path.is_symlink() for path in forbidden_files):
        raise PublicVerificationError("fresh anonymous homes contain a Hugging Face credential file")
    return {"home": str(home), "hf_home": str(hf_home)}


def release_profile(name: str) -> dict[str, Any]:
    try:
        return RELEASE_PROFILES[name]
    except KeyError as exc:
        raise PublicVerificationError(f"unknown release profile: {name}") from exc


def expected_bundle(
    bundle: Path,
    profile_name: str = DEFAULT_RELEASE_PROFILE,
) -> tuple[dict[str, str], dict[str, Any]]:
    if not bundle.is_absolute() or bundle.is_symlink() or not bundle.is_dir():
        raise PublicVerificationError("--bundle must be an absolute non-symlink directory")
    profile = release_profile(profile_name)
    result = dataset_validator.validate(bundle)
    if (
        result["inventory_sha256"] != profile["inventory_sha256"]
        or result["manifest_sha256"] != profile["manifest_sha256"]
        or result["file_count"] != EXPECTED_FILE_COUNT
        or result["payload_count"] != EXPECTED_PAYLOAD_COUNT
        or result["payload_rows"] != EXPECTED_PAYLOAD_ROWS
    ):
        raise PublicVerificationError(f"reviewed public bundle pin or summary drifted: {result}")
    inventory = json.loads((bundle / dataset_validator.INVENTORY_PATH).read_text(encoding="utf-8"))
    if (
        inventory.get("publication_status") != "owner_approved_for_public_release"
        or inventory.get("publication_decision", {}).get("decision") != "publish_full_13_payload_collection"
    ):
        raise PublicVerificationError("bundle does not contain the approved full-13 publication decision")
    checksums = dataset_validator.parse_checksum_manifest(bundle / dataset_validator.MANIFEST_PATH)
    checksums[dataset_validator.MANIFEST_PATH] = profile["manifest_sha256"]
    if len(checksums) != EXPECTED_FILE_COUNT:
        raise PublicVerificationError("reviewed public bundle has an unexpected exact file count")
    return checksums, result


def prove_preserved_revisions_anonymously(
    api: private_handoff.GuardedHfApi,
    revisions: list[str],
) -> list[dict[str, Any]]:
    results = []
    for revision in revisions:
        info = api.repo_info(
            REPO_ID,
            repo_type=REPO_TYPE,
            revision=revision,
            files_metadata=True,
            token=False,
        )
        if (
            info.id != REPO_ID
            or info.sha != revision
            or info.private is not False
            or getattr(info, "gated", None) is not False
        ):
            raise PublicVerificationError(f"preserved revision did not remain anonymously reachable: {revision}")
        results.append(
            {
                "revision": revision,
                "anonymous_api": "reachable",
                "file_count": len(info.siblings or []),
                "files": sorted(item.rfilename for item in (info.siblings or [])),
            }
        )
    return results


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise PublicVerificationError(f"refusing to replace receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_args(args: argparse.Namespace) -> None:
    profile_name = getattr(args, "release_profile", DEFAULT_RELEASE_PROFILE)
    profile = release_profile(profile_name)
    if not is_commit_sha(args.revision):
        raise PublicVerificationError("--revision must be an exact lowercase 40-character commit")
    if not args.local_dir.is_absolute() or args.local_dir.is_symlink() or args.local_dir.exists():
        raise PublicVerificationError("--local-dir must be a new absolute non-symlink path")
    if not args.receipt.is_absolute() or args.receipt.is_symlink() or args.receipt.exists():
        raise PublicVerificationError("--receipt must be a new absolute non-symlink path")
    try:
        args.receipt.relative_to(args.local_dir)
    except ValueError:
        pass
    else:
        raise PublicVerificationError("--receipt must live outside --local-dir")
    if len(args.preserved_revision) != len(set(args.preserved_revision)) or any(
        not is_commit_sha(revision) for revision in args.preserved_revision
    ):
        raise PublicVerificationError("--preserved-revision values must be unique lowercase commit SHAs")
    if args.preserved_revision != profile["preserved_revisions"]:
        raise PublicVerificationError(
            "--preserved-revision must match the selected release profile: "
            f"{profile['preserved_revisions']}"
        )


def verify(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    profile_name = getattr(args, "release_profile", DEFAULT_RELEASE_PROFILE)
    profile = release_profile(profile_name)
    anonymous_environment = validate_anonymous_environment()
    checksums, local_bundle_result = expected_bundle(args.bundle, profile_name)
    args.local_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{args.local_dir.name}.partial-", dir=args.local_dir.parent))

    try:
        from huggingface_hub import HfApi, hf_hub_download

        initial_admission = private_handoff.validate_network_admission(private_handoff.admit_hf_network())
        http_observer = private_handoff.configure_hf_http_observer(initial_admission)
        api = private_handoff.GuardedHfApi(
            HfApi(endpoint=private_handoff.HF_ENDPOINT, token=False),
            initial_admission,
        )
        info = api.repo_info(
            REPO_ID,
            repo_type=REPO_TYPE,
            revision=args.revision,
            files_metadata=True,
            timeout=private_handoff.HF_METADATA_TIMEOUT_SECONDS,
            token=False,
        )
        if (
            info.id != REPO_ID
            or info.sha != args.revision
            or info.private is not False
            or getattr(info, "gated", None) is not False
        ):
            raise PublicVerificationError("anonymous API did not resolve the exact public ungated revision")
        commit_ids = private_handoff.list_commit_ids(api, REPO_ID)
        expected_history = [args.revision, *args.preserved_revision]
        if commit_ids != expected_history:
            raise PublicVerificationError(
                "anonymous main history does not preserve the reviewed parent: "
                f"expected={expected_history}, observed={commit_ids}"
            )
        refs = api.list_repo_refs(
            REPO_ID,
            repo_type=REPO_TYPE,
            include_pull_requests=True,
            token=False,
        )
        branches = [
            {
                "name": getattr(item, "name", None),
                "ref": getattr(item, "ref", None),
                "target_commit": getattr(item, "target_commit", None),
            }
            for item in (getattr(refs, "branches", None) or [])
        ]
        if branches != [
            {
                "name": "main",
                "ref": "refs/heads/main",
                "target_commit": args.revision,
            }
        ]:
            raise PublicVerificationError(f"anonymous main ref does not target the release: {branches}")
        if getattr(refs, "tags", None) or getattr(refs, "pull_requests", None):
            raise PublicVerificationError("anonymous repository exposes unexpected tags or PR refs")
        remote_files = {item.rfilename: item for item in info.siblings or []}
        if set(remote_files) != set(checksums):
            raise PublicVerificationError(
                "anonymous remote file allowlist mismatch: "
                f"missing={sorted(set(checksums) - set(remote_files))}, "
                f"unexpected={sorted(set(remote_files) - set(checksums))}"
            )
        for relative in sorted(checksums):
            local_size = (args.bundle / relative).stat().st_size
            if getattr(remote_files[relative], "size", None) != local_size:
                raise PublicVerificationError(f"anonymous remote size mismatch: {relative}")
            api.guard_call(
                hf_hub_download,
                REPO_ID,
                relative,
                repo_type=REPO_TYPE,
                revision=args.revision,
                local_dir=partial,
                endpoint=private_handoff.HF_ENDPOINT,
                token=False,
            )

        downloaded_result = dataset_validator.validate(partial)
        if (
            downloaded_result["manifest_sha256"] != profile["manifest_sha256"]
            or downloaded_result["inventory_sha256"] != profile["inventory_sha256"]
            or downloaded_result["file_count"] != EXPECTED_FILE_COUNT
            or downloaded_result["payload_count"] != EXPECTED_PAYLOAD_COUNT
            or downloaded_result["payload_rows"] != EXPECTED_PAYLOAD_ROWS
        ):
            raise PublicVerificationError("anonymous downloaded bundle failed reviewed pins")
        for relative, expected in sorted(checksums.items()):
            observed = dataset_validator.sha256_file(partial / relative)
            if observed != expected:
                raise PublicVerificationError(f"anonymous file SHA-256 mismatch: {relative}")

        preserved_history = prove_preserved_revisions_anonymously(
            api,
            args.preserved_revision,
        )
        observed_hosts = http_observer.observed_hosts
        final_admission = api._admit_same_route(tuple(observed_hosts))
        if final_admission["connection_hosts_verified"] != observed_hosts:
            raise PublicVerificationError("route receipt does not bind every anonymous connection host")
        cache_dir = partial / ".cache"
        if cache_dir.is_symlink():
            raise PublicVerificationError("Hugging Face created an unexpected cache symlink")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        partial.rename(args.local_dir)
    except Exception as exc:
        raise PublicVerificationError(
            f"anonymous public verification failed; partial files remain at {partial}: {exc}"
        ) from exc

    file_hashes = {relative: dataset_validator.sha256_file(args.local_dir / relative) for relative in sorted(checksums)}
    result = {
        "schema_version": 1,
        "receipt_kind": "anonymous_hf_dataset_public_release",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "endpoint": private_handoff.HF_ENDPOINT,
        "revision": args.revision,
        "anonymous": True,
        "api_private": False,
        "api_gated": False,
        "remote_file_count": len(remote_files),
        "commit_ids": commit_ids,
        "main_ref": branches[0],
        "file_count": downloaded_result["file_count"],
        "payload_count": downloaded_result["payload_count"],
        "payload_rows": downloaded_result["payload_rows"],
        "manifest_sha256": downloaded_result["manifest_sha256"],
        "inventory_sha256": downloaded_result["inventory_sha256"],
        "file_sha256": file_hashes,
        "history_policy": "preserve_existing_history",
        "release_profile": profile_name,
        "preserved_history": preserved_history,
        "authentication": {
            "token_argument": False,
            "implicit_token_disabled": True,
            "fresh_home": anonymous_environment["home"],
            "fresh_hf_home": anonymous_environment["hf_home"],
        },
        "local_dir": str(args.local_dir),
        "reviewed_local_bundle": {
            "path": str(args.bundle),
            "manifest_sha256": local_bundle_result["manifest_sha256"],
            "inventory_sha256": local_bundle_result["inventory_sha256"],
        },
        "route_admission": final_admission,
        "observed_connection_hosts": observed_hosts,
    }
    write_json_new(args.receipt, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-profile",
        choices=sorted(RELEASE_PROFILES),
        default=DEFAULT_RELEASE_PROFILE,
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--preserved-revision", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = verify(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        PublicVerificationError,
        dataset_validator.ValidationError,
        private_handoff.PublishError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
