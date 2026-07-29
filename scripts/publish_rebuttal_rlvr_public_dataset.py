#!/usr/bin/env python3
"""Atomically replace the audited HF dataset HEAD after explicit admission."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any


DEFAULT_REPO_ID = "beichenhang/EnsembleLLM-data"
REPO_TYPE = "dataset"
CHUNK_SIZE = 1024 * 1024


class PublishError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: str) -> Path:
    if (
        not value
        or value.startswith("/")
        or value.endswith("/")
        or "//" in value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise PublishError(f"unsafe checksum path: {value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise PublishError(f"unsafe checksum path: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts:
        raise PublishError(f"unsafe checksum path: {value!r}")
    return Path(*pure.parts)


def load_and_verify_bundle(bundle: Path) -> dict[str, str]:
    if not bundle.is_absolute() or bundle.is_symlink() or not bundle.is_dir():
        raise PublishError("bundle must be an absolute, non-symlink directory")
    required = {
        "README.md",
        ".gitattributes",
        "metadata/publication_inventory.json",
        "metadata/checksums.sha256",
    }
    actual_files = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    missing = sorted(required.difference(actual_files))
    if missing:
        raise PublishError(f"bundle is missing required files: {missing}")
    symlinks = sorted(str(path) for path in bundle.rglob("*") if path.is_symlink())
    if symlinks:
        raise PublishError(f"bundle contains symlinks: {symlinks}")

    inventory = json.loads((bundle / "metadata/publication_inventory.json").read_text(encoding="utf-8"))
    if inventory.get("schema_version") != 1:
        raise PublishError("bundle inventory schema_version must be 1")
    if inventory.get("layout_version") != "rebuttal-rlvr-dataset-v1":
        raise PublishError("bundle has an unexpected layout_version")

    declared_files: set[str] = set()

    def register_declared(items: Any, label: str, prefix: str) -> None:
        if not isinstance(items, list):
            raise PublishError(f"bundle inventory {label} must be a list")
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("path_in_repo"), str):
                raise PublishError(f"bundle inventory {label} has an invalid path entry")
            relative = item["path_in_repo"]
            safe_relative(relative)
            if not relative.startswith(prefix):
                raise PublishError(f"bundle inventory {label} path must be below {prefix}: {relative}")
            if relative in declared_files or relative in required:
                raise PublishError(f"duplicate or reserved declared bundle path: {relative}")
            declared_files.add(relative)

    register_declared(inventory.get("public_assets"), "public_assets", "data/")
    register_declared(inventory.get("license_files"), "license_files", "LICENSES/")
    register_declared(inventory.get("processing_files"), "processing_files", "processing/")
    expected_files = required | declared_files
    if actual_files != expected_files:
        raise PublishError(
            f"bundle file allowlist mismatch: missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )

    checksums: dict[str, str] = {}
    for line in (bundle / "metadata/checksums.sha256").read_text(encoding="utf-8").splitlines():
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise PublishError(f"invalid checksum line: {line!r}") from exc
        path = bundle / safe_relative(relative)
        if not path.is_file() or path.is_symlink():
            raise PublishError(f"checksum target is not a regular file: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise PublishError(f"bundle checksum mismatch: {relative}")
        checksums[relative] = expected

    expected_checksum_targets = actual_files - {"metadata/checksums.sha256"}
    if set(checksums) != expected_checksum_targets:
        raise PublishError("checksums.sha256 does not cover the complete bundle")
    checksums["metadata/checksums.sha256"] = sha256_file(bundle / "metadata/checksums.sha256")
    return checksums


def repo_snapshot(info: Any) -> dict[str, Any]:
    return {
        "id": info.id,
        "sha": info.sha,
        "private": info.private,
        "gated": getattr(info, "gated", None),
        "last_modified": info.last_modified.isoformat() if info.last_modified else None,
        "files": [
            {
                "path": item.rfilename,
                "size": item.size,
                "lfs": bool(item.lfs),
            }
            for item in sorted(info.siblings or [], key=lambda value: value.rfilename)
        ],
    }


def verify_remote(api: Any, repo_id: str, revision: str, bundle: Path, checksums: dict[str, str]) -> None:
    from huggingface_hub import hf_hub_download

    info = api.repo_info(repo_id, repo_type=REPO_TYPE, revision=revision, files_metadata=True)
    expected_names = set(checksums)
    remote_names = {item.rfilename for item in info.siblings or []}
    if remote_names != expected_names:
        raise PublishError(
            f"remote file set mismatch: missing={sorted(expected_names - remote_names)}, "
            f"unexpected={sorted(remote_names - expected_names)}"
        )
    for relative, expected_sha in sorted(checksums.items()):
        downloaded = Path(
            hf_hub_download(
                repo_id,
                relative,
                repo_type=REPO_TYPE,
                revision=revision,
            )
        )
        if sha256_file(downloaded) != expected_sha:
            raise PublishError(f"remote SHA-256 mismatch: {relative}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--expected-parent", required=True, help="Current 40-hex remote HEAD.")
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--confirm-repo-id",
        help="Required with --apply and must exactly equal --repo-id.",
    )
    parser.add_argument(
        "--make-ungated-public",
        action="store_true",
        help="After remote byte verification, enforce private=false and gated=false.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if len(args.expected_parent) != 40 or any(ch not in "0123456789abcdef" for ch in args.expected_parent):
            raise PublishError("--expected-parent must be a lowercase 40-hex commit")
        if not args.receipt_dir.is_absolute():
            raise PublishError("--receipt-dir must be absolute")
        checksums = load_and_verify_bundle(args.bundle)

        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise PublishError("huggingface_hub is required") from exc
        api = HfApi()
        info = api.repo_info(args.repo_id, repo_type=REPO_TYPE, files_metadata=True)
        if info.id != args.repo_id:
            raise PublishError(f"resolved unexpected repository: {info.id}")
        if info.sha != args.expected_parent:
            raise PublishError(f"remote HEAD drifted: expected {args.expected_parent}, observed {info.sha}")
        try:
            api.auth_check(args.repo_id, repo_type=REPO_TYPE, write=True)
        except Exception as exc:
            raise PublishError(f"current Hugging Face credential has no write access to {args.repo_id}") from exc

        preflight = {
            "ok": True,
            "mode": "preflight",
            "repo": repo_snapshot(info),
            "bundle_file_count": len(checksums),
        }
        if not args.apply:
            print(json.dumps(preflight, indent=2, sort_keys=True))
            return 0

        if args.confirm_repo_id != args.repo_id:
            raise PublishError("--confirm-repo-id must exactly equal --repo-id")
        if not args.make_ungated_public:
            raise PublishError("--apply requires --make-ungated-public for this handoff")
        if args.receipt_dir.exists():
            raise PublishError("receipt directory already exists; use a new attempt directory")
        args.receipt_dir.mkdir(parents=True)
        (args.receipt_dir / "pre_publish_remote.json").write_text(
            json.dumps(repo_snapshot(info), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        commit = api.upload_folder(
            repo_id=args.repo_id,
            repo_type=REPO_TYPE,
            folder_path=args.bundle,
            delete_patterns="*",
            parent_commit=args.expected_parent,
            commit_message="Replace legacy data with licensed RLVR public bundle",
            commit_description=(
                "Atomic HEAD replacement. Restricted competition/test assets are intentionally excluded; "
                "see metadata/publication_inventory.json."
            ),
        )
        revision = commit.oid
        verify_remote(api, args.repo_id, revision, args.bundle, checksums)
        api.update_repo_settings(
            args.repo_id,
            repo_type=REPO_TYPE,
            private=False,
            gated=False,
        )
        final_info = api.repo_info(args.repo_id, repo_type=REPO_TYPE, revision=revision, files_metadata=True)
        if final_info.private or getattr(final_info, "gated", None) is not False:
            raise PublishError("repository settings did not converge to ungated public")
        receipt = {
            "schema_version": 1,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "repo_id": args.repo_id,
            "parent_commit": args.expected_parent,
            "published_commit": revision,
            "commit_url": commit.commit_url,
            "verified_file_sha256": checksums,
            "final_remote": repo_snapshot(final_info),
            "history_note": "Old files were removed from the new HEAD but remain in repository history unless the owner performs a separate history purge.",
        }
        receipt_path = args.receipt_dir / "publish_receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"ok": True, "commit": revision, "receipt": str(receipt_path)}, sort_keys=True))
        return 0
    except (PublishError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
