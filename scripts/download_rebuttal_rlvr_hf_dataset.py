#!/usr/bin/env python3
"""Download the pinned RLVR dataset through the admitted large-traffic route."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
from typing import Any

try:
    from hf_large_traffic_route import (
        HF_ENDPOINT,
        HF_METADATA_TIMEOUT_SECONDS,
        HF_ROUTE_REQUIRED_HOSTS,
        RouteAdmissionError,
        admit_hf_network,
        configure_hf_http_observer,
        validate_route_admission,
    )
except ModuleNotFoundError as exc:
    if exc.name != "hf_large_traffic_route":
        raise
    from scripts.hf_large_traffic_route import (
        HF_ENDPOINT,
        HF_METADATA_TIMEOUT_SECONDS,
        HF_ROUTE_REQUIRED_HOSTS,
        RouteAdmissionError,
        admit_hf_network,
        configure_hf_http_observer,
        validate_route_admission,
    )


DEFAULT_REPO_ID = "AlexGeek/RLdataset"
VERIFIED_DATASET_COMMIT = "da622cf077ca3f0eaf0ebc55dd4e115d0ebc0b9c"
REPO_TYPE = "dataset"
MANIFEST_PATH = "metadata/checksums.sha256"
INVENTORY_PATH = "metadata/publication_inventory.json"
REVIEWED_PRIVATE_INVENTORY_SHA256 = "b5b646a28b2e6bf8a6f531f986d921fbc20e5dc7c454453c3c7ce12a2674aa5a"
REVIEWED_CHECKSUMS_SHA256 = "5e35eab998946be30857425525641b72bc7a1937376f7b797b45d47e71a73a59"
CHUNK_SIZE = 1024 * 1024
AMBIENT_TOKEN_VARIABLES = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
)


class DownloadError(RuntimeError):
    pass


def require_official_hf_endpoint() -> None:
    if HF_ENDPOINT != "https://huggingface.co":
        raise DownloadError("downloader endpoint pin drifted from https://huggingface.co")


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
        raise DownloadError(f"unsafe dataset path: {value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise DownloadError(f"unsafe dataset path: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts:
        raise DownloadError(f"unsafe dataset path: {value!r}")
    return Path(*pure.parts)


def parse_checksum_manifest(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise DownloadError(f"invalid checksum line: {line!r}") from exc
        safe_relative(relative)
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            raise DownloadError(f"invalid SHA-256 in checksum line: {line!r}")
        if relative in checksums:
            raise DownloadError(f"duplicate checksum path: {relative}")
        checksums[relative] = expected
    return checksums


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def admit_same_route(
    expected_leaf: str | None = None,
    *,
    connection_hosts: tuple[str, ...] = (),
) -> dict[str, Any]:
    require_official_hf_endpoint()
    admission = validate_route_admission(
        admit_hf_network(connection_hosts=connection_hosts)
        if connection_hosts
        else admit_hf_network()
    )
    selected = admission.get("selected_leaf_sha256")
    if not isinstance(selected, str) or len(selected) != 64:
        raise DownloadError("route admission did not return a valid selected-leaf digest")
    if expected_leaf is not None and selected != expected_leaf:
        raise DownloadError("large-traffic selector changed during the dataset transfer")
    return admission


def guarded_hub_call(
    expected_leaf: str,
    operation: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Bracket every Hub request with live admission, including error paths."""
    admit_same_route(expected_leaf)
    operation_error: BaseException | None = None
    try:
        return operation(*args, **kwargs)
    except BaseException as exc:
        operation_error = exc
        raise
    finally:
        try:
            admit_same_route(expected_leaf)
        except Exception as route_exc:
            if operation_error is not None:
                raise DownloadError(
                    "Hub operation failed and post-operation route admission also failed: "
                    f"operation={type(operation_error).__name__}; route={type(route_exc).__name__}: {route_exc}"
                ) from route_exc
            raise


def verify_download(root: Path) -> dict[str, str]:
    manifest = root / MANIFEST_PATH
    inventory = root / INVENTORY_PATH
    if not manifest.is_file() or manifest.is_symlink():
        raise DownloadError("downloaded checksum manifest is missing or is a symlink")
    if not inventory.is_file() or inventory.is_symlink():
        raise DownloadError("downloaded publication inventory is missing or is a symlink")
    if sha256_file(manifest) != REVIEWED_CHECKSUMS_SHA256:
        raise DownloadError("downloaded checksum manifest does not match the reviewed pin")
    if sha256_file(inventory) != REVIEWED_PRIVATE_INVENTORY_SHA256:
        raise DownloadError("downloaded publication inventory does not match the reviewed pin")

    checksums = parse_checksum_manifest(manifest)
    checksums[MANIFEST_PATH] = REVIEWED_CHECKSUMS_SHA256
    expected = set(checksums)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise DownloadError(
            f"downloaded file allowlist mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    symlinks = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_symlink()
    ]
    if symlinks:
        raise DownloadError(f"downloaded dataset contains symlinks: {sorted(symlinks)}")
    for relative, expected_sha in sorted(checksums.items()):
        path = root / safe_relative(relative)
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha:
            raise DownloadError(f"downloaded SHA-256 mismatch: {relative}")
    return checksums


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=VERIFIED_DATASET_COMMIT)
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    return parser.parse_args()


def validate_private_credential_home() -> str:
    ambient = [name for name in AMBIENT_TOKEN_VARIABLES if os.environ.get(name)]
    if ambient:
        raise DownloadError(
            "ambient Hugging Face token variables are forbidden; use only the operator-owned HF_HOME token file"
        )
    hf_home_value = os.environ.get("HF_HOME")
    if not hf_home_value:
        raise DownloadError("private download requires an explicit operator-owned HF_HOME")
    hf_home = Path(hf_home_value)
    if (
        not hf_home.is_absolute()
        or hf_home.is_symlink()
        or not hf_home.is_dir()
        or hf_home.resolve() != hf_home
    ):
        raise DownloadError("HF_HOME must be an existing absolute non-symlink directory")
    home_stat = hf_home.stat()
    if home_stat.st_uid != os.geteuid() or stat.S_IMODE(home_stat.st_mode) & 0o077:
        raise DownloadError("HF_HOME must be owned by the current operator and inaccessible to group/other")

    token_path = hf_home / "token"
    configured_token_path = os.environ.get("HF_TOKEN_PATH")
    if configured_token_path and Path(configured_token_path) != token_path:
        raise DownloadError("HF_TOKEN_PATH must resolve to the token file inside the selected HF_HOME")
    if (
        not token_path.is_file()
        or token_path.is_symlink()
        or token_path.resolve() != token_path
        or token_path.stat().st_uid != os.geteuid()
        or stat.S_IMODE(token_path.stat().st_mode) & 0o077
    ):
        raise DownloadError("HF_HOME/token must be an operator-owned 0600-style regular file")
    token = token_path.read_text(encoding="utf-8").strip()
    if not token or not token.startswith("hf_") or any(character.isspace() for character in token):
        raise DownloadError("HF_HOME/token is not a valid single-line Hugging Face token")
    os.environ["HF_TOKEN_PATH"] = str(token_path)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    return token


def validate_args(args: argparse.Namespace) -> None:
    require_official_hf_endpoint()
    if args.repo_id != DEFAULT_REPO_ID:
        raise DownloadError(f"--repo-id must remain pinned to {DEFAULT_REPO_ID}")
    if args.revision != VERIFIED_DATASET_COMMIT:
        raise DownloadError(f"--revision must remain pinned to {VERIFIED_DATASET_COMMIT}")
    if not args.local_dir.is_absolute() or args.local_dir.is_symlink():
        raise DownloadError("--local-dir must be an absolute, non-symlink path")
    if args.local_dir.exists():
        raise DownloadError("--local-dir already exists; refusing to mix a pinned download with existing files")
    if not args.receipt.is_absolute() or args.receipt.is_symlink() or args.receipt.exists():
        raise DownloadError("--receipt must be a new absolute, non-symlink path")
    try:
        args.receipt.relative_to(args.local_dir)
    except ValueError:
        pass
    else:
        raise DownloadError("--receipt must live outside --local-dir")
    configured_endpoint = os.environ.get("HF_ENDPOINT")
    if configured_endpoint and configured_endpoint.rstrip("/") != HF_ENDPOINT:
        raise DownloadError(f"HF_ENDPOINT override is forbidden: expected {HF_ENDPOINT}")
    args.hf_token = validate_private_credential_home()


def validate_remote_visibility(info: Any) -> None:
    private = getattr(info, "private", None)
    if private is not True:
        raise DownloadError("the pinned private_handoff_only bundle is unexpectedly public")
    if getattr(info, "gated", None) is not False:
        raise DownloadError("the pinned repository must remain ungated")


def download(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    token = args.hf_token
    initial_route = admit_same_route()
    selected_leaf = initial_route["selected_leaf_sha256"]
    args.local_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{args.local_dir.name}.partial-", dir=args.local_dir.parent))

    try:
        from huggingface_hub import HfApi, hf_hub_download

        http_observer = configure_hf_http_observer(initial_route)

        def fetch(relative: str) -> Path:
            downloaded = Path(
                guarded_hub_call(
                    selected_leaf,
                    hf_hub_download,
                    args.repo_id,
                    relative,
                    repo_type=REPO_TYPE,
                    revision=args.revision,
                    local_dir=partial,
                    endpoint=HF_ENDPOINT,
                    token=token,
                )
            )
            return downloaded

        manifest = fetch(MANIFEST_PATH)
        if sha256_file(manifest) != REVIEWED_CHECKSUMS_SHA256:
            raise DownloadError("remote checksum manifest does not match the reviewed pin")
        checksums = parse_checksum_manifest(manifest)
        expected_files = set(checksums) | {MANIFEST_PATH}

        api = HfApi(endpoint=HF_ENDPOINT, token=token)
        info = guarded_hub_call(
            selected_leaf,
            api.repo_info,
            args.repo_id,
            repo_type=REPO_TYPE,
            revision=args.revision,
            files_metadata=True,
            timeout=HF_METADATA_TIMEOUT_SECONDS,
        )
        remote_files = {item.rfilename for item in info.siblings or []}
        if info.sha != args.revision or remote_files != expected_files:
            raise DownloadError("immutable remote revision or exact file allowlist does not match the reviewed bundle")
        validate_remote_visibility(info)

        for relative in sorted(checksums):
            if relative != MANIFEST_PATH:
                fetch(relative)
        cache_dir = partial / ".cache"
        if cache_dir.is_symlink():
            raise DownloadError("huggingface_hub created an unexpected symlink cache")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        verified = verify_download(partial)
        observed_hosts = http_observer.observed_hosts
        final_route = admit_same_route(
            selected_leaf,
            connection_hosts=tuple(observed_hosts),
        )
        if final_route["connection_hosts_verified"] != observed_hosts:
            raise DownloadError("final route admission does not bind every observed connection host")
        partial.rename(args.local_dir)
    except Exception as exc:
        raise DownloadError(f"dataset download failed; partial files remain at {partial}: {exc}") from exc

    receipt = {
        "schema_version": 2,
        "receipt_kind": "pinned_hf_dataset_download",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "repo_type": REPO_TYPE,
        "revision": args.revision,
        "local_dir": str(args.local_dir),
        "remote_private": bool(info.private),
        "remote_gated": getattr(info, "gated", None),
        "file_count": len(verified),
        "inventory_sha256": REVIEWED_PRIVATE_INVENTORY_SHA256,
        "checksum_manifest_sha256": REVIEWED_CHECKSUMS_SHA256,
        "endpoint": HF_ENDPOINT,
        "authentication_mode": "explicit_operator_hf_home_token_private_only",
        "route_admission": final_route,
        "observed_connection_hosts": observed_hosts,
    }
    write_json(args.receipt, receipt)
    return receipt


def main() -> int:
    args = parse_args()
    try:
        result = download(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (DownloadError, RouteAdmissionError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
