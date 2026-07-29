#!/usr/bin/env python3
"""Read-only verifier for the completed private RLVR dataset handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Callable

try:
    from hf_large_traffic_route import (
        DEFAULT_MIHOMO_RUNTIME_CONFIG,
        HF_ENDPOINT,
        HF_METADATA_TIMEOUT_SECONDS,
        HF_ROUTE_GROUP,
        HF_ROUTE_REQUIRED_HOSTS,
        PROXY_URL,
        RESIDENTIAL_NODE_MARKERS,
        RouteAdmissionError,
        admit_hf_network as _admit_hf_network,
        configure_hf_http_observer as _configure_hf_http_observer,
        enforce_hf_proxy as _enforce_hf_proxy,
        first_domain_route as _first_domain_route,
        mihomo_controller_json as _mihomo_controller_json,
        validate_hf_route_admission as _validate_hf_route_admission,
        validate_route_admission as _validate_route_admission,
    )
except ModuleNotFoundError as exc:
    if exc.name != "hf_large_traffic_route":
        raise
    from scripts.hf_large_traffic_route import (
        DEFAULT_MIHOMO_RUNTIME_CONFIG,
        HF_ENDPOINT,
        HF_METADATA_TIMEOUT_SECONDS,
        HF_ROUTE_GROUP,
        HF_ROUTE_REQUIRED_HOSTS,
        PROXY_URL,
        RESIDENTIAL_NODE_MARKERS,
        RouteAdmissionError,
        admit_hf_network as _admit_hf_network,
        configure_hf_http_observer as _configure_hf_http_observer,
        enforce_hf_proxy as _enforce_hf_proxy,
        first_domain_route as _first_domain_route,
        mihomo_controller_json as _mihomo_controller_json,
        validate_hf_route_admission as _validate_hf_route_admission,
        validate_route_admission as _validate_route_admission,
    )


DEFAULT_REPO_ID = "AlexGeek/RLdataset"
REPO_TYPE = "dataset"
EMPTY_PARENT = "EMPTY"
CHUNK_SIZE = 1024 * 1024
REVIEWED_BUNDLE_PURPOSE = "RLVR mathematical and code-task complete private handoff candidate"
REVIEWED_PRIVATE_INVENTORY_SHA256 = "b5b646a28b2e6bf8a6f531f986d921fbc20e5dc7c454453c3c7ce12a2674aa5a"
REVIEWED_CHECKSUMS_SHA256 = "5e35eab998946be30857425525641b72bc7a1937376f7b797b45d47e71a73a59"
EXPECTED_PAYLOAD_COUNTS = {
    "file_count": 13,
    "math_training_files": 1,
    "math_evaluation_files": 7,
    "code_training_files": 1,
    "code_evaluation_files": 4,
}
REQUIRED_PURGED_LEGACY_PROBES = (
    ("04668b0284dbef3f5aad51bf570a46416d09287d", ".gitattributes"),
    ("df19d512f4306aee8c3abce0387f968b09a5b234", ".gitattributes"),
    ("bf278c1214db9ea8d3b26e41fb33dca1520a9b07", ".gitattributes"),
    ("989d5c978c46fb2b2c5c7b242f0b53e95e91ddd2", ".gitattributes"),
)
SAFE_BOOTSTRAP_REVISION = "685c0fcb81f225158f83dacd8ae99c647010bed0"
SAFE_BOOTSTRAP_GITATTRIBUTES_SHA256 = "9e75dd981de037ec3769f24f790e126bc5a160b6871f510214e68dc70649aeeb"
SAFE_UPLOADED_REVISION = "e0d4f9ea24081e654c33d522ba6b4eed1a82c5a3"
COMPLETED_PRIVATE_REVISION = "da622cf077ca3f0eaf0ebc55dd4e115d0ebc0b9c"
CURRENT_BUNDLE_PUBLIC_TRANSITION_BLOCK = (
    "the reviewed bundle is private_handoff_only; build and privately verify a new "
    "public-reviewed bundle and immutable commit before changing repository visibility"
)


class PublishError(RuntimeError):
    pass


def require_official_hf_endpoint() -> None:
    if HF_ENDPOINT != "https://huggingface.co":
        raise PublishError("publisher endpoint pin drifted from https://huggingface.co")


def _translate_route_error(operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return operation(*args, **kwargs)
    except RouteAdmissionError as exc:
        raise PublishError(str(exc)) from exc


def enforce_hf_proxy() -> None:
    require_official_hf_endpoint()
    _translate_route_error(_enforce_hf_proxy)


def mihomo_controller_json(base_url: str, secret: str, path: str) -> dict[str, Any]:
    return _translate_route_error(_mihomo_controller_json, base_url, secret, path)


def first_domain_route(rules: list[Any], hostname: str) -> str | None:
    return _translate_route_error(_first_domain_route, rules, hostname)


def validate_hf_route_admission(
    config_path: Path | None = None,
    *,
    fetch_json: Callable[[str, str, str], dict[str, Any]] = mihomo_controller_json,
    connection_hosts: tuple[str, ...] = (),
) -> dict[str, Any]:
    return _translate_route_error(
        _validate_hf_route_admission,
        config_path,
        fetch_json=fetch_json,
        connection_hosts=connection_hosts,
    )


def admit_hf_network(*, connection_hosts: tuple[str, ...] = ()) -> dict[str, Any]:
    require_official_hf_endpoint()
    if connection_hosts:
        return _translate_route_error(_admit_hf_network, connection_hosts=connection_hosts)
    return _translate_route_error(_admit_hf_network)


def configure_hf_http_observer(initial_admission: dict[str, Any], **kwargs: Any) -> Any:
    return _translate_route_error(_configure_hf_http_observer, initial_admission, **kwargs)


def validate_network_admission(admission: Any) -> dict[str, Any]:
    return _translate_route_error(_validate_route_admission, admission)


class GuardedHfApi:
    """Route-admit every Hub API call against one initial selector leaf."""

    def __init__(self, api: Any, initial_admission: dict[str, Any]) -> None:
        self._api = api
        self._selected_leaf = validate_network_admission(initial_admission)["selected_leaf_sha256"]

    def _admit_same_route(self, connection_hosts: tuple[str, ...] = ()) -> dict[str, Any]:
        admission = validate_network_admission(
            admit_hf_network(connection_hosts=connection_hosts)
            if connection_hosts
            else admit_hf_network()
        )
        if admission["selected_leaf_sha256"] != self._selected_leaf:
            raise PublishError("large-traffic selector changed during the Hugging Face operation")
        return admission

    def guard_call(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._admit_same_route()
        try:
            return operation(*args, **kwargs)
        finally:
            self._admit_same_route()

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._api, name)
        if not callable(attribute):
            return attribute

        def guarded(*args: Any, **kwargs: Any) -> Any:
            if name == "repo_info":
                kwargs.setdefault("timeout", HF_METADATA_TIMEOUT_SECONDS)
            return self.guard_call(attribute, *args, **kwargs)

        return guarded


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
        raise PublishError(f"unsafe bundle path: {value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise PublishError(f"unsafe bundle path: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts:
        raise PublishError(f"unsafe bundle path: {value!r}")
    return Path(*pure.parts)


def parse_checksum_manifest(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise PublishError(f"invalid checksum line: {line!r}") from exc
        safe_relative(relative)
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            raise PublishError(f"invalid SHA-256 in checksum line: {line!r}")
        if relative in checksums:
            raise PublishError(f"duplicate checksum path: {relative}")
        checksums[relative] = expected
    return checksums


def load_and_verify_bundle(bundle: Path) -> dict[str, str]:
    if not bundle.is_absolute() or bundle.is_symlink() or not bundle.is_dir():
        raise PublishError("bundle must be an absolute, non-symlink directory")
    required = {
        ".gitattributes",
        "README.md",
        "metadata/publication_inventory.json",
        "metadata/checksums.sha256",
    }
    actual_files = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    symlinks = sorted(str(path) for path in bundle.rglob("*") if path.is_symlink())
    if symlinks:
        raise PublishError(f"bundle contains symlinks: {symlinks}")
    missing = sorted(required - actual_files)
    if missing:
        raise PublishError(f"bundle is missing required files: {missing}")

    inventory_path = bundle / "metadata/publication_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("schema_version") != 1:
        raise PublishError("bundle inventory schema_version must be 1")
    if inventory.get("bundle_purpose") != REVIEWED_BUNDLE_PURPOSE:
        raise PublishError("bundle has an unexpected purpose")
    if inventory.get("layout_version") != "meituan-handoff-data-v1":
        raise PublishError("bundle has an unexpected layout version")
    if inventory.get("publication_status") != "private_handoff_only":
        raise PublishError("full-scope bundle must remain marked private_handoff_only")
    summary = inventory.get("payload_summary")
    if not isinstance(summary, dict) or any(summary.get(key) != value for key, value in EXPECTED_PAYLOAD_COUNTS.items()):
        raise PublishError("bundle payload summary does not match Math train + Math-7 + code train + Code-4")
    items = inventory.get("files")
    if not isinstance(items, list) or len(items) != EXPECTED_PAYLOAD_COUNTS["file_count"]:
        raise PublishError("bundle inventory must declare exactly 13 payload files")

    declared: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("relative_path"), str):
            raise PublishError("bundle inventory has an invalid file entry")
        relative = item["relative_path"]
        safe_relative(relative)
        if not relative.startswith("data/") or relative in declared:
            raise PublishError(f"invalid or duplicate dataset payload path: {relative}")
        declared[relative] = item

    expected_files = required | set(declared)
    if actual_files != expected_files:
        raise PublishError(
            f"bundle file allowlist mismatch: missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )
    manifest_checksums = parse_checksum_manifest(bundle / "metadata/checksums.sha256")
    manifest_targets = set(declared) | {
        ".gitattributes",
        "README.md",
        "metadata/publication_inventory.json",
    }
    if set(manifest_checksums) != manifest_targets:
        raise PublishError("checksums.sha256 must cover .gitattributes, README, inventory, and all 13 payload files")

    for relative, expected in sorted(manifest_checksums.items()):
        path = bundle / safe_relative(relative)
        if not path.is_file() or path.is_symlink():
            raise PublishError(f"manifest target is not a regular file: {relative}")
        if sha256_file(path) != expected:
            raise PublishError(f"bundle SHA-256 mismatch: {relative}")
    for relative, item in sorted(declared.items()):
        expected = manifest_checksums[relative]
        if item.get("sha256") != expected:
            raise PublishError(f"inventory/checksum disagreement: {relative}")
        path = bundle / safe_relative(relative)
        if path.stat().st_size != item.get("bytes"):
            raise PublishError(f"bundle size mismatch: {relative}")

    # The manifest covers every file except itself; add its reviewed hash so
    # immutable-revision remote verification covers the exact 17-file tree.
    checksums = dict(manifest_checksums)
    checksums["metadata/checksums.sha256"] = sha256_file(bundle / "metadata/checksums.sha256")
    return checksums


def verify_reviewed_bundle(bundle: Path) -> None:
    inventory_path = bundle / "metadata/publication_inventory.json"
    checksums_path = bundle / "metadata/checksums.sha256"
    if sha256_file(inventory_path) != REVIEWED_PRIVATE_INVENTORY_SHA256:
        raise PublishError("bundle publication inventory does not match the reviewed full-scope inventory")
    if sha256_file(checksums_path) != REVIEWED_CHECKSUMS_SHA256:
        raise PublishError("bundle checksum manifest does not match the reviewed full-scope bundle")


def repo_snapshot(info: Any) -> dict[str, Any]:
    def file_snapshot(item: Any) -> dict[str, Any]:
        lfs = getattr(item, "lfs", None)
        return {
            "path": item.rfilename,
            "size": item.size,
            "blob_id": getattr(item, "blob_id", None),
            "lfs": (
                {
                    "sha256": getattr(lfs, "sha256", None),
                    "size": getattr(lfs, "size", None),
                    "pointer_size": getattr(lfs, "pointer_size", None),
                }
                if lfs
                else None
            ),
        }

    modified = getattr(info, "last_modified", None)
    return {
        "id": info.id,
        "sha": getattr(info, "sha", None),
        "private": info.private,
        "gated": getattr(info, "gated", None),
        "last_modified": modified.isoformat() if modified else None,
        "files": [file_snapshot(item) for item in sorted(info.siblings or [], key=lambda value: value.rfilename)],
    }


def guarded_external_call(api: Any, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    guard_call = getattr(api, "guard_call", None)
    if not callable(guard_call):
        raise PublishError("every Hugging Face request requires GuardedHfApi")
    return guard_call(operation, *args, **kwargs)


def download_repo_file_sha256(api: Any, repo_id: str, relative: str, revision: str) -> str:
    from huggingface_hub import hf_hub_download

    downloaded = Path(
        guarded_external_call(
            api,
            hf_hub_download,
            repo_id,
            relative,
            repo_type=REPO_TYPE,
            revision=revision,
            endpoint=HF_ENDPOINT,
        )
    )
    return sha256_file(downloaded)


def verify_remote(api: Any, repo_id: str, revision: str, checksums: dict[str, str]) -> None:
    info = api.repo_info(repo_id, repo_type=REPO_TYPE, revision=revision, files_metadata=True)
    expected_names = set(checksums)
    remote_names = {item.rfilename for item in info.siblings or []}
    if remote_names != expected_names:
        raise PublishError(
            f"remote file set mismatch: missing={sorted(expected_names - remote_names)}, "
            f"unexpected={sorted(remote_names - expected_names)}"
        )
    for relative, expected_sha in sorted(checksums.items()):
        if download_repo_file_sha256(api, repo_id, relative, revision) != expected_sha:
            raise PublishError(f"remote SHA-256 mismatch: {relative}")


def is_commit_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(ch in "0123456789abcdef" for ch in value)


def commit_id(value: Any) -> str:
    result = getattr(value, "commit_id", None) or getattr(value, "oid", None)
    if not is_commit_sha(result):
        raise PublishError("Hugging Face returned an invalid commit id")
    return result


def list_commit_ids(api: Any, repo_id: str, *, revision: str | None = None) -> list[str]:
    kwargs: dict[str, Any] = {"repo_type": REPO_TYPE}
    if revision is not None:
        kwargs["revision"] = revision
    return [commit_id(item) for item in api.list_repo_commits(repo_id, **kwargs)]


def validate_single_main_ref(
    api: Any,
    repo_id: str,
    checksums: dict[str, str],
    *,
    allow_empty: bool,
) -> dict[str, Any]:
    refs = api.list_repo_refs(repo_id, repo_type=REPO_TYPE, include_pull_requests=True)

    def names(label: str) -> list[str]:
        return [item.name for item in (getattr(refs, label, None) or [])]

    converts = list(getattr(refs, "converts", None) or [])
    snapshot = {
        "branches": names("branches"),
        "converts": [
            {
                "name": getattr(item, "name", None),
                "ref": getattr(item, "ref", None),
                "target_commit": getattr(item, "target_commit", None),
            }
            for item in converts
        ],
        "tags": names("tags"),
        "pull_requests": names("pull_requests"),
    }
    allowed_branch_sets = ([], ["main"]) if allow_empty else (["main"],)
    if snapshot["branches"] not in allowed_branch_sets or snapshot["tags"] or snapshot["pull_requests"]:
        raise PublishError(f"history squash requires only the main branch and no other refs: {snapshot}")
    if len(converts) > 1:
        raise PublishError(f"only the automatic parquet convert ref is allowed: {snapshot}")
    if converts:
        convert = converts[0]
        target = getattr(convert, "target_commit", None)
        if (
            getattr(convert, "name", None) != "parquet"
            or getattr(convert, "ref", None) != "refs/convert/parquet"
            or not is_commit_sha(target)
        ):
            raise PublishError(f"only the exact refs/convert/parquet auto-convert ref is allowed: {snapshot}")
        convert_commits = list_commit_ids(api, repo_id, revision="refs/convert/parquet")
        snapshot["converts"][0]["commit_ids"] = convert_commits
        if convert_commits != [target]:
            raise PublishError(
                "refs/convert/parquet must contain exactly its one root target commit: "
                f"target={target}, observed={convert_commits}"
            )
        convert_info = api.repo_info(
            repo_id,
            repo_type=REPO_TYPE,
            revision="refs/convert/parquet",
            files_metadata=True,
        )
        if convert_info.id != repo_id or convert_info.sha != target:
            raise PublishError(
                "refs/convert/parquet metadata does not resolve to its advertised target commit"
            )
        convert_files = {item.rfilename: item for item in convert_info.siblings or []}
        if ".gitattributes" not in convert_files:
            raise PublishError("refs/convert/parquet is missing the reviewed .gitattributes")
        expected_attributes = checksums.get(".gitattributes")
        if expected_attributes is None:
            raise PublishError("reviewed bundle checksums are missing .gitattributes")
        if (
            download_repo_file_sha256(api, repo_id, ".gitattributes", "refs/convert/parquet")
            != expected_attributes
        ):
            raise PublishError("refs/convert/parquet has an unreviewed .gitattributes")

        reviewed_payload_oids = {
            digest for path, digest in checksums.items() if path.endswith(".parquet")
        }
        actual_payload_oids: set[str] = set()
        for path, item in convert_files.items():
            if path == ".gitattributes":
                if getattr(item, "lfs", None) is not None:
                    raise PublishError("refs/convert/parquet stores .gitattributes as an unexpected LFS object")
                continue
            lfs = getattr(item, "lfs", None)
            oid = getattr(lfs, "sha256", None) if lfs is not None else None
            if not path.endswith(".parquet") or not isinstance(oid, str):
                raise PublishError(f"refs/convert/parquet contains a non-LFS parquet payload: {path}")
            if oid not in reviewed_payload_oids:
                raise PublishError(f"refs/convert/parquet contains an unreviewed LFS object: {path}")
            actual_payload_oids.add(oid)
        if not actual_payload_oids:
            raise PublishError("refs/convert/parquet contains no reviewed payload")
        snapshot["converts"][0]["payload_lfs_sha256"] = sorted(actual_payload_oids)
    return snapshot


def parent_matches(info: Any, expected_parent: str) -> bool:
    observed = getattr(info, "sha", None)
    return observed in {None, ""} if expected_parent == EMPTY_PARENT else observed == expected_parent


def validate_private_parent(
    api: Any,
    info: Any,
    repo_id: str,
    expected_parent: str,
    checksums: dict[str, str],
) -> tuple[list[str], dict[str, Any]]:
    if info.id != repo_id:
        raise PublishError(f"resolved unexpected repository: {info.id}")
    if not parent_matches(info, expected_parent):
        raise PublishError(f"remote HEAD drifted: expected {expected_parent}, observed {getattr(info, 'sha', None)}")
    if not info.private:
        raise PublishError("repository must remain private during upload, verification, and history squash")
    commits = list_commit_ids(api, repo_id)
    if expected_parent == EMPTY_PARENT:
        if commits or info.siblings:
            raise PublishError("EMPTY parent requires a freshly recreated repository with no commits or files")
    elif not commits or commits[0] != expected_parent:
        raise PublishError(f"commit history is not rooted at the expected HEAD: {commits}")
    refs = validate_single_main_ref(
        api,
        repo_id,
        checksums,
        allow_empty=expected_parent == EMPTY_PARENT,
    )
    return commits, refs


def is_not_found_error(exc: Exception) -> bool:
    try:
        from huggingface_hub.errors import EntryNotFoundError, RevisionNotFoundError

        if isinstance(exc, (EntryNotFoundError, RevisionNotFoundError)):
            return True
    except ImportError:
        pass
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 404


def assert_old_revisions_unreachable(api: Any, repo_id: str, probes: list[dict[str, str]]) -> list[dict[str, str]]:
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    results: list[dict[str, str]] = []
    for probe in probes:
        revision = probe["revision"]
        path = probe["path"]
        try:
            api.repo_info(repo_id, repo_type=REPO_TYPE, revision=revision, files_metadata=False)
        except Exception as exc:
            if not is_not_found_error(exc):
                raise PublishError(f"could not prove old revision unreachable through API: {revision}: {exc}") from exc
        else:
            raise PublishError(f"old revision is still reachable through API after squash: {revision}")

        try:
            guarded_external_call(
                api,
                get_hf_file_metadata,
                hf_hub_url(
                    repo_id,
                    path,
                    repo_type=REPO_TYPE,
                    revision=revision,
                    endpoint=HF_ENDPOINT,
                ),
                token=getattr(api, "token", None),
            )
        except Exception as exc:
            if not is_not_found_error(exc):
                raise PublishError(f"could not prove old revision unreachable through resolve: {revision}: {exc}") from exc
        else:
            raise PublishError(f"old revision is still reachable through resolve after squash: {revision}")
        results.append({"revision": revision, "api": "not_found", "resolve": "not_found", "probe_path": path})
    return results


def optional_revision_info(api: Any, repo_id: str, revision: str) -> Any | None:
    try:
        return api.repo_info(
            repo_id,
            repo_type=REPO_TYPE,
            revision=revision,
            files_metadata=True,
        )
    except Exception as exc:
        if is_not_found_error(exc):
            return None
        raise PublishError(f"could not classify retained revision {revision}: {exc}") from exc


def verify_retained_revision(
    api: Any,
    repo_id: str,
    revision: str,
    checksums: dict[str, str],
    allowed_commit_ids: tuple[tuple[str, ...], ...],
    role: str,
) -> dict[str, Any]:
    info = optional_revision_info(api, repo_id, revision)
    if info is None:
        return {"revision": revision, "role": role, "status": "not_found"}
    if info.id != repo_id or info.sha != revision:
        raise PublishError(f"retained {role} revision resolved to an unexpected repository or commit")
    commits = list_commit_ids(api, repo_id, revision=revision)
    if tuple(commits) not in allowed_commit_ids:
        raise PublishError(f"retained {role} revision has unreviewed ancestry: {commits}")
    verify_remote(api, repo_id, revision, checksums)
    return {
        "revision": revision,
        "role": role,
        "status": "verified_safe",
        "commit_ids": commits,
        "file_count": len(checksums),
    }


def verify_post_squash_history_policy(
    api: Any,
    repo_id: str,
    checksums: dict[str, str],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    legacy_probes = [
        {"revision": revision, "path": path}
        for revision, path in REQUIRED_PURGED_LEGACY_PROBES
    ]
    legacy_unreachable = assert_old_revisions_unreachable(api, repo_id, legacy_probes)
    bootstrap = verify_retained_revision(
        api,
        repo_id,
        SAFE_BOOTSTRAP_REVISION,
        {".gitattributes": SAFE_BOOTSTRAP_GITATTRIBUTES_SHA256},
        ((SAFE_BOOTSTRAP_REVISION,),),
        "bootstrap",
    )
    uploaded = verify_retained_revision(
        api,
        repo_id,
        SAFE_UPLOADED_REVISION,
        checksums,
        (
            (SAFE_UPLOADED_REVISION,),
            (SAFE_UPLOADED_REVISION, SAFE_BOOTSTRAP_REVISION),
        ),
        "uploaded_bundle",
    )
    if (
        uploaded.get("status") == "verified_safe"
        and SAFE_BOOTSTRAP_REVISION in uploaded.get("commit_ids", [])
        and bootstrap.get("status") != "verified_safe"
    ):
        raise PublishError("retained uploaded bundle depends on an unverified bootstrap revision")
    return legacy_unreachable, [bootstrap, uploaded]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only verifier for the completed private RLVR dataset revision; "
            "this command has no upload, deletion, history-rewrite, or visibility mutation path."
        )
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=COMPLETED_PRIVATE_REVISION)
    return parser.parse_args()


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.repo_id != DEFAULT_REPO_ID:
        raise PublishError(f"--repo-id must remain pinned to {DEFAULT_REPO_ID}")
    if args.revision != COMPLETED_PRIVATE_REVISION:
        raise PublishError(
            f"--revision must remain pinned to completed private revision {COMPLETED_PRIVATE_REVISION}"
        )


def completed_private_preflight(
    api: GuardedHfApi,
    args: argparse.Namespace,
    checksums: dict[str, str],
    initial_admission: dict[str, Any],
) -> dict[str, Any]:
    info = api.repo_info(
        args.repo_id,
        repo_type=REPO_TYPE,
        revision=args.revision,
        files_metadata=True,
        timeout=HF_METADATA_TIMEOUT_SECONDS,
    )
    commits, refs = validate_private_parent(
        api,
        info,
        args.repo_id,
        args.revision,
        checksums,
    )
    if commits != [args.revision]:
        raise PublishError(
            f"completed private repository must expose one root commit: {commits}"
        )
    if getattr(info, "gated", None) is not False:
        raise PublishError("completed private repository must remain ungated")
    verify_remote(api, args.repo_id, args.revision, checksums)
    legacy_unreachable, retained_verification = verify_post_squash_history_policy(
        api,
        args.repo_id,
        checksums,
    )
    final_admission = api._admit_same_route()
    if final_admission["selected_leaf_sha256"] != initial_admission["selected_leaf_sha256"]:
        raise PublishError("large-traffic selector changed during completed-state verification")
    return {
        "ok": True,
        "mode": "completed_private_read_only_preflight",
        "repo": repo_snapshot(info),
        "commit_ids": commits,
        "refs": refs,
        "verified_file_sha256": checksums,
        "legacy_revision_unreachability": legacy_unreachable,
        "retained_revision_verification": retained_verification,
        "inventory_sha256": REVIEWED_PRIVATE_INVENTORY_SHA256,
        "checksums_sha256": REVIEWED_CHECKSUMS_SHA256,
        "network_admission": final_admission,
        "mutation_enabled": False,
        "public_transition_enabled": False,
        "public_transition_block": CURRENT_BUNDLE_PUBLIC_TRANSITION_BLOCK,
    }


def main() -> int:
    args = parse_args()
    try:
        validate_cli_args(args)
        initial_admission = validate_network_admission(admit_hf_network())
        checksums = load_and_verify_bundle(args.bundle)
        verify_reviewed_bundle(args.bundle)
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise PublishError("huggingface_hub is required") from exc
        http_observer = configure_hf_http_observer(initial_admission)
        api = GuardedHfApi(HfApi(endpoint=HF_ENDPOINT), initial_admission)
        result = completed_private_preflight(api, args, checksums, initial_admission)
        observed_hosts = http_observer.observed_hosts
        final_admission = api._admit_same_route(tuple(observed_hosts))
        if final_admission["connection_hosts_verified"] != observed_hosts:
            raise PublishError("final route admission does not bind every observed connection host")
        result["network_admission"] = final_admission
        result["observed_connection_hosts"] = observed_hosts
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (PublishError, RouteAdmissionError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
