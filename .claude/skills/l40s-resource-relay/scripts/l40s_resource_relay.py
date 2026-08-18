#!/usr/bin/env python3
"""Fail-closed L40S-to-L40S resource transfer through a private rsync relay."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import itertools
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import NoReturn, Sequence


TRANSFER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
ENV_KEYS = (
    "L40S_RELAY_HOST",
    "L40S_RELAY_PORT",
    "L40S_RELAY_USER",
    "L40S_RELAY_STAGING_ROOT",
    "L40S_RELAY_IDENTITY_FILE",
    "L40S_RELAY_KNOWN_HOSTS_FILE",
    "L40S_RELAY_RECEIPT_ROOT",
)
PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in (
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "100.64.0.0/10",
        "fc00::/7",
    )
)


class RelayError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    user: str
    staging_root: str
    identity_file: Path
    known_hosts_file: Path
    receipt_root: Path

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fail(message: str) -> NoReturn:
    raise RelayError(message)


def default_config_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return base / "verl" / "l40s-resource-relay.env"


def parse_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        fail(f"private config not found: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        fail(f"private config must not be group/world accessible: {path} mode={mode:o}")
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            fail(f"invalid config line {line_number}: expected KEY=VALUE")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if key not in ENV_KEYS:
            fail(f"unsupported config key on line {line_number}: {key}")
        try:
            tokens = shlex.split(raw_value, comments=True, posix=True)
        except ValueError as exc:
            fail(f"invalid quoting on config line {line_number}: {exc}")
        if len(tokens) != 1:
            fail(f"config line {line_number} must contain exactly one literal value")
        values[key] = tokens[0]
    return values


def require_private_numeric_ip(value: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        fail("L40S_RELAY_HOST must be a numeric private IP; hostnames and public IPs are forbidden")
    if not any(address in network for network in PRIVATE_NETWORKS if address.version == network.version):
        fail("L40S_RELAY_HOST is outside the allowed private address ranges")
    return str(address)


def require_absolute_remote_path(value: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or value == "/" or ".." in path.parts:
        fail("L40S_RELAY_STAGING_ROOT must be an absolute, non-root path without '..'")
    return str(path)


def require_private_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        fail(f"{label} not found: {resolved}")
    mode = stat.S_IMODE(resolved.stat().st_mode)
    if mode & 0o077:
        fail(f"{label} must have mode 600 or stricter: {resolved} mode={mode:o}")
    return resolved


def load_config(path: Path) -> Config:
    values = parse_env_file(path)
    for key in ENV_KEYS:
        if key in os.environ:
            values[key] = os.environ[key]
    missing = [key for key in ENV_KEYS if not values.get(key)]
    if missing:
        fail("missing private configuration: " + ", ".join(missing))
    host = require_private_numeric_ip(values["L40S_RELAY_HOST"])
    try:
        port = int(values["L40S_RELAY_PORT"])
    except ValueError:
        fail("L40S_RELAY_PORT must be an integer")
    if not 1 <= port <= 65535:
        fail("L40S_RELAY_PORT must be between 1 and 65535")
    user = values["L40S_RELAY_USER"]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,31}", user):
        fail("L40S_RELAY_USER has an unsafe format")
    staging_root = require_absolute_remote_path(values["L40S_RELAY_STAGING_ROOT"])
    identity = require_private_file(Path(values["L40S_RELAY_IDENTITY_FILE"]), "identity file")
    known_hosts = require_private_file(
        Path(values["L40S_RELAY_KNOWN_HOSTS_FILE"]), "known-hosts file"
    )
    receipt_root = Path(values["L40S_RELAY_RECEIPT_ROOT"]).expanduser()
    if not receipt_root.is_absolute():
        fail("L40S_RELAY_RECEIPT_ROOT must be absolute")
    return Config(host, port, user, staging_root, identity, known_hosts, receipt_root)


def run(command: Sequence[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError as exc:
        fail(f"required command not found: {command[0]}")
    except subprocess.CalledProcessError as exc:
        detail = ""
        if capture:
            detail = (exc.stderr or exc.stdout or "").strip()
        fail(f"command failed ({exc.returncode}): {command[0]}{': ' + detail if detail else ''}")


def assert_inside_tmux() -> None:
    if not os.environ.get("TMUX"):
        fail("push and pull must run inside tmux")


def validate_gpu_names(names: list[str]) -> list[str]:
    if not names or any(name.strip().upper() != "NVIDIA L40S" for name in names):
        fail(f"endpoint rejected: all local GPUs must be NVIDIA L40S; observed={names or ['none']}")
    return names


def assert_local_l40s() -> list[str]:
    result = run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture=True
    )
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return validate_gpu_names(names)


def validate_transfer_id(value: str) -> str:
    if not TRANSFER_ID_RE.fullmatch(value):
        fail("transfer ID must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}")
    return value


def ssh_base(config: Config) -> list[str]:
    return [
        "ssh",
        "-p",
        str(config.port),
        "-i",
        str(config.identity_file),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={config.known_hosts_file}",
        "-o",
        "ConnectTimeout=10",
    ]


def rsync_shell(config: Config) -> str:
    return shlex.join(ssh_base(config))


def remote_command(config: Config, arguments: Sequence[str], *, capture: bool = False):
    return run([*ssh_base(config), config.target, shlex.join(arguments)], capture=capture)


def remote_path(config: Config, transfer_id: str, suffix: str = "") -> str:
    base = PurePosixPath(config.staging_root) / transfer_id
    return str(base / suffix) if suffix else str(base)


def remote_spec(config: Config, path: str) -> str:
    return f"{config.target}:{shlex.quote(path)}"


def rsync_base(config: Config) -> list[str]:
    return [
        "rsync",
        "-aH",
        "--numeric-ids",
        "--partial",
        "--append-verify",
        "--protect-args",
        "--human-readable",
        "--info=progress2,stats2",
        "-e",
        rsync_shell(config),
    ]


def ensure_tools() -> None:
    for command in ("rsync", "ssh", "nvidia-smi"):
        if shutil.which(command) is None:
            fail(f"required command not found: {command}")


def assert_regular_tree(source: Path) -> None:
    if source == Path("/dev") or Path("/dev") in source.parents:
        fail("raw devices and /dev paths are forbidden")
    paths = itertools.chain([source], source.rglob("*")) if source.is_dir() else iter([source])
    for path in paths:
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
            continue
        fail(f"special filesystem object is forbidden: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(source: Path, transfer_id: str) -> dict:
    source = source.resolve(strict=True)
    assert_regular_tree(source)
    root_parent = source.parent
    candidates = [source]
    if source.is_dir():
        candidates.extend(sorted(source.rglob("*"), key=lambda item: item.as_posix()))
    entries = []
    total_bytes = 0
    regular_files = 0
    for path in candidates:
        relative = path.relative_to(root_parent).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            entries.append({"path": relative, "type": "symlink", "target": os.readlink(path)})
        elif stat.S_ISDIR(mode):
            entries.append({"path": relative, "type": "directory"})
        else:
            size = path.stat().st_size
            entries.append(
                {"path": relative, "type": "file", "size": size, "sha256": sha256_file(path)}
            )
            total_bytes += size
            regular_files += 1
    manifest = {
        "schema": "l40s-resource-relay-manifest-v1",
        "transfer_id": transfer_id,
        "source_name": source.name,
        "source_kind": "directory" if source.is_dir() else "file",
        "regular_file_count": regular_files,
        "total_bytes": total_bytes,
        "entries": entries,
    }
    validate_manifest(manifest, transfer_id)
    return manifest


def canonical_json(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def manifest_digest(payload: dict) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def validate_manifest(payload: object, expected_transfer_id: str) -> dict:
    if not isinstance(payload, dict):
        fail("relay manifest must be a JSON object")
    if payload.get("schema") != "l40s-resource-relay-manifest-v1":
        fail("relay manifest schema mismatch")
    if payload.get("transfer_id") != expected_transfer_id:
        fail("relay manifest transfer ID mismatch")
    source_name = payload.get("source_name")
    if (
        not isinstance(source_name, str)
        or source_name in ("", ".", "..")
        or PurePosixPath(source_name).name != source_name
        or "/" in source_name
        or "\\" in source_name
        or "\x00" in source_name
    ):
        fail("relay manifest source_name must be one safe basename")
    source_kind = payload.get("source_kind")
    if source_kind not in ("file", "directory"):
        fail("relay manifest source_kind must be file or directory")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        fail("relay manifest entries must be a non-empty list")
    observed_paths: set[str] = set()
    directory_paths: set[str] = set()
    symlink_paths: set[str] = set()
    regular_files = 0
    total_bytes = 0
    root_types: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            fail("relay manifest entry must be an object")
        path_value = entry.get("path")
        entry_type = entry.get("type")
        if not isinstance(path_value, str) or not path_value or "\x00" in path_value:
            fail("relay manifest entry path must be a non-empty string")
        relative = PurePosixPath(path_value)
        if (
            relative.is_absolute()
            or any(part in ("", ".", "..") for part in relative.parts)
            or relative.parts[0] != source_name
            or "\\" in path_value
        ):
            fail(f"relay manifest contains an unsafe path: {path_value}")
        canonical_path = relative.as_posix()
        if canonical_path != path_value or canonical_path in observed_paths:
            fail(f"relay manifest contains a duplicate/non-canonical path: {path_value}")
        if entry_type == "directory":
            directory_paths.add(canonical_path)
        elif entry_type == "symlink":
            target = entry.get("target")
            if not isinstance(target, str) or "\x00" in target:
                fail(f"relay manifest symlink target is invalid: {path_value}")
            symlink_paths.add(canonical_path)
        elif entry_type == "file":
            size = entry.get("size")
            checksum = entry.get("sha256")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                fail(f"relay manifest file size is invalid: {path_value}")
            if not isinstance(checksum, str) or not SHA256_RE.fullmatch(checksum):
                fail(f"relay manifest file checksum is invalid: {path_value}")
            regular_files += 1
            total_bytes += size
        else:
            fail(f"relay manifest entry type is invalid: {entry_type}")
        observed_paths.add(canonical_path)
        if canonical_path == source_name:
            root_types.append(entry_type)
    expected_root_type = "directory" if source_kind == "directory" else "file"
    if root_types != [expected_root_type]:
        fail("relay manifest must contain exactly one correctly typed source root")
    if source_kind == "file" and len(entries) != 1:
        fail("a file manifest must contain only its source root entry")
    for path_value in observed_paths:
        relative = PurePosixPath(path_value)
        for parent in relative.parents:
            parent_value = parent.as_posix()
            if parent_value in (".", source_name):
                break
            if parent_value in symlink_paths:
                fail(f"relay manifest path descends through a symlink: {path_value}")
            if parent_value not in directory_paths:
                fail(f"relay manifest omits a parent directory: {parent_value}")
    if payload.get("regular_file_count") != regular_files:
        fail("relay manifest regular_file_count mismatch")
    if payload.get("total_bytes") != total_bytes:
        fail("relay manifest total_bytes mismatch")
    return payload


def validate_expected_digest(value: str) -> str:
    if not SHA256_RE.fullmatch(value):
        fail("expected manifest SHA256 must be exactly 64 lowercase hexadecimal characters")
    return value


def require_manifest_digest(manifest: dict, expected_digest: str) -> str:
    digest = manifest_digest(manifest)
    if digest != expected_digest:
        fail("relay manifest does not match the source-provided manifest SHA256")
    return digest


def ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        fail(f"private output directory must have mode 700 or stricter: {path} mode={mode:o}")


def write_private_json(path: Path, payload: dict) -> None:
    ensure_private_directory(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            fail(f"no existing parent for destination: {path}")
        candidate = candidate.parent
    return candidate


def available_bytes(path: Path) -> int:
    return shutil.disk_usage(nearest_existing_parent(path)).free


def require_capacity(available: int, required: int, label: str) -> None:
    reserve = max(1024 * 1024 * 1024, required // 20)
    if available < required + reserve:
        fail(f"insufficient {label} capacity: need={required + reserve} available={available}")


def remote_available_bytes(config: Config) -> int:
    code = "import shutil,sys; print(shutil.disk_usage(sys.argv[1]).free)"
    result = remote_command(
        config, ["python3", "-c", code, config.staging_root], capture=True
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        fail("relay capacity query returned an invalid value")


def preflight(config: Config) -> None:
    ensure_tools()
    names = assert_local_l40s()
    remote_command(config, ["test", "-d", config.staging_root])
    free = remote_available_bytes(config)
    print(json.dumps({"status": "pass", "local_gpus": names, "relay_free_bytes": free}))


def remote_manifest_exists(config: Config, transfer_id: str) -> bool:
    code = (
        "import pathlib,sys; root=pathlib.Path(sys.argv[1]).resolve(); "
        "target=root/sys.argv[2]; "
        "target.exists() or sys.exit(1); "
        "(not target.is_symlink() and target.is_dir() and target.resolve().parent == root) "
        "or sys.exit(3); manifest=target/'manifest.json'; "
        "not manifest.is_symlink() or sys.exit(3); sys.exit(0 if manifest.is_file() else 1)"
    )
    result = subprocess.run(
        [
            *ssh_base(config),
            config.target,
            shlex.join(["python3", "-c", code, config.staging_root, transfer_id]),
        ],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode not in (0, 1):
        fail("could not inspect relay manifest state")
    return result.returncode == 0


def ensure_remote_transfer_directory(config: Config, transfer_id: str) -> None:
    code = (
        "import pathlib,sys; root=pathlib.Path(sys.argv[1]).resolve(strict=True); "
        "target=root/sys.argv[2]; "
        "(not target.exists()) and target.mkdir(mode=0o700); "
        "(not target.is_symlink() and target.is_dir() and target.resolve().parent == root) "
        "or sys.exit('unsafe transfer directory'); payload=target/'payload'; "
        "(not payload.exists()) and payload.mkdir(mode=0o700); "
        "(not payload.is_symlink() and payload.is_dir() and payload.resolve().parent == target.resolve()) "
        "or sys.exit('unsafe payload directory')"
    )
    remote_command(config, ["python3", "-c", code, config.staging_root, transfer_id])


def rsync_manifest_from_relay(config: Config, transfer_id: str, local_path: Path) -> dict:
    run(
        [
            *rsync_base(config),
            remote_spec(config, remote_path(config, transfer_id, "manifest.json")),
            str(local_path),
        ]
    )
    try:
        payload = json.loads(local_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"invalid relay manifest: {exc}")
    return validate_manifest(payload, transfer_id)


def verify_rsync_source_to_relay(config: Config, source: Path, transfer_id: str) -> None:
    command = [
        "rsync",
        "-aHnci",
        "--numeric-ids",
        "--delete",
        "--protect-args",
        "--out-format=%i|%n%L",
        "-e",
        rsync_shell(config),
        str(source),
        remote_spec(config, remote_path(config, transfer_id, "payload")) + "/",
    ]
    result = run(command, capture=True)
    changes = [line for line in result.stdout.splitlines() if line.strip()]
    if changes:
        fail("source-to-relay checksum verification found changes: " + changes[0])


def verify_local_manifest(destination: Path, manifest: dict) -> None:
    expected_paths = set()
    for entry in manifest["entries"]:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            fail("manifest contains an unsafe relative path")
        path = destination.joinpath(*relative.parts)
        expected_paths.add(relative.as_posix())
        if entry["type"] == "directory":
            if not path.is_dir() or path.is_symlink():
                fail(f"directory verification failed: {relative}")
        elif entry["type"] == "symlink":
            if not path.is_symlink() or os.readlink(path) != entry["target"]:
                fail(f"symlink verification failed: {relative}")
        elif entry["type"] == "file":
            if not path.is_file() or path.is_symlink():
                fail(f"file verification failed: {relative}")
            if path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
                fail(f"checksum verification failed: {relative}")
        else:
            fail(f"unsupported manifest entry type: {entry.get('type')}")
    source_root = destination / manifest["source_name"]
    observed = {source_root.relative_to(destination).as_posix()}
    if source_root.is_dir() and not source_root.is_symlink():
        observed.update(path.relative_to(destination).as_posix() for path in source_root.rglob("*"))
    if observed != expected_paths:
        extras = sorted(observed - expected_paths)
        missing = sorted(expected_paths - observed)
        fail(f"inventory mismatch: extra={extras[:3]} missing={missing[:3]}")


def source_receipt_path(config: Config, transfer_id: str) -> Path:
    return config.receipt_root / f"{transfer_id}.source.json"


def target_receipt_path(config: Config, transfer_id: str) -> Path:
    return config.receipt_root / f"{transfer_id}.target.json"


def pulling_receipt_path(config: Config, transfer_id: str) -> Path:
    return config.receipt_root / f"{transfer_id}.pulling.json"


def push(config: Config, transfer_id: str, source: Path) -> None:
    assert_inside_tmux()
    ensure_tools()
    assert_local_l40s()
    source = source.expanduser().resolve(strict=True)
    manifest = build_manifest(source, transfer_id)
    digest = manifest_digest(manifest)
    remote_command(config, ["test", "-d", config.staging_root])
    require_capacity(remote_available_bytes(config), manifest["total_bytes"], "relay")
    with tempfile.TemporaryDirectory(prefix="l40s-relay-") as temporary_dir:
        temporary = Path(temporary_dir)
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(canonical_json(manifest))
        if remote_manifest_exists(config, transfer_id):
            existing = rsync_manifest_from_relay(config, transfer_id, temporary / "existing.json")
            if manifest_digest(existing) != digest:
                fail("completed relay transfer ID already exists with a different manifest")
        ensure_remote_transfer_directory(config, transfer_id)
        run(
            [
                *rsync_base(config),
                str(source),
                remote_spec(config, remote_path(config, transfer_id, "payload")) + "/",
            ]
        )
        verify_rsync_source_to_relay(config, source, transfer_id)
        run(
            [
                *rsync_base(config),
                str(manifest_path),
                remote_spec(config, remote_path(config, transfer_id, "manifest.json")),
            ]
        )
    receipt = {
        "schema": "l40s-resource-relay-receipt-v1",
        "status": "relay_verified",
        "transfer_id": transfer_id,
        "manifest_sha256": digest,
        "verified_at": utc_now(),
        "regular_file_count": manifest["regular_file_count"],
        "total_bytes": manifest["total_bytes"],
    }
    path = source_receipt_path(config, transfer_id)
    write_private_json(path, receipt)
    print(json.dumps({"status": "relay_verified", "receipt": str(path), **receipt}))


def pull(
    config: Config, transfer_id: str, expected_manifest_sha256: str, destination: Path
) -> None:
    assert_inside_tmux()
    ensure_tools()
    assert_local_l40s()
    destination = destination.expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        fail("destination must be a directory path")
    with tempfile.TemporaryDirectory(prefix="l40s-relay-") as temporary_dir:
        temporary = Path(temporary_dir)
        if not remote_manifest_exists(config, transfer_id):
            fail("relay transfer is incomplete or missing its verified manifest")
        manifest = rsync_manifest_from_relay(config, transfer_id, temporary / "manifest.json")
        digest = require_manifest_digest(manifest, expected_manifest_sha256)
        pulling_path = pulling_receipt_path(config, transfer_id)
        if destination.exists() and any(destination.iterdir()):
            try:
                pulling_receipt = json.loads(pulling_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                fail(f"non-empty destination is not a registered resume: {exc}")
            expected_resume = {
                "status": "pulling",
                "transfer_id": transfer_id,
                "manifest_sha256": digest,
                "destination": str(destination),
            }
            if any(pulling_receipt.get(key) != value for key, value in expected_resume.items()):
                fail("non-empty destination resume receipt does not match this transfer")
        else:
            destination.mkdir(parents=True, exist_ok=True)
            write_private_json(
                pulling_path,
                {
                    "schema": "l40s-resource-relay-receipt-v1",
                    "status": "pulling",
                    "transfer_id": transfer_id,
                    "manifest_sha256": digest,
                    "destination": str(destination),
                    "started_at": utc_now(),
                },
            )
        require_capacity(available_bytes(destination), manifest["total_bytes"], "destination")
        run(
            [
                *rsync_base(config),
                remote_spec(config, remote_path(config, transfer_id, "payload")) + "/",
                str(destination) + "/",
            ]
        )
        verify_local_manifest(destination, manifest)
    receipt = {
        "schema": "l40s-resource-relay-receipt-v1",
        "status": "target_verified",
        "transfer_id": transfer_id,
        "manifest_sha256": digest,
        "verified_at": utc_now(),
        "destination": str(destination / manifest["source_name"]),
        "regular_file_count": manifest["regular_file_count"],
        "total_bytes": manifest["total_bytes"],
    }
    path = target_receipt_path(config, transfer_id)
    write_private_json(path, receipt)
    pulling_receipt_path(config, transfer_id).unlink(missing_ok=True)
    print(json.dumps({"status": "target_verified", "receipt": str(path), **receipt}))


def status(config: Config, transfer_id: str) -> None:
    ensure_tools()
    assert_local_l40s()
    exists = remote_manifest_exists(config, transfer_id)
    print(json.dumps({"transfer_id": transfer_id, "completed_manifest": exists}))


def cleanup(config: Config, transfer_id: str, confirmation: str) -> None:
    ensure_tools()
    assert_local_l40s()
    if confirmation != transfer_id:
        fail("cleanup confirmation must exactly match the transfer ID")
    receipt_path = target_receipt_path(config, transfer_id)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"valid target receipt required before cleanup: {exc}")
    if receipt.get("status") != "target_verified" or receipt.get("transfer_id") != transfer_id:
        fail("target receipt is not a verified receipt for this transfer")
    with tempfile.TemporaryDirectory(prefix="l40s-relay-") as temporary_dir:
        manifest = rsync_manifest_from_relay(
            config, transfer_id, Path(temporary_dir) / "manifest.json"
        )
    if manifest_digest(manifest) != receipt.get("manifest_sha256"):
        fail("relay manifest no longer matches the target receipt")
    code = (
        "import pathlib,shutil,sys; "
        "root=pathlib.Path(sys.argv[1]).resolve(strict=True); candidate=root/sys.argv[2]; "
        "(candidate.exists() and not candidate.is_symlink() and candidate.is_dir()) "
        "or sys.exit('unsafe cleanup target'); target=candidate.resolve(); "
        "target.parent == root and target != root or sys.exit('unsafe cleanup target'); "
        "shutil.rmtree(target)"
    )
    remote_command(
        config,
        ["python3", "-c", code, config.staging_root, transfer_id],
    )
    print(json.dumps({"status": "relay_staging_deleted", "transfer_id": transfer_id}))


def self_test() -> None:
    accepted = ["10.1.2.3", "172.16.0.1", "192.168.5.4", "100.64.0.1", "fd00::1"]
    for value in accepted:
        assert require_private_numeric_ip(value) == str(ipaddress.ip_address(value))
    rejected = ["203.0.113.10", "8.8.8.8", "relay.example.com", "127.0.0.1", "::1"]
    for value in rejected:
        try:
            require_private_numeric_ip(value)
        except RelayError:
            pass
        else:
            raise AssertionError(f"public/non-routable value accepted: {value}")
    for value in ("ok", "transfer-01", "x.y_z"):
        assert validate_transfer_id(value) == value
    for value in ("../bad", "/bad", "bad space", ""):
        try:
            validate_transfer_id(value)
        except RelayError:
            pass
        else:
            raise AssertionError(f"unsafe transfer ID accepted: {value}")
    assert validate_expected_digest("a" * 64) == "a" * 64
    for value in ("A" * 64, "a" * 63, "not-a-digest"):
        try:
            validate_expected_digest(value)
        except RelayError:
            pass
        else:
            raise AssertionError(f"unsafe digest accepted: {value}")
    assert validate_gpu_names(["NVIDIA L40S", "NVIDIA L40S"])
    for names in ([], ["NVIDIA A800"], ["NVIDIA L40S", "NVIDIA A800"]):
        try:
            validate_gpu_names(names)
        except RelayError:
            pass
        else:
            raise AssertionError(f"non-L40S endpoint accepted: {names}")
    with tempfile.TemporaryDirectory(prefix="l40s-relay-self-test-") as temporary_dir:
        root = Path(temporary_dir)
        source = root / "payload"
        source.mkdir()
        (source / "a.txt").write_text("alpha\n", encoding="utf-8")
        (source / "empty").mkdir()
        (source / "link").symlink_to("a.txt")
        manifest = build_manifest(source, "self-test")
        digest = manifest_digest(manifest)
        assert require_manifest_digest(manifest, digest) == digest
        try:
            require_manifest_digest(manifest, "0" * 64)
        except RelayError:
            pass
        else:
            raise AssertionError("out-of-band digest canary did not fail")
        verify_local_manifest(root, manifest)
        assert manifest["regular_file_count"] == 1
        assert manifest["total_bytes"] == 6
        (source / "a.txt").write_text("broken\n", encoding="utf-8")
        try:
            verify_local_manifest(root, manifest)
        except RelayError:
            pass
        else:
            raise AssertionError("checksum canary did not fail")
        malformed = dict(manifest)
        malformed["source_name"] = ".."
        try:
            validate_manifest(malformed, "self-test")
        except RelayError:
            pass
        else:
            raise AssertionError("malformed manifest canary did not fail")
        malformed = json.loads(json.dumps(manifest))
        malformed["regular_file_count"] = 2
        try:
            validate_manifest(malformed, "self-test")
        except RelayError:
            pass
        else:
            raise AssertionError("manifest count canary did not fail")
    print(
        json.dumps(
            {
                "status": "pass",
                "checks": "private-ip,l40s,path,manifest,out-of-band-digest,checksum-canary",
            }
        )
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=default_config_path())
    subparsers = result.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    push_parser = subparsers.add_parser("push")
    push_parser.add_argument("--transfer-id", required=True)
    push_parser.add_argument("--source", required=True, type=Path)
    pull_parser = subparsers.add_parser("pull")
    pull_parser.add_argument("--transfer-id", required=True)
    pull_parser.add_argument("--expected-manifest-sha256", required=True)
    pull_parser.add_argument("--destination", required=True, type=Path)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--transfer-id", required=True)
    cleanup_parser = subparsers.add_parser("cleanup")
    cleanup_parser.add_argument("--transfer-id", required=True)
    cleanup_parser.add_argument("--confirm-transfer-id", required=True)
    subparsers.add_parser("self-test")
    return result


def main() -> int:
    arguments = parser().parse_args()
    try:
        if arguments.command == "self-test":
            self_test()
            return 0
        config = load_config(arguments.config)
        transfer_id = validate_transfer_id(arguments.transfer_id) if hasattr(arguments, "transfer_id") else ""
        if arguments.command == "preflight":
            preflight(config)
        elif arguments.command == "push":
            push(config, transfer_id, arguments.source)
        elif arguments.command == "pull":
            pull(
                config,
                transfer_id,
                validate_expected_digest(arguments.expected_manifest_sha256),
                arguments.destination,
            )
        elif arguments.command == "status":
            status(config, transfer_id)
        elif arguments.command == "cleanup":
            cleanup(config, transfer_id, arguments.confirm_transfer_id)
        return 0
    except RelayError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
