#!/usr/bin/env python3
"""Validate a downloaded RLVR dataset bundle without assuming its location."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

MANIFEST_PATH = "metadata/checksums.sha256"
INVENTORY_PATH = "metadata/publication_inventory.json"
IGNORED_DIRECTORY_PREFIXES = (".git/", ".cache/huggingface/")
CORE_COLUMNS = ("data_source", "ability", "reward_model", "prompt", "extra_info")
STRING_TYPES = {"string", "large_string"}
CHUNK_SIZE = 1024 * 1024


class ValidationError(RuntimeError):
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
        raise ValidationError(f"unsafe relative path: {value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValidationError(f"unsafe relative path: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts:
        raise ValidationError(f"unsafe relative path: {value!r}")
    return Path(*pure.parts)


def parse_checksum_manifest(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValidationError(f"cannot read checksum manifest: {path}") from exc
    if not lines:
        raise ValidationError("checksum manifest is empty")
    for line in lines:
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValidationError(f"invalid checksum line: {line!r}") from exc
        safe_relative(relative)
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            raise ValidationError(f"invalid SHA-256 in checksum line: {line!r}")
        if relative == MANIFEST_PATH:
            raise ValidationError("checksums.sha256 cannot contain its own digest")
        if relative in checksums:
            raise ValidationError(f"duplicate checksum path: {relative}")
        checksums[relative] = expected
    return checksums


def is_ignored(relative: str) -> bool:
    normalized = relative.rstrip("/") + ("/" if relative.endswith("/") else "")
    return (
        relative == ".git"
        or relative == ".cache/huggingface"
        or any(normalized.startswith(prefix) or relative.startswith(prefix) for prefix in IGNORED_DIRECTORY_PREFIXES)
    )


def iter_bundle_entries(root: Path) -> Iterable[Path]:
    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        retained_names = []
        for name in names:
            child = directory_path / name
            relative = child.relative_to(root).as_posix()
            if is_ignored(relative):
                continue
            retained_names.append(name)
            if child.is_symlink():
                yield child
        names[:] = retained_names
        for filename in filenames:
            yield directory_path / filename


def verify_exact_tree(root: Path, checksums: dict[str, str]) -> dict[str, str]:
    expected = set(checksums) | {MANIFEST_PATH}
    actual: set[str] = set()
    symlinks: list[str] = []
    non_regular: list[str] = []
    for path in iter_bundle_entries(root):
        relative = path.relative_to(root).as_posix()
        actual.add(relative)
        if path.is_symlink():
            symlinks.append(relative)
        elif not path.is_file():
            non_regular.append(relative)
    if symlinks:
        raise ValidationError(f"bundle contains symlinks: {sorted(symlinks)}")
    if non_regular:
        raise ValidationError(f"bundle contains non-regular entries: {sorted(non_regular)}")
    if actual != expected:
        raise ValidationError(
            "bundle file allowlist mismatch: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    for relative, expected_sha in sorted(checksums.items()):
        path = root / safe_relative(relative)
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValidationError(f"SHA-256 mismatch for {relative}: expected {expected_sha}, observed {actual_sha}")
    return checksums


def load_inventory(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read publication inventory: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") not in {1, 2}:
        raise ValidationError("publication inventory schema_version must be 1 or 2")
    return value


def inventory_assets(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(inventory.get("files"), list):
        raw_assets = inventory["files"]
        mapping = {
            "path": "relative_path",
            "rows": "rows",
            "bytes": "bytes",
            "sha256": "sha256",
            "dataset": "dataset",
            "category": "category",
        }
    elif isinstance(inventory.get("assets"), list):
        raw_assets = inventory["assets"]
        mapping = {
            "path": "path",
            "rows": "rows",
            "bytes": "bytes",
            "sha256": "sha256",
            "dataset": "dataset",
            "category": "category",
        }
    elif isinstance(inventory.get("public_assets"), list):
        raw_assets = inventory["public_assets"]
        mapping = {
            "path": "path_in_repo",
            "rows": "row_count",
            "bytes": "size_bytes",
            "sha256": "sha256",
            "dataset": "asset_id",
            "category": "role",
        }
    else:
        raise ValidationError("publication inventory does not contain files, assets, or public_assets")

    assets: list[dict[str, Any]] = []
    for raw in raw_assets:
        if not isinstance(raw, dict):
            raise ValidationError("publication inventory contains a non-object asset")
        try:
            asset = {name: raw[source] for name, source in mapping.items()}
        except KeyError as exc:
            raise ValidationError(f"publication inventory asset is missing {exc.args[0]}") from exc
        if not isinstance(asset["path"], str) or not asset["path"].endswith(".parquet"):
            raise ValidationError(f"inventory asset is not a parquet path: {asset['path']!r}")
        safe_relative(asset["path"])
        if not isinstance(asset["rows"], int) or asset["rows"] <= 0:
            raise ValidationError(f"inventory has invalid row count for {asset['path']}")
        if not isinstance(asset["bytes"], int) or asset["bytes"] <= 0:
            raise ValidationError(f"inventory has invalid byte count for {asset['path']}")
        if not isinstance(asset["sha256"], str) or len(asset["sha256"]) != 64:
            raise ValidationError(f"inventory has invalid SHA-256 for {asset['path']}")
        assets.append(asset)
    paths = [asset["path"] for asset in assets]
    if len(paths) != len(set(paths)):
        raise ValidationError("publication inventory contains duplicate parquet paths")
    return assets


def verify_inventory_contract(
    root: Path,
    inventory: dict[str, Any],
    checksums: dict[str, str],
) -> list[dict[str, Any]]:
    assets = inventory_assets(inventory)
    inventory_paths = {asset["path"] for asset in assets}
    manifest_parquets = {path for path in checksums if path.endswith(".parquet")}
    if inventory_paths != manifest_parquets:
        raise ValidationError(
            "inventory/manifest parquet allowlist mismatch: "
            f"inventory_only={sorted(inventory_paths - manifest_parquets)}, "
            f"manifest_only={sorted(manifest_parquets - inventory_paths)}"
        )
    for asset in assets:
        relative = asset["path"]
        path = root / safe_relative(relative)
        if asset["sha256"] != checksums[relative]:
            raise ValidationError(f"inventory/checksum disagreement for {relative}")
        if path.stat().st_size != asset["bytes"]:
            raise ValidationError(f"inventory byte-count mismatch for {relative}")
    return assets


def field_type(struct_type: Any, name: str) -> Any:
    try:
        return struct_type.field(name).type
    except (KeyError, IndexError) as exc:
        raise ValidationError(f"Parquet struct is missing field {name}") from exc


def validate_arrow_schema(path: Path, schema: Any) -> None:
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise ValidationError("pyarrow is required; install pyarrow before running validation") from exc

    names = set(schema.names)
    missing = set(CORE_COLUMNS) - names
    if missing:
        raise ValidationError(f"Parquet schema is missing core columns in {path}: {sorted(missing)}")
    for name in ("data_source", "ability"):
        if str(schema.field(name).type) not in STRING_TYPES:
            raise ValidationError(f"{path}: {name} must be string or large_string")

    reward_type = schema.field("reward_model").type
    if not pa.types.is_struct(reward_type):
        raise ValidationError(f"{path}: reward_model must be a struct")
    for name in ("ground_truth", "style"):
        if str(field_type(reward_type, name)) not in STRING_TYPES:
            raise ValidationError(f"{path}: reward_model.{name} must be a string")

    prompt_type = schema.field("prompt").type
    if not (pa.types.is_list(prompt_type) or pa.types.is_large_list(prompt_type)):
        raise ValidationError(f"{path}: prompt must be a list")
    prompt_item = prompt_type.value_type
    if not pa.types.is_struct(prompt_item):
        raise ValidationError(f"{path}: prompt items must be structs")
    for name in ("content", "role"):
        if str(field_type(prompt_item, name)) not in STRING_TYPES:
            raise ValidationError(f"{path}: prompt.{name} must be a string")

    if not pa.types.is_struct(schema.field("extra_info").type):
        raise ValidationError(f"{path}: extra_info must be a struct")
    if "split" in names and str(schema.field("split").type) not in STRING_TYPES:
        raise ValidationError(f"{path}: optional split column must be a string")


def validate_semantic_rows(path: Path, parquet_file: Any) -> int:
    checked = 0
    allowed_roles = {"system", "user", "assistant", "tool"}
    for batch in parquet_file.iter_batches(batch_size=1024, columns=list(CORE_COLUMNS)):
        for row_index, row in enumerate(batch.to_pylist(), start=checked):
            for name in ("data_source", "ability"):
                if not isinstance(row.get(name), str) or not row[name].strip():
                    raise ValidationError(f"{path}: row {row_index} has empty {name}")
            reward = row.get("reward_model")
            if not isinstance(reward, dict):
                raise ValidationError(f"{path}: row {row_index} has invalid reward_model")
            for name in ("ground_truth", "style"):
                if not isinstance(reward.get(name), str) or not reward[name].strip():
                    raise ValidationError(f"{path}: row {row_index} has empty reward_model.{name}")
            prompt = row.get("prompt")
            if not isinstance(prompt, list) or not prompt:
                raise ValidationError(f"{path}: row {row_index} has an empty prompt")
            has_user = False
            for message_index, message in enumerate(prompt):
                if not isinstance(message, dict):
                    raise ValidationError(f"{path}: row {row_index} prompt item {message_index} is not an object")
                role = message.get("role")
                content = message.get("content")
                if role not in allowed_roles:
                    raise ValidationError(
                        f"{path}: row {row_index} prompt item {message_index} has invalid role {role!r}"
                    )
                if not isinstance(content, str) or not content.strip():
                    raise ValidationError(f"{path}: row {row_index} prompt item {message_index} has empty content")
                has_user = has_user or role == "user"
            if not has_user:
                raise ValidationError(f"{path}: row {row_index} prompt has no user message")
            if not isinstance(row.get("extra_info"), dict):
                raise ValidationError(f"{path}: row {row_index} has invalid extra_info")
            checked += 1
    return checked


def verify_parquets(root: Path, assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValidationError("pyarrow is required; install pyarrow before running validation") from exc

    results: list[dict[str, Any]] = []
    for asset in sorted(assets, key=lambda item: item["path"]):
        path = root / safe_relative(asset["path"])
        try:
            parquet_file = pq.ParquetFile(path)
        except Exception as exc:
            raise ValidationError(f"cannot open Parquet payload {asset['path']}: {exc}") from exc
        rows = parquet_file.metadata.num_rows
        if rows != asset["rows"]:
            raise ValidationError(
                f"Parquet row-count mismatch for {asset['path']}: expected {asset['rows']}, observed {rows}"
            )
        validate_arrow_schema(path, parquet_file.schema_arrow)
        checked = validate_semantic_rows(path, parquet_file)
        if checked != rows:
            raise ValidationError(f"semantic row scan did not cover the full file {asset['path']}: {checked}/{rows}")
        results.append(
            {
                "path": asset["path"],
                "dataset": asset["dataset"],
                "category": asset["category"],
                "rows": rows,
                "schema_columns": parquet_file.schema_arrow.names,
                "semantic_rows_checked": checked,
            }
        )
    return results


def resolve_dataset_root(args: argparse.Namespace) -> Path:
    selected = args.dataset_root or args.repo_root
    root = selected if selected is not None else Path(__file__).resolve().parent
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    else:
        root = root.resolve()
    if not root.is_dir():
        raise ValidationError(f"dataset root is not a directory: {root}")
    return root


def validate(root: Path) -> dict[str, Any]:
    manifest = root / MANIFEST_PATH
    inventory_path = root / INVENTORY_PATH
    if not manifest.is_file() or manifest.is_symlink():
        raise ValidationError(f"missing regular checksum manifest: {manifest}")
    if not inventory_path.is_file() or inventory_path.is_symlink():
        raise ValidationError(f"missing regular publication inventory: {inventory_path}")
    checksums = parse_checksum_manifest(manifest)
    verify_exact_tree(root, checksums)
    inventory = load_inventory(inventory_path)
    assets = verify_inventory_contract(root, inventory, checksums)
    parquet_results = verify_parquets(root, assets)
    return {
        "ok": True,
        "dataset_root": str(root),
        "manifest_sha256": sha256_file(manifest),
        "inventory_sha256": sha256_file(inventory_path),
        "file_count": len(checksums) + 1,
        "payload_count": len(parquet_results),
        "payload_rows": sum(item["rows"] for item in parquet_results),
        "parquets": parquet_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the exact file tree, SHA-256 digests, inventory, Parquet row counts, "
            "Arrow schemas, and semantic prompt/reward smoke contract."
        )
    )
    roots = parser.add_mutually_exclusive_group()
    roots.add_argument("--dataset-root", type=Path, help="Downloaded RLdataset repository root")
    roots.add_argument("--repo-root", type=Path, help="Alias for --dataset-root")
    parser.add_argument("--receipt", type=Path, help="Optional new JSON receipt path")
    return parser.parse_args()


def write_receipt(path: Path, result: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValidationError(f"receipt path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    try:
        root = resolve_dataset_root(args)
        result = validate(root)
        if args.receipt is not None:
            receipt = args.receipt.resolve()
            try:
                receipt.relative_to(root)
            except ValueError:
                pass
            else:
                raise ValidationError("receipt must live outside the exact dataset tree")
            write_receipt(receipt, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValidationError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
