#!/usr/bin/env python3
"""Build the allowlisted public RLVR dataset bundle from the audited inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
from typing import Any
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = (
    REPO_ROOT
    / "docs/joint_training/reports/data/rebuttal_rlvr_dataset_publication_inventory_20260729.json"
)
DEFAULT_README = REPO_ROOT / "docs/joint_training/reports/data/rebuttal_rlvr_public_dataset_README.md"
CHUNK_SIZE = 1024 * 1024


class BundleError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: str) -> Path:
    if (
        not value
        or value.startswith("/")
        or value.endswith("/")
        or "//" in value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise BundleError(f"unsafe path_in_repo: {value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise BundleError(f"unsafe path_in_repo: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts:
        raise BundleError(f"unsafe path_in_repo: {value!r}")
    return Path(*pure.parts)


def read_inventory(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read inventory: {path}") from exc
    if value.get("schema_version") != 1:
        raise BundleError("inventory schema_version must be 1")
    if value.get("layout_version") != "rebuttal-rlvr-dataset-v1":
        raise BundleError("unexpected layout_version")
    if not value.get("public_assets") or not value.get("restricted_assets"):
        raise BundleError("inventory must contain public and restricted assets")
    return value


def parquet_rows(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise BundleError("pyarrow is required to verify parquet row counts") from exc
    return pq.ParquetFile(path).metadata.num_rows


def verify_regular_file(path: Path, expected_size: int, expected_sha: str, expected_rows: int | None) -> None:
    if path.is_symlink() or not path.is_file():
        raise BundleError(f"source must be a regular file, not a symlink: {path}")
    if path.stat().st_size != expected_size:
        raise BundleError(f"size mismatch for {path}")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise BundleError(f"SHA-256 mismatch for {path}: {actual_sha}")
    if expected_rows is not None and parquet_rows(path) != expected_rows:
        raise BundleError(f"row-count mismatch for {path}")


def verify_sources(inventory: dict[str, Any]) -> None:
    seen_destinations = {
        "README.md",
        ".gitattributes",
        "metadata/publication_inventory.json",
        "metadata/checksums.sha256",
    }

    def register_destination(value: str) -> None:
        safe_relative_path(value)
        if value in seen_destinations:
            raise BundleError(f"duplicate or reserved path_in_repo: {value}")
        seen_destinations.add(value)

    for asset in inventory["public_assets"]:
        destination = asset["path_in_repo"]
        register_destination(destination)
        verify_regular_file(
            Path(asset["local_path"]),
            int(asset["size_bytes"]),
            asset["sha256"],
            int(asset["row_count"]),
        )

    public_paths = {asset["local_path"] for asset in inventory["public_assets"]}
    restricted_paths = {asset["local_path"] for asset in inventory["restricted_assets"]}
    overlap = sorted(public_paths.intersection(restricted_paths))
    if overlap:
        raise BundleError(f"public/restricted allowlists overlap: {overlap}")

    for item in inventory.get("processing_files", []):
        register_destination(item["path_in_repo"])
        source = Path(item["local_path"])
        if source.is_symlink() or not source.is_file():
            raise BundleError(f"processing source must be a regular file: {source}")
        if sha256_file(source) != item["sha256"]:
            raise BundleError(f"processing source SHA-256 mismatch: {source}")

    for item in inventory["license_files"]:
        register_destination(item["path_in_repo"])


def public_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    public_assets = []
    for asset in inventory["public_assets"]:
        public_assets.append({key: value for key, value in asset.items() if key != "local_path"})

    restricted_assets = []
    for asset in inventory["restricted_assets"]:
        restricted_assets.append(
            {
                key: value
                for key, value in asset.items()
                if key
                not in {
                    "local_path",
                    "size_bytes",
                    "row_count",
                    "sha256",
                }
            }
        )

    return {
        "schema_version": 1,
        "inventory_id": inventory["inventory_id"],
        "layout_version": inventory["layout_version"],
        "generated_at": inventory["generated_at"],
        "public_assets": public_assets,
        "excluded_assets": restricted_assets,
        "evaluator_source_pins": inventory["evaluator_source_pins"],
        "license_files": [
            {key: value for key, value in item.items() if key != "url"}
            for item in inventory["license_files"]
        ],
        "processing_files": [
            {key: value for key, value in item.items() if key != "local_path"}
            for item in inventory.get("processing_files", [])
        ],
    }


def copy_verified(source: Path, destination: Path, expected_sha: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.is_symlink() or sha256_file(destination) != expected_sha:
        raise BundleError(f"copied file failed verification: {destination}")


def download_verified(url: str, destination: Path, expected_sha: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=30) as response:
            payload = response.read()
    except OSError as exc:
        raise BundleError(f"failed to download license: {url}") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha:
        raise BundleError(f"downloaded license SHA-256 mismatch: {url} -> {actual}")
    destination.write_bytes(payload)


def write_checksums(output: Path) -> None:
    files = sorted(path for path in output.rglob("*") if path.is_file())
    checksum_path = output / "metadata/checksums.sha256"
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for path in files:
        if path == checksum_path:
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n")
    checksum_path.write_text("".join(lines), encoding="utf-8")


def build_bundle(inventory: dict[str, Any], output: Path, readme: Path) -> None:
    if output.exists():
        raise BundleError(f"output already exists; choose a new directory: {output}")
    if not output.is_absolute():
        raise BundleError("output must be an absolute path")
    output.mkdir(parents=True)

    for asset in inventory["public_assets"]:
        copy_verified(
            Path(asset["local_path"]),
            output / safe_relative_path(asset["path_in_repo"]),
            asset["sha256"],
        )

    for item in inventory.get("processing_files", []):
        copy_verified(
            Path(item["local_path"]),
            output / safe_relative_path(item["path_in_repo"]),
            item["sha256"],
        )

    for license_file in inventory["license_files"]:
        download_verified(
            license_file["url"],
            output / safe_relative_path(license_file["path_in_repo"]),
            license_file["sha256"],
        )

    copy_verified(readme, output / "README.md", sha256_file(readme))
    (output / ".gitattributes").write_text(
        "*.parquet filter=lfs diff=lfs merge=lfs -text\n"
        "*.sqlite filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    metadata_path = output / "metadata/publication_inventory.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(public_inventory(inventory), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_checksums(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify allowlisted local sources without creating a bundle or downloading licenses.",
    )
    args = parser.parse_args()
    if not args.verify_only and args.output is None:
        parser.error("--output is required unless --verify-only is used")
    if args.verify_only and args.output is not None:
        parser.error("--output and --verify-only are mutually exclusive")
    return args


def main() -> int:
    args = parse_args()
    try:
        inventory = read_inventory(args.inventory)
        verify_sources(inventory)
        if not args.verify_only:
            build_bundle(inventory, args.output, args.readme)
    except BundleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    result = {
        "ok": True,
        "mode": "verify-only" if args.verify_only else "build",
        "public_assets": len(inventory["public_assets"]),
        "restricted_assets_excluded": len(inventory["restricted_assets"]),
    }
    if args.output is not None:
        result["output"] = str(args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
