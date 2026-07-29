from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = load_script("prepare_rebuttal_rlvr_public_dataset", "scripts/prepare_rebuttal_rlvr_public_dataset.py")
PUBLISH = load_script("publish_rebuttal_rlvr_public_dataset", "scripts/publish_rebuttal_rlvr_public_dataset.py")


@pytest.mark.parametrize(
    "value",
    ["/absolute", "../escape", "a/../b", "a//b", "a/./b", "a/", ".", "bad\npath"],
)
def test_prepare_and_publish_reject_unsafe_repo_paths(value: str) -> None:
    with pytest.raises(PREPARE.BundleError, match="unsafe"):
        PREPARE.safe_relative_path(value)
    with pytest.raises(PUBLISH.PublishError, match="unsafe"):
        PUBLISH.safe_relative(value)


def test_public_inventory_removes_host_paths_and_remote_credentials() -> None:
    source = {
        "schema_version": 1,
        "inventory_id": "test",
        "layout_version": "rebuttal-rlvr-dataset-v1",
        "generated_at": "2026-07-29",
        "target_repository": {"current_credential_identity": "secret-user"},
        "public_assets": [
            {
                "asset_id": "allowed",
                "local_path": "/host/private/data.parquet",
                "path_in_repo": "data/allowed.parquet",
                "sha256": "a" * 64,
            }
        ],
        "restricted_assets": [
            {
                "asset_id": "blocked",
                "local_path": "/host/private/blocked.parquet",
                "expected_path_below_dataset_root": "data/blocked.parquet",
                "size_bytes": 10,
                "row_count": 1,
                "sha256": "b" * 64,
                "publication_status": "blocked",
                "reason": "test",
            }
        ],
        "evaluator_source_pins": [],
        "license_files": [{"license_id": "x", "path_in_repo": "LICENSES/x", "url": "https://x", "sha256": "c" * 64}],
        "processing_files": [
            {
                "local_path": "/host/private/process.py",
                "path_in_repo": "processing/process.py",
                "sha256": "d" * 64,
            }
        ],
    }
    result = PREPARE.public_inventory(source)
    rendered = json.dumps(result)
    assert "/host/private" not in rendered
    assert "secret-user" not in rendered
    assert "https://x" not in rendered
    assert result["excluded_assets"][0]["publication_status"] == "blocked"
    assert result["processing_files"] == [
        {"path_in_repo": "processing/process.py", "sha256": "d" * 64}
    ]


def write_checksum_file(bundle: Path) -> None:
    files = sorted(path for path in bundle.rglob("*") if path.is_file())
    checksum = bundle / "metadata/checksums.sha256"
    lines = []
    for path in files:
        if path == checksum:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(bundle).as_posix()}\n")
    checksum.write_text("".join(lines))


def make_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    (bundle / "metadata").mkdir(parents=True)
    (bundle / "data").mkdir()
    (bundle / "README.md").write_text("readme\n")
    (bundle / ".gitattributes").write_text("*.parquet filter=lfs\n")
    (bundle / "data/allowed.parquet").write_bytes(b"allowed")
    (bundle / "metadata/publication_inventory.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "layout_version": "rebuttal-rlvr-dataset-v1",
                "public_assets": [{"path_in_repo": "data/allowed.parquet"}],
                "license_files": [],
                "processing_files": [],
            }
        )
    )
    write_checksum_file(bundle)
    return bundle


def test_publish_bundle_verifier_accepts_exact_allowlist(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    checksums = PUBLISH.load_and_verify_bundle(bundle)
    assert set(checksums) == {
        ".gitattributes",
        "README.md",
        "data/allowed.parquet",
        "metadata/checksums.sha256",
        "metadata/publication_inventory.json",
    }


def test_publish_bundle_verifier_rejects_unlisted_data(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    (bundle / "data/restricted.parquet").write_bytes(b"restricted")
    write_checksum_file(bundle)
    with pytest.raises(PUBLISH.PublishError, match="file allowlist mismatch"):
        PUBLISH.load_and_verify_bundle(bundle)


def test_publish_bundle_verifier_rejects_unlisted_processing_file(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    (bundle / "processing").mkdir()
    (bundle / "processing/unreviewed.py").write_text("print('not reviewed')\n")
    write_checksum_file(bundle)
    with pytest.raises(PUBLISH.PublishError, match="file allowlist mismatch"):
        PUBLISH.load_and_verify_bundle(bundle)


def test_publish_bundle_verifier_rejects_symlink(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    (bundle / "linked").symlink_to(bundle / "README.md")
    write_checksum_file(bundle)
    with pytest.raises(PUBLISH.PublishError, match="symlinks"):
        PUBLISH.load_and_verify_bundle(bundle)
