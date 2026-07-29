from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_validator():
    path = REPO_ROOT / "scripts/validate_rebuttal_rlvr_dataset.py"
    spec = importlib.util.spec_from_file_location("validate_rebuttal_rlvr_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_validator()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_rows() -> list[dict[str, object]]:
    return [
        {
            "data_source": "unit/math",
            "ability": "math",
            "reward_model": {"ground_truth": "42", "style": "rule"},
            "prompt": [
                {"content": "You are helpful.", "role": "system"},
                {"content": "What is six times seven?", "role": "user"},
            ],
            "split": "test",
            "extra_info": {"index": index},
        }
        for index in range(2)
    ]


def refresh_bundle_metadata(root: Path, *, expected_rows: int | None = None) -> None:
    parquet = root / "data/math/example.parquet"
    inventory = {
        "schema_version": 2,
        "assets": [
            {
                "path": "data/math/example.parquet",
                "rows": expected_rows if expected_rows is not None else pq.ParquetFile(parquet).metadata.num_rows,
                "bytes": parquet.stat().st_size,
                "sha256": digest(parquet),
                "dataset": "unit fixture",
                "category": "math_evaluation",
            }
        ],
    }
    inventory_path = root / "metadata/publication_inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != VALIDATOR.MANIFEST_PATH
    )
    manifest = root / VALIDATOR.MANIFEST_PATH
    manifest.write_text(
        "".join(f"{digest(path)}  {path.relative_to(root).as_posix()}\n" for path in paths),
        encoding="utf-8",
    )


def make_bundle(root: Path, rows: list[dict[str, object]] | None = None) -> Path:
    (root / "data/math").mkdir(parents=True)
    (root / "metadata").mkdir()
    (root / "README.md").write_text("fixture\n", encoding="utf-8")
    (root / ".gitattributes").write_text("*.parquet filter=lfs\n", encoding="utf-8")
    (root / "validate_dataset.py").write_text("# fixture validator\n", encoding="utf-8")
    pq.write_table(pa.Table.from_pylist(rows or valid_rows()), root / "data/math/example.parquet")
    refresh_bundle_metadata(root)
    return root


def test_validator_accepts_exact_bundle_from_unrelated_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_bundle(tmp_path / "bundle")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    result = VALIDATOR.validate(root)

    assert result["ok"] is True
    assert result["file_count"] == 6
    assert result["payload_count"] == 1
    assert result["payload_rows"] == 2
    assert result["parquets"][0]["semantic_rows_checked"] == 2


def test_validator_allows_hf_and_git_client_metadata(tmp_path: Path) -> None:
    root = make_bundle(tmp_path / "bundle")
    (root / ".git/objects").mkdir(parents=True)
    (root / ".git/objects/client-state").write_text("ignored\n", encoding="utf-8")
    (root / ".cache/huggingface/download").mkdir(parents=True)
    (root / ".cache/huggingface/download/client-state").write_text("ignored\n", encoding="utf-8")

    assert VALIDATOR.validate(root)["ok"] is True


def test_validator_rejects_unexpected_file(tmp_path: Path) -> None:
    root = make_bundle(tmp_path / "bundle")
    (root / "stale.bin").write_bytes(b"legacy")

    with pytest.raises(VALIDATOR.ValidationError, match="allowlist mismatch"):
        VALIDATOR.validate(root)


def test_validator_rejects_hash_mismatch(tmp_path: Path) -> None:
    root = make_bundle(tmp_path / "bundle")
    (root / "README.md").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(VALIDATOR.ValidationError, match="SHA-256 mismatch"):
        VALIDATOR.validate(root)


def test_validator_rejects_inventory_row_count_mismatch(tmp_path: Path) -> None:
    root = make_bundle(tmp_path / "bundle")
    refresh_bundle_metadata(root, expected_rows=3)

    with pytest.raises(VALIDATOR.ValidationError, match="row-count mismatch"):
        VALIDATOR.validate(root)


def test_validator_rejects_wrong_arrow_schema(tmp_path: Path) -> None:
    root = make_bundle(tmp_path / "bundle")
    rows = valid_rows()
    for row in rows:
        del row["ability"]
    pq.write_table(pa.Table.from_pylist(rows), root / "data/math/example.parquet")
    refresh_bundle_metadata(root)

    with pytest.raises(VALIDATOR.ValidationError, match="missing core columns"):
        VALIDATOR.validate(root)


def test_validator_rejects_empty_semantic_field(tmp_path: Path) -> None:
    root = make_bundle(tmp_path / "bundle")
    rows = valid_rows()
    rows[1]["ability"] = ""
    pq.write_table(pa.Table.from_pylist(rows), root / "data/math/example.parquet")
    refresh_bundle_metadata(root)

    with pytest.raises(VALIDATOR.ValidationError, match="empty ability"):
        VALIDATOR.validate(root)


@pytest.mark.parametrize("value", ["../escape", "/absolute", "a//b", "a/./b"])
def test_validator_rejects_unsafe_manifest_paths(value: str) -> None:
    with pytest.raises(VALIDATOR.ValidationError, match="unsafe relative path"):
        VALIDATOR.safe_relative(value)


def test_public_consumer_docs_do_not_export_this_hosts_network_or_storage_policy() -> None:
    guide = (REPO_ROOT / "docs/joint_training/guides/rebuttal_rlvr_hf_public_consumer_handoff.md").read_text(
        encoding="utf-8"
    )
    readme = (REPO_ROOT / "docs/joint_training/reports/data/rebuttal_rlvr_full_dataset_README.md").read_text(
        encoding="utf-8"
    )
    for document in (guide, readme):
        assert "hf download AlexGeek/RLdataset" in document
        assert "--dataset-root" in document
        assert "127.0.0.1:7890" not in document
        assert "大流量" not in document
        assert "Mihomo" not in document
        assert "/mnt/dolphinfs" not in document
        assert "Dockerfile" not in document
    assert "b1c264a92ace36dace52babdda651e415d9e9f82" in guide
    assert "REPLACE_WITH_VERIFIED_PUBLIC_COMMIT" not in guide
    assert "DATASET_REVISION" in readme
    assert "floating `main`" in readme
    assert "sibling layout is convenient but optional" in readme
    assert "This is a recommendation, not a constraint" in guide


def test_sanitized_public_receipt_pins_verified_release_without_host_details() -> None:
    receipt_path = (
        REPO_ROOT
        / "docs/joint_training/reports/data/rebuttal_rlvr_hf_public_receipt_20260730.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["repository"] == {
        "gated": False,
        "head": "b1c264a92ace36dace52babdda651e415d9e9f82",
        "history": [
            "b1c264a92ace36dace52babdda651e415d9e9f82",
            "da622cf077ca3f0eaf0ebc55dd4e115d0ebc0b9c",
        ],
        "id": "AlexGeek/RLdataset",
        "preserved_parent": "da622cf077ca3f0eaf0ebc55dd4e115d0ebc0b9c",
        "private": False,
        "type": "dataset",
        "url": "https://huggingface.co/datasets/AlexGeek/RLdataset",
    }
    assert receipt["bundle"]["file_count"] == 18
    assert receipt["bundle"]["payload_count"] == 13
    assert receipt["bundle"]["payload_rows"] == 22860
    assert receipt["verification"]["anonymous"]["ok"] is True
    assert receipt["verification"]["anonymous"]["credential_mode"] == "credential_free_fresh_home"
    serialized = json.dumps(receipt, sort_keys=True)
    assert "/data-1" not in serialized
    assert "/mnt/dolphinfs" not in serialized
    assert "Hong Kong" not in serialized
    assert "大流量" not in serialized


def test_full_dataset_readme_matches_reviewed_payload_paths_and_hashes() -> None:
    readme = (REPO_ROOT / "docs/joint_training/reports/data/rebuttal_rlvr_full_dataset_README.md").read_text(
        encoding="utf-8"
    )
    inventory = json.loads(
        (REPO_ROOT / "docs/joint_training/reports/data/rebuttal_rlvr_full_dataset_inventory_20260729.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(inventory["files"]) == 13
    for asset in inventory["files"]:
        assert f"`{asset['relative_path']}`" in readme
        assert f"`{asset['sha256']}`" in readme


def test_full_dataset_readme_cites_every_upstream_source_and_license() -> None:
    readme = (REPO_ROOT / "docs/joint_training/reports/data/rebuttal_rlvr_full_dataset_README.md").read_text(
        encoding="utf-8"
    )
    source_catalog = json.loads(
        (
            REPO_ROOT / "docs/joint_training/reports/data/rebuttal_rlvr_full_dataset_upstream_sources_20260730.json"
        ).read_text(encoding="utf-8")
    )
    assert source_catalog["collection_license"] == "other"
    assert len(source_catalog["sources"]) == 13
    for source in source_catalog["sources"]:
        assert source["source_url"] in readme
        assert source["upstream_license"]
        assert source["citation_url"] in readme


def test_full_candidate_is_portable_and_validator_bound_when_materialized() -> None:
    candidate = Path("/data-1/tmp/verl_agent_scratch/rlvr_full_public_release_20260730_v4r3")
    if not candidate.is_dir():
        pytest.skip("operational v4 full release is not materialized on this host")
    inventory = json.loads((candidate / "metadata/publication_inventory.json").read_text(encoding="utf-8"))
    assert inventory["publication_status"] == "owner_approved_for_public_release"
    assert inventory["publication_decision"]["decision"] == "publish_full_13_payload_collection"
    assert inventory["attribution_and_license"]["owner_decision_id"] == "rebuttal-rlvr-full13-public-20260729"
    assert inventory["packaging"] == {
        "file_format": "parquet",
        "payload_copy_mode": "byte_for_byte_from_reviewed_runtime_files",
        "payload_sha256_preserved": True,
        "publication_format_conversion": False,
    }
    assert inventory["validator"]["path"] == "validate_dataset.py"
    assert len(inventory["assets"]) == 13
    serialized = json.dumps(inventory, sort_keys=True)
    assert "/data-1" not in serialized
    assert "/data-2" not in serialized
    assert VALIDATOR.validate(candidate)["file_count"] == 18


def test_full13_publication_decision_matches_exact_payload_contract() -> None:
    decision = json.loads(
        (
            REPO_ROOT / "docs/joint_training/reports/data/rebuttal_rlvr_full_dataset_publication_decision_20260729.json"
        ).read_text(encoding="utf-8")
    )
    assert decision["decision"] == "publish_full_13_payload_collection"
    assert decision["target_repository"] == {
        "repo_id": "AlexGeek/RLdataset",
        "repo_type": "dataset",
        "visibility": "public",
        "gated": False,
    }
    assert decision["scope"] == {
        "payload_files": 13,
        "payload_rows": 22860,
        "math_training_files": 1,
        "math_evaluation_files": 7,
        "code_training_files": 1,
        "code_evaluation_files": 4,
    }
    assert decision["format_policy"]["publication_format_conversion"] is False
    assert decision["history_policy"]["policy"] == "preserve_existing_history"
    assert decision["attribution_and_license_policy"]["collection_license"] == "other"
