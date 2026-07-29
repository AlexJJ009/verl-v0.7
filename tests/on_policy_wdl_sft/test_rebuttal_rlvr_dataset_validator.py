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
    assert "REPLACE_WITH_VERIFIED_PUBLIC_COMMIT" in guide
    assert "DATASET_REVISION" in readme
    assert "floating `main`" in readme
    assert "sibling layout is convenient but optional" in readme
    assert "This is a recommendation, not a constraint" in guide


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


def test_full_candidate_is_portable_and_validator_bound_when_materialized() -> None:
    candidate = Path("/data-1/tmp/verl_agent_scratch/rlvr_full_upload_candidate_20260729_v4r1")
    if not candidate.is_dir():
        pytest.skip("operational v4 full candidate is not materialized on this host")
    inventory = json.loads((candidate / "metadata/publication_inventory.json").read_text(encoding="utf-8"))
    assert inventory["publication_status"] == "private_candidate_pending_owner_decision"
    assert inventory["validator"]["path"] == "validate_dataset.py"
    assert len(inventory["assets"]) == 13
    serialized = json.dumps(inventory, sort_keys=True)
    assert "/data-1" not in serialized
    assert "/data-2" not in serialized
    assert VALIDATOR.validate(candidate)["file_count"] == 18
