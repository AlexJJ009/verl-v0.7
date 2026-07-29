#!/usr/bin/env python3
"""Build the portable 13-payload RLVR dataset candidate from the verified v3 bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from validate_rebuttal_rlvr_dataset import ValidationError, sha256_file, validate

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_README = REPO_ROOT / "docs/joint_training/reports/data/rebuttal_rlvr_full_dataset_README.md"
DEFAULT_VALIDATOR = REPO_ROOT / "scripts/validate_rebuttal_rlvr_dataset.py"
DEFAULT_DECISION = (
    REPO_ROOT / "docs/joint_training/reports/data/rebuttal_rlvr_full_dataset_publication_decision_20260729.json"
)
DEFAULT_UPSTREAM_SOURCES = (
    REPO_ROOT / "docs/joint_training/reports/data/rebuttal_rlvr_full_dataset_upstream_sources_20260730.json"
)
APPROVED_DECISION_ID = "rebuttal-rlvr-full13-public-20260729"
SOURCE_INVENTORY_SHA256 = "b5b646a28b2e6bf8a6f531f986d921fbc20e5dc7c454453c3c7ce12a2674aa5a"
SOURCE_MANIFEST_SHA256 = "5e35eab998946be30857425525641b72bc7a1937376f7b797b45d47e71a73a59"
EXPECTED_SOURCE_SUMMARY = {
    "file_count": 17,
    "payload_count": 13,
    "payload_rows": 22860,
}


class CandidateError(RuntimeError):
    pass


def require_regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise CandidateError(f"{label} must be a regular file: {path}")


def verify_source(source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not source.is_absolute() or source.is_symlink() or not source.is_dir():
        raise CandidateError("--source-bundle must be an absolute, non-symlink directory")
    result = validate(source)
    for key, expected in EXPECTED_SOURCE_SUMMARY.items():
        if result.get(key) != expected:
            raise CandidateError(f"source bundle {key} drifted: expected {expected}, observed {result.get(key)}")
    if result["inventory_sha256"] != SOURCE_INVENTORY_SHA256:
        raise CandidateError("source publication inventory does not match the reviewed v3 pin")
    if result["manifest_sha256"] != SOURCE_MANIFEST_SHA256:
        raise CandidateError("source checksum manifest does not match the reviewed v3 pin")
    inventory_path = source / "metadata/publication_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    return result, inventory


def load_approved_decision(path: Path) -> dict[str, Any]:
    require_regular(path, "publication decision")
    try:
        decision = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CandidateError(f"invalid publication decision JSON: {path}") from exc
    expected_scope = {
        "payload_files": 13,
        "payload_rows": 22860,
        "math_training_files": 1,
        "math_evaluation_files": 7,
        "code_training_files": 1,
        "code_evaluation_files": 4,
    }
    if (
        decision.get("schema_version") != 1
        or decision.get("decision_id") != APPROVED_DECISION_ID
        or decision.get("decision") != "publish_full_13_payload_collection"
        or decision.get("scope") != expected_scope
        or decision.get("target_repository")
        != {
            "repo_id": "AlexGeek/RLdataset",
            "repo_type": "dataset",
            "visibility": "public",
            "gated": False,
        }
        or decision.get("format_policy", {}).get("publication_format_conversion") is not False
        or decision.get("history_policy", {}).get("policy") != "preserve_existing_history"
        or decision.get("attribution_and_license_policy", {}).get("collection_license") != "other"
    ):
        raise CandidateError("publication decision does not approve the exact full-13 target contract")
    return decision


def load_upstream_sources(path: Path, expected_datasets: set[str]) -> dict[str, Any]:
    require_regular(path, "upstream source catalog")
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CandidateError(f"invalid upstream source catalog JSON: {path}") from exc
    sources = catalog.get("sources")
    if (
        catalog.get("schema_version") != 1
        or catalog.get("collection_license") != "other"
        or not isinstance(sources, list)
        or len(sources) != 13
    ):
        raise CandidateError("upstream source catalog does not describe the full mixed-license set")
    by_dataset: dict[str, dict[str, Any]] = {}
    required_fields = {
        "dataset",
        "source_id",
        "source_revision",
        "source_url",
        "canonical_url",
        "upstream_license",
        "citation_url",
        "modifications",
    }
    for item in sources:
        if not isinstance(item, dict) or set(item) != required_fields:
            raise CandidateError("upstream source catalog contains an invalid entry")
        dataset = item["dataset"]
        if not isinstance(dataset, str) or dataset in by_dataset:
            raise CandidateError(f"invalid or duplicate upstream dataset: {dataset!r}")
        if any(not isinstance(item[field], str) or not item[field] for field in required_fields):
            raise CandidateError(f"upstream source entry has an empty field: {dataset}")
        by_dataset[dataset] = item
    if set(by_dataset) != expected_datasets:
        raise CandidateError(
            "upstream source catalog dataset mismatch: "
            f"missing={sorted(expected_datasets - set(by_dataset))}, "
            f"unexpected={sorted(set(by_dataset) - expected_datasets)}"
        )
    return {
        "reviewed_at": catalog["reviewed_at"],
        "collection_license": catalog["collection_license"],
        "policy": catalog["policy"],
        "by_dataset": by_dataset,
    }


def portable_inventory(
    source_inventory: dict[str, Any],
    decision: dict[str, Any],
    upstream_sources: dict[str, Any],
) -> dict[str, Any]:
    assets = []
    for item in source_inventory["files"]:
        assets.append(
            {
                "category": item["category"],
                "dataset": item["dataset"],
                "path": item["relative_path"],
                "rows": item["rows"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
                "upstream": upstream_sources["by_dataset"][item["dataset"]],
            }
        )

    excluded_assets = []
    for item in source_inventory.get("excluded_evaluator_assets", []):
        excluded_assets.append(
            {
                "asset": Path(item["path"]).name,
                "reason": item["reason"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
        )

    return {
        "schema_version": 2,
        "bundle_id": "rebuttal-rlvr-full-dataset-v4",
        "generated_at": decision["decided_at"],
        "target_repository": {
            "repo_id": "AlexGeek/RLdataset",
            "repo_type": "dataset",
            "required_visibility_during_review": "private",
            "approved_visibility_after_verification": "public",
            "approved_gated_state": False,
        },
        "layout_version": "rlvr-dataset-v2",
        "publication_status": "owner_approved_for_public_release",
        "publication_decision": {
            "decision_id": decision["decision_id"],
            "decided_at": decision["decided_at"],
            "decision_authority": decision["decision_authority"],
            "decision": decision["decision"],
            "owner_rationale": decision["owner_rationale"],
            "owner_statement_evidence": decision["owner_statement_evidence"],
            "attribution_and_license_policy": decision["attribution_and_license_policy"],
            "format_policy": decision["format_policy"],
            "history_policy": decision["history_policy"],
            "artifact_binding": decision["artifact_binding"],
            "acknowledgements": decision["acknowledgements"],
        },
        "packaging": {
            "publication_format_conversion": False,
            "payload_copy_mode": "byte_for_byte_from_reviewed_runtime_files",
            "payload_sha256_preserved": True,
            "file_format": "parquet",
        },
        "payload_summary": source_inventory["payload_summary"],
        "evaluation_contracts": source_inventory["evaluation_contracts"],
        "assets": assets,
        "validator": {
            "path": "validate_dataset.py",
            "checks": [
                "exact_file_allowlist",
                "no_dataset_symlinks",
                "sha256",
                "inventory_bytes_and_rows",
                "parquet_arrow_schema",
                "all_row_prompt_and_reward_semantics",
            ],
        },
        "evaluator_source_pins": source_inventory["evaluator_source_pins"],
        "excluded_evaluator_assets": excluded_assets,
        "attribution_and_license": {
            "policy": decision["attribution_and_license_policy"],
            "owner_decision_id": decision["decision_id"],
            "source_reviewed_at": upstream_sources["reviewed_at"],
            "collection_license": upstream_sources["collection_license"],
            "source_policy": upstream_sources["policy"],
        },
        "notes": [
            "Paths are relative to the downloaded repository root; no source-host path is part of the contract.",
            "No executable Code-7 contract exists; the executable code suite has four dataset files.",
            "Evaluator installations and large evaluator caches are not dataset payloads.",
            "The repository owner approved public release of the complete collection.",
            "Publication copies the actual runtime Parquet files without another format conversion.",
            "Existing Hugging Face repository history is preserved.",
        ],
    }


def copy_verified(source: Path, destination: Path, expected_sha: str | None = None) -> None:
    require_regular(source, "copy source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    require_regular(destination, "copied destination")
    if expected_sha is not None and sha256_file(destination) != expected_sha:
        raise CandidateError(f"copied file failed SHA-256 verification: {destination}")


def write_manifest(output: Path) -> None:
    manifest = output / "metadata/checksums.sha256"
    files = sorted(path for path in output.rglob("*") if path.is_file())
    if manifest in files:
        raise CandidateError("checksum manifest unexpectedly existed before generation")
    manifest.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n" for path in files),
        encoding="utf-8",
    )


def build_candidate(
    source: Path,
    output: Path,
    readme: Path,
    validator: Path,
    decision_path: Path,
    upstream_sources_path: Path,
) -> dict[str, Any]:
    if not output.is_absolute():
        raise CandidateError("--output must be an absolute path")
    if output.exists() or output.is_symlink():
        raise CandidateError(f"--output already exists: {output}")
    require_regular(readme, "README template")
    require_regular(validator, "validator")
    decision = load_approved_decision(decision_path)
    source_result, source_inventory = verify_source(source)
    expected_datasets = {item["dataset"] for item in source_inventory["files"]}
    upstream_sources = load_upstream_sources(upstream_sources_path, expected_datasets)

    output.mkdir(parents=True)
    copy_verified(source / ".gitattributes", output / ".gitattributes")
    for item in source_inventory["files"]:
        relative = Path(item["relative_path"])
        copy_verified(source / relative, output / relative, item["sha256"])
    copy_verified(readme, output / "README.md", sha256_file(readme))
    copy_verified(validator, output / "validate_dataset.py", sha256_file(validator))

    inventory_path = output / "metadata/publication_inventory.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        json.dumps(
            portable_inventory(source_inventory, decision, upstream_sources),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_manifest(output)
    result = validate(output)
    if result["file_count"] != 18 or result["payload_count"] != 13 or result["payload_rows"] != 22860:
        raise CandidateError(f"built candidate summary is unexpected: {result}")
    return {
        "ok": True,
        "mode": "build_full_public_release_candidate",
        "source": source_result,
        "candidate": result,
        "publication_status": "owner_approved_for_public_release",
        "publication_decision_id": decision["decision_id"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument("--validator", type=Path, default=DEFAULT_VALIDATOR)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    parser.add_argument(
        "--upstream-sources",
        type=Path,
        default=DEFAULT_UPSTREAM_SOURCES,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_candidate(
            args.source_bundle,
            args.output,
            args.readme,
            args.validator,
            args.decision,
            args.upstream_sources,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (CandidateError, ValidationError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
