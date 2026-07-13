#!/usr/bin/env python3
"""Validate, normalize, query, and inventory experiment manifests."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import jsonschema
import yaml


def _load_workload_hashing():
    path = ROOT / "recipe/on_policy_wdl_sft/code_task/calibration_workload_descriptor.py"
    spec = importlib.util.spec_from_file_location("calibration_workload_descriptor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load artifact hashing: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.artifact_sha256, module.file_sha256


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/schema.json"


def load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    jsonschema.validate(data, json.loads(SCHEMA.read_text()))
    return data


class ManifestPolicyError(ValueError):
    def __init__(self, code: str, message: str, **context: object) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context

    def as_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "context": self.context}


def _policy_error(code: str, message: str, **context: object) -> None:
    raise ManifestPolicyError(code, message, **context)


def canonicalize(data: dict) -> dict:
    result = json.loads(json.dumps(data, sort_keys=True))
    result["runs"] = sorted(result["runs"], key=lambda item: item["order"])
    return result


def validate_policy_v1(result: dict) -> None:
    for label in ("run_prefix", "id", "tmux_name"):
        values = [item[label] for item in result["runs"]]
        if len(values) != len(set(values)):
            _policy_error("duplicate_identity", f"duplicate {label}", field=label)
    runs_by_id = {item["id"]: item for item in result["runs"]}
    runs_by_phase: dict[str, list[dict]] = {}
    for item in result["runs"]:
        runs_by_phase.setdefault(item["phase"], []).append(item)
        source_run_id = item.get("source", {}).get("run_id")
        if source_run_id is not None and source_run_id not in runs_by_id:
            _policy_error("missing_source_run", f"missing source run for {item['id']}", run_id=item["id"], source_run_id=source_run_id)
        if not item["artifact_dir"].startswith("/data-2/"):
            _policy_error("artifact_mount", f"artifact_dir must use /data-2: {item['id']}", run_id=item["id"], artifact_dir=item["artifact_dir"])

    workloads = result["calibration_workloads"]
    artifact_sha256, file_sha256 = _load_workload_hashing()
    shared_eligibility = None
    for phase, workload in workloads.items():
        if workload["phase"] != phase:
            _policy_error("workload_phase", f"{phase}: calibration workload identity mismatch", phase=phase, declared_phase=workload["phase"])
        sources = workload["model_sources"]
        for source in sources:
            if source["state"] == "pending":
                producer = source.get("producer")
                if not producer:
                    _policy_error("pending_producer_missing", f"{phase}: pending model source requires producer", phase=phase, role=source["role"])
                run = runs_by_id.get(producer["run_id"])
                if run is None or run["final_step"] != producer["final_step"]:
                    _policy_error("pending_producer_identity", f"{phase}: pending producer identity mismatch", phase=phase, producer=producer)
                consumers = [item for item in runs_by_phase.get(phase, []) if item.get("source", {}).get("run_id") == run["id"]]
                if not consumers or all(producer["provenance_path"] != item["provenance_file"] for item in consumers):
                    _policy_error("pending_provenance_path", f"{phase}: pending provenance path mismatch", phase=phase, producer=producer)
                if source["path"] != producer["output_path"]:
                    _policy_error("pending_output_path", f"{phase}: pending output path mismatch", phase=phase, source_path=source["path"], output_path=producer["output_path"])
            elif any(item.get("source", {}).get("run_id") for item in runs_by_phase.get(phase, [])):
                producer = source.get("producer")
                run = runs_by_id.get(producer["run_id"]) if producer else None
                provenance = source.get("provenance")
                if not producer or run is None or not provenance or producer["final_step"] != run["final_step"]:
                    _policy_error("materialized_producer_binding", f"{phase} materialized source requires current producer binding", phase=phase, producer=producer)
            if phase == "stage1" and source["path"] != result["paths"]["stage1_init_model"]:
                _policy_error("stage1_init_path", "stage1: init model path mismatch", source_path=source["path"], manifest_path=result["paths"]["stage1_init_model"])
            if phase == "stage1":
                provenance = source.get("provenance")
                if not provenance or provenance["path"] != result["paths"]["stage1_init_provenance"]:
                    _policy_error("stage1_provenance_path", "stage1: init provenance path mismatch", phase=phase)
            if source["state"] == "materialized":
                path = Path(source["path"])
                if artifact_sha256(path) != source["artifact_sha256"]:
                    _policy_error("artifact_hash", f"{phase}: model artifact hash mismatch: {source['role']}", phase=phase, role=source["role"])
                provenance = source.get("provenance")
                if provenance is not None and file_sha256(Path(provenance["path"])) != provenance["sha256"]:
                    _policy_error("provenance_hash", f"{phase}: model provenance hash mismatch: {source['role']}", phase=phase, role=source["role"])
        counts = workload["rollout_model_parameter_counts"]
        if len(counts) != len(sources) or sum(counts) != workload["rollout_model_parameter_count_sum"]:
            _policy_error("parameter_counts", f"{phase}: calibration parameter counts mismatch", phase=phase)
        if workload["log2_rollout_model_parameter_count_sum"] != round(math.log2(sum(counts)), 6):
            _policy_error("parameter_log2", f"{phase}: calibration log2 parameter count mismatch", phase=phase)
        names = [item["name"] for item in workload["datasets"]]
        if names != result["semantics"]["validation_datasets"]:
            _policy_error("dataset_order", f"{phase}: calibration dataset order mismatch", phase=phase, datasets=names)
        for dataset in workload["datasets"]:
            expected_hash = result["semantics"]["validation_dataset_hashes"].get(dataset["name"])
            if dataset["sha256"] != expected_hash:
                _policy_error("dataset_hash", f"{phase}: calibration dataset hash mismatch: {dataset['name']}", phase=phase, dataset=dataset["name"])
            if sum(dataset["difficulty_stratum_counts"].values()) != dataset["row_count"]:
                _policy_error("dataset_strata", f"{phase}: difficulty stratum count mismatch: {dataset['name']}", phase=phase, dataset=dataset["name"])
        eligibility = workload["validation_eligibility"]
        eligible_counts = eligibility["per_dataset_eligible_counts"]
        if set(eligible_counts) != set(names):
            _policy_error("eligibility_datasets", f"{phase}: eligibility dataset identity mismatch", phase=phase)
        if eligibility["submitted_prompt_count"] != sum(eligible_counts.values()):
            _policy_error("submitted_prompt_count", f"{phase}: submitted prompt count mismatch", phase=phase)
        source_counts = {item["name"]: item["row_count"] for item in workload["datasets"]}
        if any(eligible_counts[name] > source_counts[name] for name in names):
            _policy_error("eligible_count", f"{phase}: eligible count exceeds source row count", phase=phase)
        if shared_eligibility is None:
            shared_eligibility = eligibility
        elif eligibility != shared_eligibility:
            _policy_error("eligibility_phase_drift", f"{phase}: validation eligibility differs across phases", phase=phase)


def normalize(data: dict) -> dict:
    result = canonicalize(data)
    policy_version = result.get("schema_version")
    if policy_version != 1:
        _policy_error("unsupported_policy_version", f"unsupported manifest policy version: {policy_version}", policy_version=policy_version)
    validate_policy_v1(result)
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return result


def cmd_validate(args) -> int:
    report = normalize(load(args.manifest))
    print(json.dumps({"ok": True, "experiment_id": report["experiment_id"], "manifest_sha256": report["manifest_sha256"], "run_count": len(report["runs"])}, sort_keys=True))
    return 0


def cmd_render(args) -> int:
    report = normalize(load(args.manifest))
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.format == "tsv":
        print("id\tchain\tfraction\tphase\torder\trun_prefix\tfinal_step\ttmux_name\ttrain_file\ttrain_file_sha256\tartifact_dir\tprovenance_file")
        for item in report["runs"]:
            print("\t".join(str(item[key]) for key in ("id", "chain", "fraction", "phase", "order", "run_prefix", "final_step", "tmux_name", "train_file", "train_file_sha256", "artifact_dir", "provenance_file")))
    return 0


def cmd_run(args) -> int:
    report = normalize(load(args.manifest))
    item = next((run for run in report["runs"] if run["id"] == args.run_id), None)
    if item is None:
        raise SystemExit(f"unknown run id: {args.run_id}")
    print(json.dumps({**item, "manifest_sha256": report["manifest_sha256"], "resource_profile": report["resource_profile"], "paths": report["paths"], "preflight": report["preflight"], "semantics": report["semantics"]}, indent=2, sort_keys=True))
    return 0


def runnable_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.sh")
        if path.name.startswith("monitor") or "queue" in path.name
    )


def cmd_inventory(args) -> int:
    root = args.root.resolve()
    index = ROOT / "docs/joint_training/guides/training_script_index.md"
    index_text = index.read_text(encoding="utf-8", errors="replace") if index.is_file() else ""
    manifest = root / "experiment_manifest/stage123.yaml"
    manifest_paths = {
        "code_task/run_code_task_qwen3_1p7b_stage123_queue.sh",
        "code_task/monitor_code_task_qwen3_1p7b_stage123_notify.sh",
        "code_task/run_s1_code_qwen3_1p7b_stage123_common.sh",
        "code_task/run_s2_code_qwen3_1p7b_stage123_common.sh",
        "code_task/run_s3_code_qwen3_1p7b_stage123_common.sh",
    }
    entries = []
    for path in runnable_paths(root):
        rel = path.relative_to(root).as_posix()
        if rel in manifest_paths and manifest.is_file():
            classification = "manifest-native"
            evidence = [str(manifest.relative_to(ROOT)), str(path.relative_to(ROOT))]
        elif rel in index_text or path.name in index_text:
            classification = "legacy-traceable"
            evidence = [str(index.relative_to(ROOT)), str(path.relative_to(ROOT))]
        else:
            classification = "legacy-unresolved"
            evidence = [str(path.relative_to(ROOT))]
        entries.append({"path": rel, "classification": classification, "evidence_paths": evidence})
    report = {
        "schema_version": 1,
        "root": str(root),
        "entries": entries,
        "counts": {name: sum(item["classification"] == name for item in entries) for name in ("manifest-native", "legacy-traceable", "legacy-unresolved")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(args.output), "counts": report["counts"]}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("validate", cmd_validate), ("render", cmd_render)):
        p = sub.add_parser(name); p.add_argument("manifest", type=Path); p.set_defaults(func=fn)
        if name == "render": p.add_argument("--format", choices=("json", "tsv"), default="json")
    p = sub.add_parser("run"); p.add_argument("manifest", type=Path); p.add_argument("--run-id", required=True); p.set_defaults(func=cmd_run)
    p = sub.add_parser("inventory"); p.add_argument("--root", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.set_defaults(func=cmd_inventory)
    args = parser.parse_args()
    try:
        return args.func(args)
    except (jsonschema.ValidationError, ValueError, yaml.YAMLError) as exc:
        print(f"manifest invalid: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
