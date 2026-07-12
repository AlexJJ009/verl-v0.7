#!/usr/bin/env python3
"""Validate, normalize, query, and inventory experiment manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import jsonschema
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/schema.json"


def load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    jsonschema.validate(data, json.loads(SCHEMA.read_text()))
    return data


def normalize(data: dict) -> dict:
    result = json.loads(json.dumps(data, sort_keys=True))
    result["runs"] = sorted(result["runs"], key=lambda item: item["order"])
    prefixes = [item["run_prefix"] for item in result["runs"]]
    ids = [item["id"] for item in result["runs"]]
    tmux = [item["tmux_name"] for item in result["runs"]]
    for label, values in (("run_prefix", prefixes), ("id", ids), ("tmux_name", tmux)):
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {label}")
    known = set(ids)
    for item in result["runs"]:
        if item["phase"] == "stage3" and item["source"].get("run_id") not in known:
            raise ValueError(f"missing source run for {item['id']}")
        if not item["artifact_dir"].startswith("/data-2/"):
            raise ValueError(f"artifact_dir must use /data-2: {item['id']}")
    workloads = result["calibration_workloads"]
    expected = {
        "stage1": ("base_pretrained", ["rollout"], 1),
        "stage2": ("fixed_model2_joint_rollout", ["model1", "model2"], 2),
        "stage3": ("stage2_model2_handoff", ["rollout"], 1),
    }
    for phase, (provenance, roles, count) in expected.items():
        workload = workloads[phase]
        if workload["phase"] != phase or workload["model_provenance_class"] != provenance:
            raise ValueError(f"{phase}: calibration workload identity mismatch")
        if [item["role"] for item in workload["model_sources"]] != roles:
            raise ValueError(f"{phase}: calibration model source roles mismatch")
        counts = workload["rollout_model_parameter_counts"]
        if len(counts) != count or sum(counts) != workload["rollout_model_parameter_count_sum"]:
            raise ValueError(f"{phase}: calibration parameter counts mismatch")
        if workload["log2_rollout_model_parameter_count_sum"] != round(math.log2(sum(counts)), 6):
            raise ValueError(f"{phase}: calibration log2 parameter count mismatch")
        names = [item["name"] for item in workload["datasets"]]
        if names != result["semantics"]["validation_datasets"]:
            raise ValueError(f"{phase}: calibration dataset order mismatch")
        for dataset in workload["datasets"]:
            if dataset["sha256"] != result["semantics"]["validation_dataset_hashes"][dataset["name"]]:
                raise ValueError(f"{phase}: calibration dataset hash mismatch: {dataset['name']}")
            if sum(dataset["difficulty_stratum_counts"].values()) != dataset["row_count"]:
                raise ValueError(f"{phase}: difficulty stratum count mismatch: {dataset['name']}")
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
