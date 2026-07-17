#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load(path: Path) -> dict:
    overlay = yaml.safe_load(path.read_text())
    base_path = Path(overlay["base_manifest"])
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    rendered = subprocess.check_output(
        [sys.executable, str(ROOT / "scripts/experiment_manifest.py"), "render", str(base_path), "--format", "json"],
        text=True,
    )
    manifest = json.loads(rendered)
    for key in (
        "experiment_id",
        "resource_profile",
        "semantics",
        "paths",
        "monitor",
        "release",
        "validation",
        "hypotheses",
        "decision_policy",
        "runs",
    ):
        if key in overlay:
            if isinstance(overlay[key], dict) and isinstance(manifest.get(key), dict):
                manifest[key] = {**manifest[key], **overlay[key]}
            else:
                manifest[key] = overlay[key]
    validate(manifest)
    manifest["manifest_sha256"] = canonical_sha256({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    return manifest


def validate(manifest: dict) -> None:
    runs = sorted(manifest["runs"], key=lambda run: run["order"])
    manifest["runs"] = runs
    ids = [run["id"] for run in runs]
    for field in ("id", "run_prefix", "tmux_name", "order"):
        values = [run[field] for run in runs]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate matrix run {field}")
    seen: set[str] = set()
    for run in runs:
        if run["phase"] == "stage2":
            views = run.get("validation_views")
            if views != ["model1", "model2"]:
                raise ValueError(f"{run['id']}: Stage2 must validate model1 and model2")
        if run["phase"] == "stage3":
            source = run["source"]
            if source.get("run_id") not in seen or source.get("submodel") not in {"model1", "model2"}:
                raise ValueError(f"{run['id']}: invalid Stage3 source")
        seen.add(run["id"])
    validation = manifest["validation"]
    if validation != {
        "n": 3,
        "temperature": 0.2,
        "top_p": 0.95,
        "do_sample": True,
        "primary_metric": "val-core/model2/HumanEval+/acc/mean@3",
        "secondary_metrics": ["acc/pass@3", "acc/std@3"],
    }:
        raise ValueError("matrix validation protocol drift")
    if manifest.get("hypotheses") != {
        "primary": "model2_only_kl_improves_stage2_model2_and_persists_into_model2_stage3",
        "secondary": "observe_model1_joint_training_behavior_and_indirect_response_to_model2_kl",
    }:
        raise ValueError("matrix scientific hypotheses drift")
    decision_policy = manifest.get("decision_policy", {})
    if decision_policy.get("primary_endpoint") != "stage2_final_step20_model2_macro_mean_at_3":
        raise ValueError("matrix primary endpoint drift")
    if decision_policy.get("minimum_effect_pp") != 1.0:
        raise ValueError("matrix minimum effect drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "render"))
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = load(args.manifest)
    if args.command == "validate":
        print(json.dumps({"ok": True, "run_count": len(manifest["runs"]), "manifest_sha256": manifest["manifest_sha256"]}, sort_keys=True))
    else:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
