from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(module); return module


def fixture(tmp_path: Path):
    manifest_tool = load("release_manifest", ROOT / "scripts/experiment_manifest.py")
    verifier = load("release_verifier", ROOT / "scripts/verify_manifest_release_provenance.py")
    manifest = manifest_tool.normalize(manifest_tool.load(MANIFEST))
    run = manifest["runs"][0]
    train = tmp_path / "train.parquet"; train.write_bytes(Path(run["train_file"]).read_bytes())
    run["train_file"] = str(train); run["train_file_sha256"] = hashlib.sha256(train.read_bytes()).hexdigest()
    receipt = tmp_path / "receipt.json"; receipt.write_text('{"status":"pass"}')
    provenance = {
        "manifest_sha256": manifest["manifest_sha256"],
        "profile_sha256": manifest["resource_profile"]["sha256"],
        "train_file_sha256": run["train_file_sha256"],
        "preflight_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "run_prefix": run["run_prefix"],
        "release_eligible": True,
    }
    return verifier, manifest, run, train, receipt, provenance


def test_valid_manifest_provenance_passes(tmp_path: Path):
    verifier, manifest, run, train, receipt, provenance = fixture(tmp_path)
    assert verifier.verify(manifest, provenance, run["run_prefix"], run["final_step"], train, receipt)["ok"]


def test_mismatches_and_incomplete_release_fail(tmp_path: Path):
    verifier, manifest, run, train, receipt, provenance = fixture(tmp_path)
    provenance["manifest_sha256"] = "0" * 64
    provenance["release_eligible"] = False
    result = verifier.verify(manifest, provenance, run["run_prefix"], run["final_step"] + 1, train, receipt)
    assert not result["ok"]
    assert {"manifest_sha256 mismatch", "final_step mismatch", "run provenance is not release eligible"}.issubset(result["failures"])


def test_tampered_receipt_and_train_file_fail(tmp_path: Path):
    verifier, manifest, run, train, receipt, provenance = fixture(tmp_path)
    receipt.write_text("tampered"); train.write_bytes(b"tampered")
    result = verifier.verify(manifest, provenance, run["run_prefix"], run["final_step"], train, receipt)
    assert not result["ok"]
    assert "preflight_receipt_sha256 mismatch" in result["failures"]
    assert "train_file content mismatch" in result["failures"]


def test_release_wrapper_blocks_hook_until_manifest_provenance_passes(tmp_path: Path):
    verifier, manifest, run, train, receipt, provenance = fixture(tmp_path)
    normalized = tmp_path / "normalized.json"; normalized.write_text(json.dumps(manifest))
    provenance_file = tmp_path / "provenance.json"; provenance_file.write_text(json.dumps(provenance))
    fake_repo = tmp_path / "repo"; (fake_repo / "scripts").mkdir(parents=True)
    (fake_repo / "scripts/verify_manifest_release_provenance.py").write_bytes((ROOT / "scripts/verify_manifest_release_provenance.py").read_bytes())
    marker = tmp_path / "hook-called"
    hook = fake_repo / "scripts/code_task_training_release_hook.sh"
    hook.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\ntouch {marker}\n")
    hook.chmod(0o755)
    env = {
        **os.environ,
        "REPO": str(fake_repo),
        "RUN_PREFIX": run["run_prefix"],
        "FINAL_STEP": str(run["final_step"]),
        "TRAIN_FILE": str(train),
        "EXPERIMENT_NORMALIZED_MANIFEST": str(normalized),
        "EXPERIMENT_RUN_PROVENANCE": str(provenance_file),
        "EXPERIMENT_PREFLIGHT_RECEIPT": str(receipt),
    }
    wrapper = ROOT / "scripts/manifest_code_task_training_release_hook.sh"
    assert subprocess.run(["bash", str(wrapper)], env=env, capture_output=True).returncode == 0
    assert marker.is_file()
    marker.unlink(); provenance["release_eligible"] = False; provenance_file.write_text(json.dumps(provenance))
    failed = subprocess.run(["bash", str(wrapper)], env=env, capture_output=True, text=True)
    assert failed.returncode != 0
    assert not marker.exists()
    assert "not release eligible" in failed.stdout + failed.stderr
