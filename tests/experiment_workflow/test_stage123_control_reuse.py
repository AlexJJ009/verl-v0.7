from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import yaml

from scripts import stage123_control_reuse as control_reuse


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/stage123_control_reuse.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def fixture(tmp_path: Path) -> dict[str, Path]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"control")
    runtime = tmp_path / "runtime" / "frac25-stage1-control" / "logs"
    metrics = runtime / "metrics" / "OnPolicyWDLSFT-CodeTask" / "control.jsonl"
    validation = runtime / "validation" / "60.jsonl"
    metrics.parent.mkdir(parents=True)
    validation.parent.mkdir(parents=True)
    metrics.write_text('{"step":60}\n')
    validation.write_text('{"HumanEval+":0.5}\n')
    manifest = tmp_path / "stage123.yaml"
    profile = tmp_path / "profile.sh"
    preflight = tmp_path / "preflight.json"
    manifest.write_text(
        """runs:
- id: frac25-stage2
  run_prefix: frac25-stage2
  artifact_dir: /data-2/model_weights/stage123-old/frac25-stage2
  provenance_file: /data-2/model_weights/stage123-old/frac25-stage2.provenance.json
- id: frac25-stage3
  run_prefix: frac25-stage3
  artifact_dir: /data-2/model_weights/stage123-old/frac25-stage3
  provenance_file: /data-2/model_weights/stage123-old/frac25-stage3.provenance.json
calibration_workloads:
  stage3:
    model_sources:
    - role: model2
      state: pending
      path: /data-2/model_weights/stage123-old/frac25-stage2/stage2_final_model2
      producer:
        run_id: frac25-stage2
        final_step: 20
        output_path: /data-2/model_weights/stage123-old/frac25-stage2/stage2_final_model2
        provenance_path: /data-2/model_weights/stage123-old/frac25-stage3.provenance.json
"""
    )
    profile.write_text("PROFILE=fixture\n")
    preflight.write_text("{}\n")
    provenance = tmp_path / "control.provenance.json"
    write_json(
        provenance,
        {
            "checkpoint": str(checkpoint),
            "metrics": str(metrics),
            "final_step": 60,
            "manifest_sha256": "a" * 64,
            "train_file_sha256": "b" * 64,
        },
    )
    bundle = tmp_path / "admission.json"
    write_json(
        bundle,
        {
            "bindings": {
                "manifest_sha256": "a" * 64,
                "resource_profile_sha256": sha(profile),
                "preflight_result_sha256": sha(preflight),
            },
            "inputs": {"manifest": str(manifest), "resource_profile": str(profile), "preflight_result": str(preflight)},
        },
    )
    stage2 = tmp_path / "state" / "frac25-stage2.json"
    batch = tmp_path / "state" / "batch.json"
    write_json(
        stage2,
        {
            "run_id": "frac25-stage2",
            "child_id": "123",
            "failure": {"context": {"returncode": 1}},
            "transitions": [{"from": "pending", "to": "running"}, {"from": "running", "to": "failed"}],
        },
    )
    write_json(
        batch,
        {
            "status": "completed_with_failures",
            "phases": [{"run_id": "frac25-stage1-control", "status": "succeeded"}, {"run_id": "frac25-stage2", "status": "failed"}],
        },
    )
    queue_log = tmp_path / "queue.log"
    queue_log.write_text("RuntimeError: existing checkpoint root forbids automatic retry/resume for frac25-stage2\n")
    adapter = tmp_path / "adapter.py"
    adapter.write_text("if existing: raise RuntimeError('existing checkpoint root forbids automatic retry/resume')\nsubprocess.run(command)\n")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "legacy.bin").write_bytes(b"legacy")
    return locals()


def certify(paths: dict[str, Path], output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "certify",
            "--control-provenance", str(paths["provenance"]),
            "--stage2-state", str(paths["stage2"]),
            "--batch-state", str(paths["batch"]),
            "--queue-log", str(paths["queue_log"]),
            "--admission-bundle", str(paths["bundle"]),
            "--manifest", str(paths["manifest"]),
            "--adapter", str(paths["adapter"]),
            "--stage2-artifact-dir", str(paths["tmp_path"] / "stage2-artifacts"),
            "--legacy-checkpoint", str(paths["legacy"]),
            "--output", str(output),
        ],
        text=True,
        capture_output=True,
    )


def test_certified_control_reuse_preserves_old_evidence_and_prepares_distinct_treatment(tmp_path: Path):
    paths = fixture(tmp_path)
    old_batch = paths["batch"].read_bytes()
    certificate = tmp_path / "certificate.json"
    result = certify(paths, certificate)
    assert result.returncode == 0, result.stdout + result.stderr
    assert paths["batch"].read_bytes() == old_batch
    payload = json.loads(certificate.read_text())
    assert payload["eligible"] is True
    assert payload["failure_boundary"] == "child-started_pre-training-work"
    output_root = tmp_path / "treatment"
    prepared = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "prepare",
            "--certificate",
            str(certificate),
            "--manifest",
            str(paths["manifest"]),
            "--output-root",
            str(output_root),
            "--artifact-root",
            "/data-2/model_weights/stage123-test/reuse-001",
            "--execution-id",
            "reuse-001",
        ],
        text=True,
        capture_output=True,
    )
    assert prepared.returncode == 0, prepared.stderr
    admission = output_root / "treatment-admission.json"
    batch_manifest = output_root / "treatment-batch-manifest.json"
    assert batch_manifest.is_file()
    assert "prepared_not_authorized" in batch_manifest.read_text()
    treatment_manifest = yaml.safe_load((output_root / "treatment-manifest.yaml").read_text())
    stage2 = next(run for run in treatment_manifest["runs"] if run["id"] == "frac25-stage2")
    pending_source = treatment_manifest["calibration_workloads"]["stage3"]["model_sources"][0]
    assert pending_source["path"] == f"{stage2['artifact_dir']}/stage2_final_model2"
    assert pending_source["producer"]["output_path"] == pending_source["path"]
    assert pending_source["producer"]["provenance_path"] == "/data-2/model_weights/stage123-test/reuse-001/frac25-stage3.provenance.json"
    assert subprocess.run([sys.executable, str(TOOL), "validate-treatment", "--admission", str(admission), "--allow-prepared"], text=True, capture_output=True).returncode == 0
    assert subprocess.run([sys.executable, str(TOOL), "validate-treatment", "--admission", str(admission), "--allow-prepared", "--run-id", "frac25-stage1-control"], text=True, capture_output=True).returncode != 0
    payload = json.loads(admission.read_text())
    payload["treatment_manifest_sha256"] = "0" * 64
    payload["admission_sha256"] = hashlib.sha256(json.dumps({key: value for key, value in payload.items() if key != "admission_sha256"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    admission.write_text(json.dumps(payload, sort_keys=True))
    assert subprocess.run([sys.executable, str(TOOL), "validate-treatment", "--admission", str(admission), "--allow-prepared"], text=True, capture_output=True).returncode != 0
    assert subprocess.run([sys.executable, str(TOOL), "validate-treatment", "--admission", str(admission)], text=True, capture_output=True).returncode != 0


def test_prepare_rejects_non_data2_artifact_root(tmp_path: Path):
    paths = fixture(tmp_path)
    certificate = tmp_path / "certificate.json"
    assert certify(paths, certificate).returncode == 0
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "prepare",
            "--certificate",
            str(certificate),
            "--manifest",
            str(paths["manifest"]),
            "--output-root",
            str(tmp_path / "treatment"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--execution-id",
            "reuse-002",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "treatment artifact root must remain under /data-2" in result.stderr


def test_authorize_treatment_accepts_old_host_facts_and_rejects_failed_facts_and_profile_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paths = fixture(tmp_path)
    certificate = tmp_path / "certificate.json"
    assert certify(paths, certificate).returncode == 0
    output_root = tmp_path / "treatment"
    prepare = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "prepare",
            "--certificate",
            str(certificate),
            "--manifest",
            str(paths["manifest"]),
            "--output-root",
            str(output_root),
            "--artifact-root",
            "/data-2/model_weights/stage123-test/reuse-003",
            "--execution-id",
            "reuse-003",
        ],
        text=True,
        capture_output=True,
    )
    assert prepare.returncode == 0, prepare.stderr
    admission = output_root / "treatment-admission.json"
    host_facts = tmp_path / "host-facts.json"
    old_facts = {
        "artifact_type": "stage123_host_facts",
        "ok": True,
        "completed_at": (datetime.now(timezone.utc) - timedelta(minutes=16)).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "tmux": {"stage123_conflicts": []},
    }
    write_json(host_facts, {**old_facts, "ok": False})
    args = SimpleNamespace(admission=admission, batch_manifest=output_root / "treatment-batch-manifest.json", host_facts=host_facts, decision_id="D-test")
    with pytest.raises(ValueError, match="host facts are missing, invalid, or failed"):
        control_reuse.authorize_treatment(args)

    write_json(host_facts, old_facts)

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(command, 0, "\n".join(["NVIDIA L40S"] * 8), "")
        return subprocess.CompletedProcess(command, 0, "profile-does-not-match", "")

    monkeypatch.setattr(control_reuse.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="resource profile identity"):
        control_reuse.authorize_treatment(args)

    def fake_matching_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(command, 0, "\n".join(["NVIDIA L40S"] * 8), "")
        return subprocess.CompletedProcess(command, 0, sha(paths["profile"]), "")

    monkeypatch.setattr(control_reuse.subprocess, "run", fake_matching_run)
    assert control_reuse.authorize_treatment(args) == 0
    result = subprocess.run(
        [sys.executable, str(TOOL), "validate-treatment", "--admission", str(admission), "--run-id", "frac25-stage3"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_authorized_batch_manifest_rebinds_authorized_admission_without_rewriting_template(tmp_path: Path):
    paths = fixture(tmp_path)
    certificate = tmp_path / "certificate.json"
    assert certify(paths, certificate).returncode == 0
    output_root = tmp_path / "treatment"
    prepare = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "prepare",
            "--certificate",
            str(certificate),
            "--manifest",
            str(paths["manifest"]),
            "--output-root",
            str(output_root),
            "--artifact-root",
            "/data-2/model_weights/stage123-test/reuse-004",
            "--execution-id",
            "reuse-004",
        ],
        text=True,
        capture_output=True,
    )
    assert prepare.returncode == 0, prepare.stderr
    admission = output_root / "treatment-admission.json"
    host_facts = tmp_path / "host-facts.json"
    write_json(
        host_facts,
        {
            "artifact_type": "stage123_host_facts",
            "ok": True,
            "completed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "tmux": {"stage123_conflicts": []},
        },
    )
    admission_payload = json.loads(admission.read_text())
    admission_payload["status"] = "authorized"
    admission_payload["authorization"] = {"host_facts_path": str(host_facts), "host_facts_sha256": sha(host_facts)}
    admission_payload["admission_sha256"] = hashlib.sha256(json.dumps({key: value for key, value in admission_payload.items() if key != "admission_sha256"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    admission.write_text(json.dumps(admission_payload, sort_keys=True))
    prepared = output_root / "treatment-batch-manifest.json"
    prepared_bytes = prepared.read_bytes()
    authorized = control_reuse.authorize_batch_manifest(prepared, admission, "D-test")
    assert prepared.read_bytes() == prepared_bytes
    payload = json.loads(authorized.read_text())
    assert "prepared_not_authorized" not in payload
    assert payload["authorization_id"] == "D-test"
    assert payload["prepared_batch_manifest_sha256"] == json.loads(prepared.read_text())["batch_manifest_sha256"]
    assert payload["items"][0]["admission_bundle_sha256"] == sha(admission)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/experiment_execution_core.py"), "batch-validate", "--manifest", str(authorized), "--repo-root", str(ROOT)], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_certification_rejects_any_stage2_output_artifact(tmp_path: Path):
    paths = fixture(tmp_path)
    artifact = paths["tmp_path"] / "stage2-artifacts" / "stage2_final_model2"
    artifact.mkdir(parents=True)
    result = certify(paths, tmp_path / "certificate.json")
    assert result.returncode != 0
    assert "failed Stage2 produced forbidden artifact" in result.stdout
