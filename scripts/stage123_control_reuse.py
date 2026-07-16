#!/usr/bin/env python3
"""Prepare and validate the non-training Stage123 certified-control recovery."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
STAGE2_ID = "frac25-stage2"
STAGE3_ID = "frac25-stage3"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(path: Path, *, allow_empty: bool = False) -> str:
    if not path.is_dir():
        raise ValueError(f"checkpoint directory is missing: {path}")
    rows = []
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file():
            rows.append((str(candidate.relative_to(path)), digest(candidate)))
    if not rows and not allow_empty:
        raise ValueError(f"checkpoint directory is empty: {path}")
    return hashlib.sha256(canonical_json(rows).encode()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def workspace_path(value: str) -> Path:
    path = Path(value)
    if str(path).startswith("/workspace/verl/"):
        return ROOT / path.relative_to("/workspace/verl")
    return path


def expected_validation(metrics: Path, final_step: int) -> Path:
    return metrics.parents[2] / "validation" / f"{final_step}.jsonl"


def stage2_pretraining_proof(
    *,
    batch: dict[str, Any],
    state: dict[str, Any],
    queue_log: Path,
    adapter: Path,
    stage2_artifact_dir: Path,
    legacy_checkpoint: Path,
) -> list[str]:
    failures: list[str] = []
    transitions = [(event.get("from"), event.get("to")) for event in state.get("transitions", [])]
    if state.get("run_id") != STAGE2_ID or not state.get("child_id"):
        failures.append("stage2 state does not record a started child")
    if transitions != [("pending", "running"), ("running", "failed")]:
        failures.append("stage2 state does not preserve pending->running->failed")
    if state.get("failure", {}).get("context", {}).get("returncode") != 1:
        failures.append("stage2 state does not preserve expected child exit")
    if batch.get("status") != "completed_with_failures":
        failures.append("old batch is not terminal completed_with_failures")
    phases = {entry.get("run_id"): entry.get("status") for entry in batch.get("phases", [])}
    if phases.get("frac25-stage1-control") != "succeeded" or phases.get(STAGE2_ID) != "failed":
        failures.append("old batch phase terminal states do not prove succeeded control and failed stage2")
    marker = "existing checkpoint root forbids automatic retry/resume for frac25-stage2"
    if not queue_log.is_file() or marker not in queue_log.read_text(errors="replace"):
        failures.append("queue log lacks the preserved Stage2 pre-wrapper existing-checkpoint failure")
    adapter_source = adapter.read_text() if adapter.is_file() else ""
    guard = "existing checkpoint root forbids automatic retry/resume"
    if guard not in adapter_source or "subprocess.run(command" not in adapter_source:
        failures.append("stage2 adapter source does not contain expected guard and wrapper invocation")
    elif adapter_source.index(guard) > adapter_source.index("subprocess.run(command"):
        failures.append("stage2 adapter guard is not ordered before wrapper invocation")
    if not legacy_checkpoint.is_dir():
        failures.append("legacy diagnostic checkpoint is missing")
    for relative in ("stage2_final_joint", "stage2_final_model2", "frac25-stage2.provenance.json"):
        if (stage2_artifact_dir / relative).exists():
            failures.append(f"failed Stage2 produced forbidden artifact: {relative}")
    return failures


def training_plane(bundle: dict[str, Any], control: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    bindings = bundle.get("bindings", {})
    inputs = bundle.get("inputs", {})
    manifest = workspace_path(str(inputs.get("manifest", manifest_path)))
    profile = workspace_path(str(inputs.get("resource_profile", "")))
    evaluator = workspace_path(str(inputs.get("preflight_result", "")))
    required = {
        "manifest_sha256": bindings.get("manifest_sha256"),
        "resource_profile_sha256": bindings.get("resource_profile_sha256"),
        "train_file_sha256": control.get("train_file_sha256"),
        "evaluator_preflight_sha256": bindings.get("preflight_result_sha256"),
    }
    if not all(isinstance(value, str) and len(value) == 64 for value in required.values()):
        raise ValueError("admission bundle/control provenance lacks training-plane hashes")
    if not manifest.is_file():
        raise ValueError("scientific manifest is missing")
    try:
        rendered = json.loads(
            subprocess.check_output(
                [sys.executable, str(ROOT / "scripts/experiment_manifest.py"), "render", str(manifest), "--format", "json"], text=True
            )
        )
        manifest_identity = rendered.get("manifest_sha256")
    except subprocess.CalledProcessError:
        manifest_identity = control.get("manifest_sha256")
    if manifest_identity != required["manifest_sha256"] or control.get("manifest_sha256") != required["manifest_sha256"]:
        raise ValueError("control provenance manifest hash no longer matches admission bundle")
    if not profile.is_file():
        raise ValueError("resource profile is missing")
    try:
        from execution_results import rendered_resource_profile_sha256

        profile_hash = rendered_resource_profile_sha256(profile)
    except ValueError:
        profile_hash = digest(profile)
    if profile_hash != required["resource_profile_sha256"]:
        raise ValueError("resource profile hash no longer matches admission bundle")
    if not evaluator.is_file() or digest(evaluator) != required["evaluator_preflight_sha256"]:
        raise ValueError("evaluator/preflight hash no longer matches admission bundle")
    return {
        **required,
        "manifest_file_sha256": digest(manifest),
        "resource_profile_file_sha256": digest(profile),
        "training_plane_identity_sha256": hashlib.sha256(canonical_json(required).encode()).hexdigest(),
    }


def certify(args: argparse.Namespace) -> int:
    control = load_json(args.control_provenance)
    state = load_json(args.stage2_state)
    batch = load_json(args.batch_state)
    bundle = load_json(args.admission_bundle)
    checkpoint = Path(str(control.get("checkpoint", "")))
    metrics = Path(str(control.get("metrics", "")))
    validation = expected_validation(metrics, int(control.get("final_step", -1)))
    failures = stage2_pretraining_proof(
        batch=batch,
        state=state,
        queue_log=args.queue_log,
        adapter=args.adapter,
        stage2_artifact_dir=args.stage2_artifact_dir,
        legacy_checkpoint=args.legacy_checkpoint,
    )
    for label, path in {"control checkpoint": checkpoint, "control metrics": metrics, "control validation": validation}.items():
        if not path.exists():
            failures.append(f"missing {label}: {path}")
    try:
        plane = training_plane(bundle, control, args.manifest)
    except ValueError as exc:
        failures.append(str(exc))
        plane = {}
    certificate = {
        "schema_version": 1,
        "result_type": "stage123_certified_control_reuse",
        "eligible": not failures,
        "failure_boundary": "child-started_pre-training-work",
        "disclosure": "The completed matched Stage1 control is reused only as baseline evidence; treatment Stage2 -> extraction -> Stage3 receives a new execution identity and does not use the control final weights as Stage2 input.",
        "control": {
            "provenance_path": str(args.control_provenance),
            "provenance_sha256": digest(args.control_provenance),
            "checkpoint": str(checkpoint),
            "checkpoint_tree_sha256": tree_digest(checkpoint) if checkpoint.is_dir() else None,
            "metrics": str(metrics),
            "metrics_sha256": digest(metrics) if metrics.is_file() else None,
            "validation": str(validation),
            "validation_sha256": digest(validation) if validation.is_file() else None,
            "final_step": control.get("final_step"),
        },
        "training_plane": plane,
        "original_admission": {
            "path": str(args.admission_bundle),
            "sha256": digest(args.admission_bundle),
            "acceptance_report_path": bundle.get("acceptance_report_path"),
            "acceptance_report_sha256": bundle.get("acceptance_report_sha256"),
        },
        "old_failure": {
            "batch_state_path": str(args.batch_state),
            "batch_state_sha256": digest(args.batch_state),
            "stage2_state_path": str(args.stage2_state),
            "stage2_state_sha256": digest(args.stage2_state),
            "queue_log_path": str(args.queue_log),
            "queue_log_sha256": digest(args.queue_log) if args.queue_log.is_file() else None,
            "adapter_path": str(args.adapter),
            "adapter_sha256": digest(args.adapter) if args.adapter.is_file() else None,
            "legacy_checkpoint": str(args.legacy_checkpoint),
            "legacy_checkpoint_tree_sha256": tree_digest(args.legacy_checkpoint, allow_empty=True) if args.legacy_checkpoint.is_dir() else None,
        },
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(certificate, sort_keys=True))
    return 0 if certificate["eligible"] else 1


def validate_certificate(path: Path) -> dict[str, Any]:
    certificate = load_json(path)
    if certificate.get("result_type") != "stage123_certified_control_reuse" or certificate.get("eligible") is not True:
        raise ValueError("control reuse certificate is not eligible")
    for section, key in (("control", "provenance_path"), ("old_failure", "batch_state_path"), ("old_failure", "stage2_state_path")):
        value = certificate.get(section, {}).get(key)
        if not isinstance(value, str) or not Path(value).is_file():
            raise ValueError(f"certificate missing preserved evidence: {section}.{key}")
    return certificate


def prepare(args: argparse.Namespace) -> int:
    certificate = validate_certificate(args.certificate)
    execution_id = args.execution_id
    if not execution_id or "/" in execution_id:
        raise ValueError("execution_id must be a non-empty path-safe identifier")
    if args.output_root.exists():
        raise ValueError(f"treatment output root already exists: {args.output_root}")
    if not str(args.artifact_root).startswith("/data-2/"):
        raise ValueError("treatment artifact root must remain under /data-2")
    args.output_root.mkdir(parents=True)
    state_root = args.output_root / "state"
    monitor_path = args.output_root / "monitor" / "treatment-monitor.log"
    provenance_path = args.output_root / "provenance" / "treatment-reuse.provenance.json"
    state_root.mkdir()
    monitor_path.parent.mkdir()
    provenance_path.parent.mkdir()
    manifest_copy = args.output_root / "treatment-manifest.yaml"
    manifest_data = yaml.safe_load(args.manifest.read_text())
    if not isinstance(manifest_data, dict) or not isinstance(manifest_data.get("runs"), list):
        raise ValueError("treatment source manifest lacks runs")
    for run in manifest_data["runs"]:
        if run.get("id") in {STAGE2_ID, STAGE3_ID}:
            run["run_prefix"] = f"{run['run_prefix']}-TREATMENT-{execution_id}"
            run["artifact_dir"] = str(args.artifact_root / run["id"])
            run["provenance_file"] = str(args.artifact_root / f"{run['id']}.provenance.json")
    runs_by_id = {run.get("id"): run for run in manifest_data["runs"] if isinstance(run, dict)}
    if STAGE2_ID not in runs_by_id or STAGE3_ID not in runs_by_id:
        raise ValueError("treatment source manifest must contain Stage2 and Stage3 runs")
    stage2 = runs_by_id[STAGE2_ID]
    stage3 = runs_by_id[STAGE3_ID]
    for source in manifest_data.get("calibration_workloads", {}).get("stage3", {}).get("model_sources", []):
        producer = source.get("producer", {})
        if source.get("state") == "pending" and producer.get("run_id") == STAGE2_ID:
            output_path = str(Path(stage2["artifact_dir"]) / "stage2_final_model2")
            source["path"] = output_path
            producer["output_path"] = output_path
            producer["provenance_path"] = stage3["provenance_file"]
    manifest_copy.write_text(yaml.safe_dump(manifest_data, sort_keys=False))
    admission = {
        "schema_version": 1,
        "bundle_type": "stage123_treatment_reuse_admission",
        "status": "prepared_not_authorized",
        "execution_id": execution_id,
        "certificate_path": str(args.certificate),
        "certificate_sha256": digest(args.certificate),
        "original_manifest_path": str(args.manifest),
        "original_manifest_sha256": certificate["training_plane"]["manifest_sha256"],
        "original_manifest_file_sha256": digest(args.manifest),
        "treatment_manifest_path": str(manifest_copy),
        "treatment_manifest_sha256": digest(manifest_copy),
        "expected_run_ids": [STAGE2_ID, STAGE3_ID],
        "state_root": str(state_root),
        "monitor_path": str(monitor_path),
        "provenance_path": str(provenance_path),
        "control_reuse_disclosure": certificate["disclosure"],
    }
    admission["admission_sha256"] = hashlib.sha256(canonical_json(admission).encode()).hexdigest()
    admission_path = args.output_root / "treatment-admission.json"
    admission_path.write_text(json.dumps(admission, indent=2, sort_keys=True) + "\n")
    provenance_path.write_text(json.dumps({"schema_version": 1, "status": "prepared_not_authorized", "execution_id": execution_id, "certificate_sha256": admission["certificate_sha256"], "control_reuse": admission["control_reuse_disclosure"]}, indent=2, sort_keys=True) + "\n")
    commands = [
        ["/data-1/verl07/run_train.sh", "python", "/workspace/verl/scripts/stage123_phase_adapter.py", "--manifest", str(manifest_copy), "--run-id", run_id]
        for run_id in admission["expected_run_ids"]
    ]
    batch_manifest = {
        "schema_version": 1,
        "prepared_not_authorized": True,
        "authorization_id": f"pending-training-authorization:{execution_id}",
        "batch_id": f"stage123-treatment-reuse-{execution_id}",
        "created_at": "prepared-without-training",
        "failure_policy_id": "batch-fallback-v1",
        "operator_control_path": str(args.output_root / "operator-controls.jsonl"),
        "items": [{
            "item_id": f"stage123-treatment-reuse-{execution_id}",
            "goal_id": "stage123-primary-chain-execution",
            "plan_sha256": "0" * 64,
            "adapter_type": "stage123_treatment_reuse_v1",
            "admission_bundle_path": str(admission_path),
            "admission_bundle_sha256": digest(admission_path),
            "implementation_tree_sha256": "0" * 64,
            "expected_run_ids": admission["expected_run_ids"],
            "command_sha256": hashlib.sha256(canonical_json(commands).encode()).hexdigest(),
            "timeout_seconds": 86400,
        }],
    }
    batch_manifest["batch_manifest_sha256"] = hashlib.sha256(canonical_json(batch_manifest).encode()).hexdigest()
    batch_path = args.output_root / "treatment-batch-manifest.json"
    batch_path.write_text(json.dumps(batch_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "admission_path": str(admission_path), "batch_manifest_path": str(batch_path), "state_root": str(state_root), "monitor_path": str(monitor_path), "provenance_path": str(provenance_path)}, sort_keys=True))
    return 0


def validate_treatment(args: argparse.Namespace) -> int:
    admission = load_json(args.admission)
    expected_hash = admission.pop("admission_sha256", None)
    if expected_hash != hashlib.sha256(canonical_json(admission).encode()).hexdigest():
        raise ValueError("treatment admission hash mismatch")
    certificate_path = Path(str(admission.get("certificate_path", "")))
    if digest(certificate_path) != admission.get("certificate_sha256"):
        raise ValueError("treatment certificate hash mismatch")
    certificate = validate_certificate(certificate_path)
    if certificate["training_plane"].get("manifest_sha256") != admission.get("original_manifest_sha256"):
        raise ValueError("treatment manifest does not bind certified scientific manifest")
    manifest_path = Path(str(admission.get("treatment_manifest_path", "")))
    if not manifest_path.is_file() or digest(manifest_path) != admission.get("treatment_manifest_sha256"):
        raise ValueError("treatment manifest is missing")
    expected_run_ids = admission.get("expected_run_ids")
    if not isinstance(expected_run_ids, list) or expected_run_ids != [STAGE2_ID, STAGE3_ID]:
        raise ValueError("treatment admission run ids must be exactly Stage2 then Stage3")
    if args.run_id is not None and args.run_id not in expected_run_ids:
        raise ValueError("requested run id is not admitted for treatment reuse")
    if not args.allow_prepared and admission.get("status") != "authorized":
        raise ValueError("treatment admission is prepared but not authorized")
    if admission.get("status") == "authorized":
        authorization = admission.get("authorization", {})
        facts_path = Path(str(authorization.get("host_facts_path", "")))
        if not facts_path.is_file() or digest(facts_path) != authorization.get("host_facts_sha256"):
            raise ValueError("authorized treatment host facts binding mismatch")
        facts = load_json(facts_path)
        if facts.get("artifact_type") != "stage123_host_facts" or facts.get("ok") is not True:
            raise ValueError("authorized treatment host facts are invalid or failed")
    print(json.dumps({"ok": True, "execution_id": admission.get("execution_id"), "status": admission.get("status")}, sort_keys=True))
    return 0


def authorize_batch_manifest(prepared_path: Path, admission_path: Path, decision_id: str) -> Path:
    prepared = load_json(prepared_path)
    if prepared.get("prepared_not_authorized") is not True:
        raise ValueError("treatment batch manifest is not a prepared template")
    prepared_hash = prepared.get("batch_manifest_sha256")
    if prepared_hash != hashlib.sha256(canonical_json({key: value for key, value in prepared.items() if key != "batch_manifest_sha256"}).encode()).hexdigest():
        raise ValueError("prepared treatment batch manifest hash mismatch")
    items = prepared.get("items")
    if not isinstance(items, list) or len(items) != 1:
        raise ValueError("prepared treatment batch manifest must contain exactly one item")
    item = items[0]
    if item.get("admission_bundle_path") != str(admission_path):
        raise ValueError("prepared treatment batch manifest admission path mismatch")
    authorized = dict(prepared)
    authorized.pop("prepared_not_authorized", None)
    authorized["authorization_id"] = decision_id
    authorized["authorized_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    authorized["prepared_batch_manifest_sha256"] = prepared_hash
    authorized_item = dict(item)
    authorized_item["admission_bundle_sha256"] = digest(admission_path)
    authorized["items"] = [authorized_item]
    authorized["batch_manifest_sha256"] = hashlib.sha256(canonical_json({key: value for key, value in authorized.items() if key != "batch_manifest_sha256"}).encode()).hexdigest()
    output = prepared_path.with_name("authorized-treatment-batch-manifest.json")
    if output.exists():
        raise ValueError(f"authorized treatment batch manifest already exists: {output}")
    output.write_text(json.dumps(authorized, indent=2, sort_keys=True) + "\n")
    return output


def authorize_treatment(args: argparse.Namespace) -> int:
    admission = load_json(args.admission)
    if admission.get("status") != "prepared_not_authorized":
        raise ValueError("treatment admission is not in the preparable state")
    # Verify all immutable recovery inputs before adding fresh host evidence.
    validate_args = argparse.Namespace(admission=args.admission, allow_prepared=True, run_id=STAGE2_ID)
    validate_treatment(validate_args)
    host_facts = load_json(args.host_facts)
    if host_facts.get("artifact_type") != "stage123_host_facts" or host_facts.get("ok") is not True:
        raise ValueError("host facts are missing, invalid, or failed")
    if host_facts.get("tmux", {}).get("stage123_conflicts"):
        raise ValueError("host facts report an active Stage123 conflict")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, capture_output=True, check=False)
    rows = [row.strip() for row in gpu.stdout.splitlines() if row.strip()]
    if gpu.returncode != 0 or rows != ["NVIDIA L40S"] * 8:
        raise ValueError("GPU inventory is not exactly eight NVIDIA L40S devices")
    profile = ROOT / "recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh"
    profile_result = subprocess.run(["bash", "-lc", f"source {profile}; stage123_profile_hash"], text=True, capture_output=True, check=False)
    profile_hash = profile_result.stdout.strip()
    certificate = validate_certificate(Path(admission["certificate_path"]))
    if profile_result.returncode != 0 or profile_hash != certificate["training_plane"]["resource_profile_sha256"]:
        raise ValueError("resource profile identity does not match certified control")
    admission["status"] = "authorized"
    admission["authorization"] = {
        "decision_id": args.decision_id,
        "authorized_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "host_facts_path": str(args.host_facts),
        "host_facts_sha256": digest(args.host_facts),
        "gpu_inventory": rows,
        "resource_profile_sha256": profile_hash,
    }
    admission["admission_sha256"] = hashlib.sha256(canonical_json({key: value for key, value in admission.items() if key != "admission_sha256"}).encode()).hexdigest()
    args.admission.write_text(json.dumps(admission, indent=2, sort_keys=True) + "\n")
    batch_manifest = authorize_batch_manifest(args.batch_manifest, args.admission, args.decision_id)
    print(json.dumps({"ok": True, "execution_id": admission["execution_id"], "status": admission["status"], "admission_sha256": admission["admission_sha256"], "authorized_batch_manifest": str(batch_manifest)}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    certify_parser = sub.add_parser("certify")
    for name in ("control_provenance", "stage2_state", "batch_state", "queue_log", "admission_bundle", "manifest", "adapter", "stage2_artifact_dir", "legacy_checkpoint", "output"):
        certify_parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    certify_parser.set_defaults(func=certify)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--certificate", type=Path, required=True)
    prepare_parser.add_argument("--manifest", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--artifact-root", type=Path, required=True)
    prepare_parser.add_argument("--execution-id", required=True)
    prepare_parser.set_defaults(func=prepare)
    validate_parser = sub.add_parser("validate-treatment")
    validate_parser.add_argument("--admission", type=Path, required=True)
    validate_parser.add_argument("--allow-prepared", action="store_true")
    validate_parser.add_argument("--run-id")
    validate_parser.set_defaults(func=validate_treatment)
    authorize_parser = sub.add_parser("authorize-treatment")
    authorize_parser.add_argument("--admission", type=Path, required=True)
    authorize_parser.add_argument("--batch-manifest", type=Path, required=True)
    authorize_parser.add_argument("--host-facts", type=Path, required=True)
    authorize_parser.add_argument("--decision-id", required=True)
    authorize_parser.set_defaults(func=authorize_treatment)
    args = parser.parse_args()
    try:
        return args.func(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
