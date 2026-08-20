#!/usr/bin/env python3
"""Create and validate immutable Code Stage123 admissions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import yaml
from packaging.version import Version

RUNTIME_FILES = (
    "scripts/code_stage123_admission.py",
    "scripts/code_stage123_monitor.py",
    "scripts/code_stage123_probe_phase.py",
    "scripts/run_code_stage123_gpu_utilization_probe.py",
    "scripts/math_stage123_queue.py",
    "recipe/on_policy_wdl_sft/code_task/stage123_manifest_gate.sh",
    "recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh",
    "recipe/on_policy_wdl_sft/code_task/run_code_qwen3_1p7b_stage123_cotmask_v3_queue.sh",
    "recipe/on_policy_wdl_sft/code_task/run_s1_code_qwen3_1p7b_stage123_common.sh",
    "recipe/on_policy_wdl_sft/code_task/run_s2_code_qwen3_1p7b_stage123_common.sh",
    "recipe/on_policy_wdl_sft/code_task/run_s3_code_qwen3_1p7b_stage123_common.sh",
    "recipe/on_policy_wdl_sft/code_task/official_aligned_reward.py",
    "recipe/on_policy_wdl_sft/code_task/eval_code_vllm.py",
    "recipe/on_policy_wdl_sft/code_task/run_s1_code_base.sh",
    "recipe/on_policy_wdl_sft/code_task/run_s2_code_model2_rollout_common.sh",
    "recipe/on_policy_wdl_sft/ablation_single_model/_common_ablation.sh",
    "recipe/on_policy_wdl_sft/_common_wdl_sft_is_joint.sh",
    "verl/utils/tokenizer.py",
    "verl/trainer/ppo/ray_trainer.py",
    "tests/experiment_workflow/test_code_stage123_gpu_utilization_probe.py",
    "tests/on_policy_wdl_sft/test_code_stage123_cotmask_v3.py",
    "tests/on_policy_wdl_sft/test_code_task_reward_and_metrics.py",
    "tests/joint_training/regression/test_validation_generation_logging.py",
    "tests/utils/test_tokenizer_mistral_regex_guard_on_cpu.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_hashes(repo_root: Path) -> dict[str, str]:
    result = {}
    for relative in RUNTIME_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise SystemExit(f"runtime file missing: {path}")
        result[relative] = sha256(path)
    return result


def gpu_facts() -> list[dict[str, object]]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"], text=True
    )
    return [
        {"index": int(index), "name": name.strip(), "memory_total_mib": int(memory)}
        for index, name, memory in (line.split(",") for line in output.splitlines())
    ]


def scientific_manifest(manifest: dict) -> dict:
    normalized = json.loads(json.dumps(manifest))
    normalized.pop("status", None)
    normalized.pop("launch_allowed", None)
    paths = normalized.get("paths", {})
    paths.pop("admission", None)
    paths.pop("gpu_probe_report", None)
    paths.pop("event_log", None)
    paths.pop("state_root", None)
    return normalized


def validate_probe_training_arms(candidate: dict) -> None:
    zero_step = candidate["zero_step"][0]
    one_step = candidate["one_step"]
    if zero_step.get("run_id") != "b0-stage1":
        raise SystemExit("GPU probe zero-step validation did not use b0-stage1")
    if [item.get("run_id") for item in one_step] != ["b01-stage1", "b01-stage2-m2kl"]:
        raise SystemExit("GPU probe one-step arms did not cover beta=0.1 Stage1 and joint Stage2")
    for item in one_step:
        observed = item.get("observed_training_metrics", {})
        positive_loss = float(observed.get("actor/wdl_sft_loss_positive", 0.0))
        grad_norm = float(observed.get("actor/grad_norm", 0.0))
        if item.get("training_contract_complete") is not True:
            raise SystemExit("GPU probe scientific training contract is incomplete")
        if int(observed.get("wdl_sft/n_correct", 0)) <= 0:
            raise SystemExit("GPU probe one-step arm produced no positive samples")
        if not math.isfinite(positive_loss) or positive_loss <= 0:
            raise SystemExit("GPU probe one-step arm produced no positive loss")
        if not math.isfinite(grad_norm) or grad_norm <= 0:
            raise SystemExit("GPU probe one-step arm produced no gradient")


def validate_probe(report: dict, manifest: dict, manifest_path: Path) -> float:
    if report.get("result_type") != "code_stage123_gpu_utilization_probe" or report.get("status") != "passed":
        raise SystemExit("GPU probe is not passing")
    snapshot = Path(report.get("manifest_snapshot", ""))
    if not snapshot.is_file() or sha256(snapshot) != report.get("manifest_snapshot_sha256"):
        raise SystemExit("GPU probe manifest snapshot is missing or changed")
    if report.get("manifest_sha256") != sha256(snapshot):
        raise SystemExit("GPU probe does not bind its manifest snapshot")
    snapshot_manifest = yaml.safe_load(snapshot.read_text())
    if scientific_manifest(snapshot_manifest) != scientific_manifest(manifest):
        raise SystemExit("scientific manifest changed after GPU probe")
    utilization = float(report.get("selected_rollout_gpu_memory_utilization", 0.0))
    if utilization != float(manifest["resources"]["rollout_gpu_memory_utilization"]):
        raise SystemExit("GPU probe utilization does not match manifest")
    selected = [item for item in report.get("candidates", []) if item.get("status") == "passed"]
    if len(selected) != 1 or float(selected[0]["rollout_gpu_memory_utilization"]) != utilization:
        raise SystemExit("GPU probe selected candidate is ambiguous")
    candidate = selected[0]
    if len(candidate.get("zero_step", [])) != 1 or len(candidate.get("one_step", [])) != 2:
        raise SystemExit("GPU probe did not cover the admitted representative paths")
    validate_probe_training_arms(candidate)
    if not all(
        item.get("runtime_contract_complete") is True for item in [*candidate["zero_step"], *candidate["one_step"]]
    ):
        raise SystemExit("GPU probe runtime contract is incomplete")
    if not candidate["zero_step"][0].get("validation_complete"):
        raise SystemExit("GPU probe full Code-3 validation is incomplete")
    if any(int(item.get("optimizer_steps", 0)) < 1 for item in candidate["one_step"]):
        raise SystemExit("GPU probe representative optimizer steps are incomplete")
    minimum_required_headroom = int(report.get("minimum_required_gpu_headroom_mib", 0))
    if minimum_required_headroom != 512:
        raise SystemExit("GPU probe headroom contract is not the admitted 512 MiB floor")
    if int(candidate.get("minimum_observed_gpu_headroom_mib", 0)) < minimum_required_headroom:
        raise SystemExit("GPU probe has insufficient memory headroom")
    return utilization


def create(args: argparse.Namespace) -> int:
    manifest = yaml.safe_load(args.manifest.read_text())
    report = json.loads(args.probe_report.read_text())
    utilization = validate_probe(report, manifest, args.manifest)
    if manifest.get("task") != "code" or manifest.get("status") != "launch_ready_gpu_probe_passed":
        raise SystemExit("manifest is not launch-ready")
    if manifest.get("launch_allowed") is not True:
        raise SystemExit("manifest launch_allowed is not true")
    if Path(manifest["paths"].get("gpu_probe_report", "")) != args.probe_report:
        raise SystemExit("manifest does not bind the selected GPU probe report")
    if Path(manifest["paths"].get("admission", "")) != args.output:
        raise SystemExit("manifest does not bind the admission output")
    reviewer = json.loads(args.reviewer_receipt.read_text())
    if reviewer.get("decision") != "pass" or reviewer.get("p0") != 0 or reviewer.get("p1") != 0:
        raise SystemExit("reviewer receipt is not passing")
    selection = Path(manifest["paths"]["model1_selection"])
    dataset = Path(manifest["paths"]["dataset_receipt"])
    selected_model = Path(json.loads(selection.read_text())["identity"]["model_path"])
    model_config = json.loads((selected_model / "config.json").read_text())
    transformers_version = model_config.get("transformers_version")
    if (
        model_config.get("model_type") == "qwen3"
        and transformers_version
        and Version("4.57.2") < Version(transformers_version) < Version("5.0.0")
    ):
        raise SystemExit("Qwen3 Model1 config would trigger the Transformers Mistral-regex false positive")
    facts = gpu_facts()
    if len(facts) != 8 or any(item["name"] != "NVIDIA L40S" for item in facts):
        raise SystemExit("admission requires eight NVIDIA L40S GPUs")
    payload = {
        "schema_version": 1,
        "admission_type": "code_stage123_step20_training_admission",
        "decision": "accepted",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "model1_selection": str(selection),
        "model1_selection_sha256": sha256(selection),
        "dataset_receipt": str(dataset),
        "dataset_receipt_sha256": sha256(dataset),
        "gpu_probe_report": str(args.probe_report),
        "gpu_probe_report_sha256": sha256(args.probe_report),
        "rollout_gpu_memory_utilization": utilization,
        "run_ids": [run["id"] for run in manifest["runs"]],
        "runtime_file_hashes": runtime_hashes(args.repo_root),
        "gpu_facts": facts,
        "reviewer_receipt": str(args.reviewer_receipt),
        "reviewer_receipt_sha256": sha256(args.reviewer_receipt),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "admission": str(args.output), "sha256": sha256(args.output)}))
    return 0


def validate(args: argparse.Namespace) -> int:
    if not args.admission.is_file():
        raise SystemExit(f"admission file missing: {args.admission}")
    admission = json.loads(args.admission.read_text())
    if (
        admission.get("decision") != "accepted"
        or admission.get("admission_type") != "code_stage123_step20_training_admission"
    ):
        raise SystemExit("invalid Code Stage123 admission")
    if args.run_id not in admission.get("run_ids", []):
        raise SystemExit("run is absent from admission")
    reviewer_path = Path(admission["reviewer_receipt"])
    if sha256(reviewer_path) != admission["reviewer_receipt_sha256"]:
        raise SystemExit("reviewer receipt hash mismatch")
    reviewer = json.loads(reviewer_path.read_text())
    if reviewer.get("decision") != "pass" or reviewer.get("p0") != 0 or reviewer.get("p1") != 0:
        raise SystemExit("review result is not passing")
    manifest_path = Path(admission["manifest"])
    manifest = yaml.safe_load(manifest_path.read_text())
    if sha256(manifest_path) != admission["manifest_sha256"]:
        raise SystemExit("manifest hash mismatch")
    if sha256(Path(admission["model1_selection"])) != admission["model1_selection_sha256"]:
        raise SystemExit("Model1 receipt hash mismatch")
    if sha256(Path(admission["dataset_receipt"])) != admission["dataset_receipt_sha256"]:
        raise SystemExit("dataset receipt hash mismatch")
    probe_path = Path(admission["gpu_probe_report"])
    if sha256(probe_path) != admission["gpu_probe_report_sha256"]:
        raise SystemExit("GPU probe report hash mismatch")
    utilization = validate_probe(json.loads(probe_path.read_text()), manifest, manifest_path)
    if utilization != float(admission["rollout_gpu_memory_utilization"]):
        raise SystemExit("admitted utilization mismatch")
    if runtime_hashes(args.repo_root) != admission["runtime_file_hashes"]:
        raise SystemExit("runtime file hash mismatch")
    if gpu_facts() != admission["gpu_facts"]:
        raise SystemExit("GPU facts mismatch")
    print(json.dumps({"ok": True, "run_id": args.run_id, "utilization": utilization}))
    return 0


def compute_hash(args: argparse.Namespace) -> int:
    if not args.path.is_file():
        raise SystemExit(f"hash input missing: {args.path}")
    print(sha256(args.path))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--manifest", type=Path, required=True)
    create_parser.add_argument("--probe-report", type=Path, required=True)
    create_parser.add_argument("--output", type=Path, required=True)
    create_parser.add_argument("--reviewer-receipt", type=Path, required=True)
    create_parser.add_argument("--repo-root", type=Path, required=True)
    create_parser.set_defaults(func=create)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--admission", type=Path, required=True)
    validate_parser.add_argument("--run-id", required=True)
    validate_parser.add_argument("--repo-root", type=Path, required=True)
    validate_parser.set_defaults(func=validate)
    hash_parser = subparsers.add_parser("hash")
    hash_parser.add_argument("--path", type=Path, required=True)
    hash_parser.set_defaults(func=compute_hash)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
