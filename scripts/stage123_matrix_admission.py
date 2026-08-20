#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

RUNTIME_FILES = (
    "scripts/stage123_matrix_manifest.py",
    "scripts/stage123_matrix_admission.py",
    "scripts/stage123_phase_adapter.py",
    "scripts/run_stage123_matrix_memory_probe.py",
    "scripts/run_stage123_matrix_throughput_probe.py",
    "recipe/on_policy_wdl_sft/code_task/qwen3_1p7b_stage123_resource_profile.sh",
    "recipe/on_policy_wdl_sft/code_task/run_stage123_matrix_throughput_probe_phase.sh",
    "recipe/on_policy_wdl_sft/code_task/run_s2_code_qwen3_1p7b_stage123_common.sh",
    "recipe/on_policy_wdl_sft/code_task/run_s2_code_model2_rollout_common.sh",
    "recipe/on_policy_wdl_sft/_common_wdl_sft_is_joint.sh",
)
MODEL1_FILE_FIELDS = {
    "config.json": "model1_config_sha256",
    "tokenizer_config.json": "model1_tokenizer_config_sha256",
    "chat_template.jinja": "model1_chat_template_sha256",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True).strip()


def render_manifest(repo_root: Path, manifest: Path) -> dict[str, Any]:
    output = subprocess.check_output(
        [sys.executable, str(repo_root / "scripts/stage123_matrix_manifest.py"), "render", str(manifest)], text=True
    )
    return json.loads(output)


def gpu_facts() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"], text=True
    )
    return [
        {"index": int(index.strip()), "name": name.strip(), "memory_total_mib": int(memory.strip())}
        for index, name, memory in (line.split(",") for line in output.splitlines())
    ]


def stage2_model1_identity(manifest: dict[str, Any]) -> dict[str, str]:
    bindings = []
    fields = (
        "model1_path",
        "model1_config_sha256",
        "model1_tokenizer_config_sha256",
        "model1_chat_template_sha256",
        "model1_provenance_path",
        "model1_provenance_sha256",
    )
    for run in manifest["runs"]:
        if run["phase"] == "stage2":
            bindings.append({field: run["source"][field] for field in fields})
    if not bindings or any(binding != bindings[0] for binding in bindings[1:]):
        raise SystemExit("all Stage2 arms must bind one identical Model1 identity")
    return bindings[0]


def verify_model1(identity: dict[str, str]) -> None:
    model_path = Path(identity["model1_path"]).resolve()
    required_suffix = "format_cold_start_fraction_cot_v3/qwen3-1p7b-kodcode-format-sft-frac25"
    if required_suffix not in str(model_path):
        raise SystemExit("Model1 is not the FRAC25 CoT-v3 format Cold Start model")
    for filename, field in MODEL1_FILE_FIELDS.items():
        path = model_path / filename
        if not path.is_file() or file_sha256(path) != identity[field]:
            raise SystemExit(f"Model1 identity mismatch: {path}")
    provenance_path = Path(identity["model1_provenance_path"])
    if not provenance_path.is_file() or file_sha256(provenance_path) != identity["model1_provenance_sha256"]:
        raise SystemExit("Model1 provenance hash mismatch")
    provenance = json.loads(provenance_path.read_text())
    if Path(provenance.get("target_dir", "")).resolve() != model_path:
        raise SystemExit("Model1 provenance target mismatch")


def runtime_file_hashes(repo_root: Path) -> dict[str, str]:
    result = {}
    for relative in RUNTIME_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise SystemExit(f"admission runtime file missing: {path}")
        result[relative] = file_sha256(path)
    return result


def validate_memory_probe(probe: dict[str, Any], manifest: dict[str, Any]) -> None:
    if probe.get("result_type") != "stage123_matrix_gpu_memory_probe" or probe.get("status") != "passed":
        raise SystemExit("memory probe is not passing")
    if probe.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise SystemExit("memory probe manifest mismatch")
    if probe.get("training_steps") != 0 or probe.get("optimizer_enabled") is not False:
        raise SystemExit("memory probe must remain zero-step with optimizer disabled")
    if float(probe.get("rollout_gpu_memory_utilization", 0.0)) < 0.4:
        raise SystemExit("memory probe used safety-only GPU memory utilization")
    if int(probe.get("rollout_max_num_batched_tokens", 0)) < 16384:
        raise SystemExit("memory probe used safety-only token batching")
    profile = manifest["resource_profile"]
    if (
        probe.get("rollout_free_cache_engine") != profile["rollout_free_cache_engine"]
        or probe.get("rollout_enable_sleep_mode") != profile["rollout_enable_sleep_mode"]
    ):
        raise SystemExit("memory probe rollout cache lifecycle mismatch")
    if probe.get("ref_fsdp_offload") is not True:
        raise SystemExit("memory probe did not exercise reference-model offload")
    if probe.get("actor_optimizer_offload") is not True:
        raise SystemExit("memory probe did not exercise optimizer offload")
    if probe.get("actor_param_offload") is not True:
        raise SystemExit("memory probe did not exercise actor parameter offload")
    if int(probe.get("ref_log_prob_micro_batch_size", 0)) != int(profile["ref_log_prob_micro_batch_size"]):
        raise SystemExit("memory probe reference log-prob micro-batch mismatch")
    if int(probe.get("ref_log_prob_max_token_len_per_gpu", 0)) != int(profile["ref_log_prob_max_token_len_per_gpu"]):
        raise SystemExit("memory probe reference dynamic token budget mismatch")


def validate_throughput_probe(probe: dict[str, Any], manifest: dict[str, Any], model1: dict[str, str]) -> None:
    if probe.get("result_type") != "stage123_matrix_throughput_probe" or probe.get("status") != "passed":
        raise SystemExit("throughput probe is not passing")
    if probe.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise SystemExit("throughput probe manifest mismatch")
    if int(probe.get("training_steps", 0)) < 1 or probe.get("optimizer_enabled") is not True:
        raise SystemExit("throughput probe must include a real optimizer step")
    utilization = float(probe.get("rollout_gpu_memory_utilization", 0.0))
    if utilization < 0.4 or utilization != float(manifest["resource_profile"]["rollout_gpu_memory_utilization"]):
        raise SystemExit("throughput probe GPU utilization mismatch")
    batched_tokens = int(probe.get("rollout_max_num_batched_tokens", 0))
    if batched_tokens != int(manifest["resource_profile"]["rollout_max_num_batched_tokens"]):
        raise SystemExit("throughput probe token batching mismatch")
    if (
        probe.get("rollout_free_cache_engine") != manifest["resource_profile"]["rollout_free_cache_engine"]
        or probe.get("rollout_enable_sleep_mode") != manifest["resource_profile"]["rollout_enable_sleep_mode"]
    ):
        raise SystemExit("throughput probe rollout cache lifecycle mismatch")
    if probe.get("ref_fsdp_offload") is not True:
        raise SystemExit("throughput probe did not exercise reference-model offload")
    if probe.get("actor_optimizer_offload") is not True:
        raise SystemExit("throughput probe did not exercise optimizer offload")
    if probe.get("actor_param_offload") is not True:
        raise SystemExit("throughput probe did not exercise actor parameter offload")
    if int(probe.get("minimum_gpu_headroom_mib", 0)) != int(manifest["resource_profile"]["minimum_gpu_headroom_mib"]):
        raise SystemExit("throughput probe GPU headroom threshold mismatch")
    if int(probe.get("ref_log_prob_micro_batch_size", 0)) != int(
        manifest["resource_profile"]["ref_log_prob_micro_batch_size"]
    ):
        raise SystemExit("throughput probe reference log-prob micro-batch mismatch")
    if int(probe.get("ref_log_prob_max_token_len_per_gpu", 0)) != int(
        manifest["resource_profile"]["ref_log_prob_max_token_len_per_gpu"]
    ):
        raise SystemExit("throughput probe reference dynamic token budget mismatch")
    if probe.get("model1_identity") != model1:
        raise SystemExit("throughput probe Model1 identity mismatch")
    stage2_ids = {run["id"] for run in manifest["runs"] if run["phase"] == "stage2"}
    run_results = probe.get("runs", [])
    if {run.get("run_id") for run in run_results} != stage2_ids:
        raise SystemExit("throughput probe must cover every Stage2 arm")
    for run in run_results:
        metrics = run.get("metrics", {})
        if (
            run.get("status") != "passed"
            or int(run.get("optimizer_steps", 0)) < 1
            or float(metrics.get("step_time_seconds", 0.0)) <= 0.0
            or float(metrics.get("rollout_tokens_per_second", 0.0)) <= 0.0
            or run.get("cleanup", {}).get("resources_released") is not True
            or run.get("formal_checkpoint_files")
        ):
            raise SystemExit(f"invalid throughput evidence for {run.get('run_id')}")


def create(args: argparse.Namespace) -> int:
    manifest = render_manifest(args.repo_root, args.manifest)
    memory_probe = json.loads(args.memory_probe.read_text())
    throughput_probe = json.loads(args.throughput_probe.read_text())
    model1 = stage2_model1_identity(manifest)
    verify_model1(model1)
    validate_memory_probe(memory_probe, manifest)
    validate_throughput_probe(throughput_probe, manifest, model1)
    facts = gpu_facts()
    if len(facts) != 8 or any(item["name"] != "NVIDIA L40S" for item in facts):
        raise SystemExit("matrix admission requires exactly 8 NVIDIA L40S GPUs")
    payload = {
        "schema_version": 2,
        "admission_type": "stage123_matrix_training_admission",
        "decision": "accepted",
        "candidate_commit": git_output(args.repo_root, "rev-parse", "HEAD"),
        "recipe_gitlink": git_output(args.repo_root, "rev-parse", "HEAD:recipe"),
        "runtime_file_hashes": runtime_file_hashes(args.repo_root),
        "manifest": str(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "resource_profile_sha256": manifest["resource_profile"]["sha256"],
        "rollout_gpu_memory_utilization": manifest["resource_profile"]["rollout_gpu_memory_utilization"],
        "rollout_max_num_batched_tokens": manifest["resource_profile"]["rollout_max_num_batched_tokens"],
        "ref_log_prob_micro_batch_size": manifest["resource_profile"]["ref_log_prob_micro_batch_size"],
        "ref_log_prob_max_token_len_per_gpu": manifest["resource_profile"]["ref_log_prob_max_token_len_per_gpu"],
        "actor_optimizer_offload": manifest["resource_profile"]["actor_optimizer_offload"],
        "actor_param_offload": manifest["resource_profile"]["actor_param_offload"],
        "minimum_gpu_headroom_mib": manifest["resource_profile"]["minimum_gpu_headroom_mib"],
        "model1_identity": model1,
        "memory_probe": str(args.memory_probe),
        "memory_probe_sha256": file_sha256(args.memory_probe),
        "throughput_probe": str(args.throughput_probe),
        "throughput_probe_sha256": file_sha256(args.throughput_probe),
        "run_ids": [run["id"] for run in manifest["runs"]],
        "gpu_facts": facts,
        "training_steps_calibrated": throughput_probe["training_steps"],
        "optimizer_enabled_during_calibration": throughput_probe["optimizer_enabled"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"ok": True, "admission": str(args.output), "sha256": file_sha256(args.output)}, sort_keys=True))
    return 0


def validate(args: argparse.Namespace) -> int:
    admission = json.loads(args.admission.read_text())
    if admission.get("schema_version") != 2 or admission.get("decision") != "accepted":
        raise SystemExit("invalid matrix admission")
    if args.run_id not in admission.get("run_ids", []):
        raise SystemExit("run id is not admitted")
    if git_output(args.repo_root, "rev-parse", "HEAD") != admission["candidate_commit"]:
        raise SystemExit("candidate commit mismatch")
    if git_output(args.repo_root, "rev-parse", "HEAD:recipe") != admission["recipe_gitlink"]:
        raise SystemExit("recipe gitlink mismatch")
    if runtime_file_hashes(args.repo_root) != admission["runtime_file_hashes"]:
        raise SystemExit("runtime file hash mismatch")
    manifest_path = Path(admission["manifest"])
    manifest = render_manifest(args.repo_root, manifest_path)
    if manifest["manifest_sha256"] != admission["manifest_sha256"]:
        raise SystemExit("manifest hash mismatch")
    model1 = stage2_model1_identity(manifest)
    if model1 != admission["model1_identity"]:
        raise SystemExit("admitted Model1 binding mismatch")
    verify_model1(model1)
    memory_path = Path(admission["memory_probe"])
    memory_probe = json.loads(memory_path.read_text())
    if file_sha256(memory_path) != admission["memory_probe_sha256"]:
        raise SystemExit("memory probe hash mismatch")
    validate_memory_probe(memory_probe, manifest)
    throughput_path = Path(admission["throughput_probe"])
    throughput_probe = json.loads(throughput_path.read_text())
    if file_sha256(throughput_path) != admission["throughput_probe_sha256"]:
        raise SystemExit("throughput probe hash mismatch")
    validate_throughput_probe(throughput_probe, manifest, model1)
    if gpu_facts() != admission["gpu_facts"]:
        raise SystemExit("GPU facts mismatch")
    print(json.dumps({"ok": True, "run_id": args.run_id}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--repo-root", type=Path, required=True)
    create_parser.add_argument("--manifest", type=Path, required=True)
    create_parser.add_argument("--memory-probe", type=Path, required=True)
    create_parser.add_argument("--throughput-probe", type=Path, required=True)
    create_parser.add_argument("--output", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--repo-root", type=Path, required=True)
    validate_parser.add_argument("--admission", type=Path, required=True)
    validate_parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    return create(args) if args.command == "create" else validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
