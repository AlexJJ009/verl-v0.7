#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

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
        "status",
        "launch_allowed",
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
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    return manifest


def validate(manifest: dict) -> None:
    launch_allowed = manifest.get("launch_allowed") is True
    status = str(manifest.get("status", ""))
    if not launch_allowed and not any(marker in status for marker in ("invalidated", "blocked")):
        raise ValueError("non-launchable code Stage123 matrix must record a blocked or invalidated status")
    runs = sorted(manifest["runs"], key=lambda run: run["order"])
    manifest["runs"] = runs
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
            source = run.get("source", {})
            required_model1 = {
                "model1_path",
                "model1_config_sha256",
                "model1_tokenizer_config_sha256",
                "model1_chat_template_sha256",
                "model1_provenance_path",
                "model1_provenance_sha256",
            }
            missing_model1 = sorted(required_model1 - source.keys())
            if missing_model1:
                raise ValueError(f"{run['id']}: missing Model1 identity fields: {missing_model1}")
            legacy_fragment = "format_cold_start_fraction/qwen3-1p7b-kodcode-format-sft-frac25"
            cot_v3_fragment = "format_cold_start_fraction_cot_v3/qwen3-1p7b-kodcode-format-sft-frac25"
            uses_cot_v3 = cot_v3_fragment in source["model1_path"]
            if launch_allowed or "blocked" in status:
                allowed_fragment = cot_v3_fragment
            else:
                allowed_fragment = legacy_fragment
            if allowed_fragment not in source["model1_path"]:
                qualifier = "CoT-v3 " if launch_allowed or "blocked" in status else "legacy invalidated "
                raise ValueError(f"{run['id']}: Model1 must be the FRAC25 {qualifier}Cold Start model")
            if uses_cot_v3 and "invalidated_answer_only" in status:
                raise ValueError(f"{run['id']}: CoT-v3 Model1 cannot use answer-only invalidated status")
            for field in required_model1 - {"model1_path", "model1_provenance_path"}:
                if len(source[field]) != 64:
                    raise ValueError(f"{run['id']}: invalid {field}")
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
        "top_k": -1,
        "do_sample": True,
        "single_primary_metric": "val-core/code3_macro/acc/mean@3",
        "joint_primary_metric": "val-core/model2/code3_macro/acc/mean@3",
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
    profile = manifest.get("resource_profile", {})
    if float(profile.get("rollout_gpu_memory_utilization", 0.0)) < 0.4:
        raise ValueError("matrix rollout GPU memory utilization must be throughput-qualified at >= 0.4")
    if int(profile.get("rollout_max_num_batched_tokens", 0)) < 16384:
        raise ValueError("matrix rollout token batching remains safety-only")
    if not isinstance(profile.get("rollout_free_cache_engine"), bool) or not isinstance(
        profile.get("rollout_enable_sleep_mode"), bool
    ):
        raise ValueError("matrix rollout cache lifecycle must be explicit")
    if profile.get("ref_fsdp_offload") is not True:
        raise ValueError("matrix m2-KL profile must offload the reference model between forward passes")
    if profile.get("actor_optimizer_offload") is not True:
        raise ValueError("matrix profile must offload optimizer state")
    if profile.get("actor_param_offload") is not True:
        raise ValueError("matrix profile must offload actor parameters between phases")
    if int(profile.get("minimum_gpu_headroom_mib", 0)) < 1024:
        raise ValueError("matrix profile GPU headroom threshold must be at least 1024 MiB")
    if int(profile.get("ref_log_prob_micro_batch_size", 0)) != 1:
        raise ValueError("matrix KL reference log-prob micro-batch must equal calibrated value 1")
    if int(profile.get("ref_log_prob_max_token_len_per_gpu", 0)) != 9216:
        raise ValueError("matrix KL reference dynamic token budget must cover the full context")
    if profile.get("submodel_kl_reference_mode") != "standalone_enabled_submodel":
        raise ValueError("matrix must use a standalone enabled-submodel KL reference")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "render"))
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = load(args.manifest)
    if args.command == "validate":
        print(
            json.dumps(
                {"ok": True, "run_count": len(manifest["runs"]), "manifest_sha256": manifest["manifest_sha256"]},
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
