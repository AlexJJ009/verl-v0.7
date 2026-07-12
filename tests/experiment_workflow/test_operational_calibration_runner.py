import json
import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_code_task_operational_calibration.sh"
QUEUE = ROOT / "scripts/run_code_task_operational_calibration_queue.sh"


def test_controlled_termination_requires_complete_validation_evidence() -> None:
    text = RUNNER.read_text()
    for key in (
        "timing_s/testing",
        "val-core/HumanEval+/acc/pass@1",
        "val-core/MBPP+/acc/pass@1",
        "val-core/LiveCodeBench/acc/pass@1",
    ):
        assert key in text
    assert 'if [ "\\$controlled_termination" = true ] && validation_complete && ! fatal_runtime_error; then rc=0; fi' in text
    assert 'controlled_termination_after_complete_validation' in text
    assert "docker kill '$CONTAINER'" in text
    assert "docker stop --time" not in text
    assert "fatal_runtime_error" in text
    assert "Fatal Python error" in text
    assert "if fatal_runtime_error; then rc=1; fi" in text
    assert 'CALIBRATION_ROLE must be bootstrap or acceptance' in text
    assert 'REP_INDEX required' in text
    assert 'CALIBRATION_ALLOW_DEPLOYABLE' in text
    assert 'CODE_TASK_SKIP_DB_IMPORT' in text
    assert 'db_wandb_side_effects_allowed":%s' in text
    assert "Stage1 init provenance target mismatch" in text
    assert "Stage1 model artifact hash mismatch" in text
    assert "Stage1 provenance hash mismatch" in text


def test_stage1_phase_requires_bound_format_sft_identity() -> None:
    text = (ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_operational_calibration_phase.sh").read_text()
    stage1 = text[text.index(" stage1)"):text.index(" stage2)")]
    assert "STAGE1_INIT_PROVENANCE_PATH" in stage1
    assert "Stage1 calibration provenance target mismatch" in stage1
    assert 'INIT_MODEL_PATH="${STAGE1_INIT_MODEL_PATH:?}"' in stage1


def test_acceptance_requires_frozen_history_and_matching_prediction_contract() -> None:
    text = RUNNER.read_text()
    assert 'if [ "$CALIBRATION_ROLE" = acceptance ]; then' in text
    assert 'CALIBRATION_HISTORY_INDEX required for acceptance' in text
    assert 'CALIBRATION_PREDICTION_CONTRACT required for acceptance' in text
    assert 'check_calibration_prediction_contract.py' in text
    assert 'prediction contract history_index_sha256 mismatch' in text
    assert 'prediction contract manifest_sha256 mismatch' in text
    assert 'CALIBRATION_PREDICTION_CONTRACT_SHA256 mismatch' in text


def test_queue_propagates_stage1_identity_from_manifest() -> None:
    text = QUEUE.read_text()
    assert "manifest_get paths.stage1_init_model" in text
    assert "manifest_get paths.stage1_init_provenance" in text
    assert 'STAGE1_INIT_MODEL_PATH="$STAGE1_INIT_MODEL_PATH"' in text
    assert 'STAGE1_INIT_PROVENANCE_PATH="$STAGE1_INIT_PROVENANCE_PATH"' in text
    assert 'STAGE1_INIT_MODEL_PATH=$(manifest_get paths.stage1_init_model)' in text
    assert '${STAGE1_INIT_MODEL_PATH:-' not in text
    assert 'QWEN3_1P7B_MODEL_PATH=$(manifest_get paths.base_model)' in text
    assert 'QWEN3_1P7B_MODEL_PATH="$QWEN3_1P7B_MODEL_PATH"' in text
    for variable in ("CALIBRATION_HUMANEVAL_PLUS_FILE", "CALIBRATION_MBPP_PLUS_FILE", "CALIBRATION_LIVE_CODE_BENCH_FILE"):
        assert f'{variable}="${variable}"' in text


def test_resource_sampling_starts_at_validation_rollout_readiness() -> None:
    text = RUNNER.read_text()
    assert "CALIBRATION_VALIDATION_READY_FILE" in text
    assert "if [ -s '$READY' ]; then" in text
    assert "CALIBRATION_VALIDATION_READY_FILE='$READY'" in text
    assert "CALIBRATION_VALIDATION_TIMELINE_FILE='$TIMELINE'" in text
    assert "measurement_window':'validation_rollout_readiness_to_completion'" in text
    assert '"measurement_started": measurement_started' in (ROOT / "scripts/sample_nvml_gpu_wait.py").read_text()
    assert "sample_nvml_gpu_wait.py" in text
    assert "--interval 0.2" in text
    assert "wait \"\\$gpu_sampler_pid\"" in text


def test_runner_terminates_immediately_on_fatal_runtime_error() -> None:
    text = RUNNER.read_text()
    loop = text.index('while kill -0 "\\$command_pid"')
    fatal_check = text.index("if fatal_runtime_error; then", loop)
    validation_check = text.index("if validation_complete; then", loop)
    assert fatal_check < validation_check
    assert "fatal_termination=true" in text[fatal_check:validation_check]
    assert "docker kill '$CONTAINER'" in text[fatal_check:validation_check]


def _fake_runner(path: Path, log: Path, fail_on: str | None = None) -> None:
    script = f"""#!/usr/bin/env python3
import json, os, sys, yaml
from pathlib import Path
phase = sys.argv[1]
role = os.environ["CALIBRATION_ROLE"]
rep = os.environ["REP_INDEX"]
Path({str(log)!r}).parent.mkdir(parents=True, exist_ok=True)
with open({str(log)!r}, "a") as handle:
    required = [
        os.environ["QWEN3_1P7B_MODEL_PATH"],
        os.environ["CALIBRATION_HUMANEVAL_PLUS_FILE"],
        os.environ["CALIBRATION_MBPP_PLUS_FILE"],
        os.environ["CALIBRATION_LIVE_CODE_BENCH_FILE"],
    ]
    handle.write(f"{{role}} {{phase}} {{rep}} {{'|'.join(required)}}\\n")
if {fail_on!r} == f"{{role}}:{{phase}}:{{rep}}":
    raise SystemExit(42)
root = Path(os.environ["CALIBRATION_ROOT"]) / role / phase / f"rep_{{rep}}"
root.mkdir(parents=True, exist_ok=True)
(root / f"{{phase}}.status.json").write_text(json.dumps({{
    "phase": phase,
    "role": role,
    "rep_index": int(rep),
    "returncode": 0,
    "elapsed_seconds": 100,
    "timed_out": False,
    "fatal_termination": False,
}}) + "\\n")
(root / f"{{phase}}.resources.json").write_text(json.dumps({{
    "phase": phase,
    "peak_rss_gib": 50,
    "gpu_wait_fraction": 0.2,
}}) + "\\n")
(root / phase / "logs/metrics/OnPolicyWDLSFT-CodeTask").mkdir(parents=True, exist_ok=True)
(root / phase / "logs/metrics/OnPolicyWDLSFT-CodeTask/metrics.jsonl").write_text(json.dumps({{
    "step": 0,
    "data": {{"timing_s/testing": 109}},
}}) + "\\n")
(root / phase / "logs/validation/run").mkdir(parents=True, exist_ok=True)
manifest = yaml.safe_load(Path(os.environ["CALIBRATION_MANIFEST"]).read_text())
workload = manifest["calibration_workloads"][phase]
eligible_counts = workload["validation_eligibility"]["per_dataset_eligible_counts"]
rows = []
for source in workload["datasets"]:
    name = source["name"]
    rows.extend({{
        "uid": f"{{name}}-{{i}}",
        "data_source": name,
        "response_token_count": 100,
        "response_eos_present": True,
        "response_finish_reason": "stop",
        "code_reward_latency_seconds": 1.0,
        "code_reward_timeout": 0,
    }} for i in range(eligible_counts[name]))
(root / phase / "logs/validation/run/0.jsonl").write_text("\\n".join(
    json.dumps(row) for row in rows
) + "\\n")
(root / f"{{phase}}.validation_timeline.jsonl").write_text("\\n".join([
    json.dumps({{"event":"validation_ready","monotonic_seconds":10}}),
    json.dumps({{"event":"generation_complete","monotonic_seconds":90}}),
    json.dumps({{"event":"metrics_complete","monotonic_seconds":110}}),
]) + "\\n")
"""
    path.write_text(script)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _queue_env(tmp_path: Path, fake_runner: Path) -> dict[str, str]:
    source_manifest = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml"
    manifest = yaml.safe_load(source_manifest.read_text())
    for phase, workload in manifest["calibration_workloads"].items():
        counts = workload["validation_eligibility"]["per_dataset_eligible_counts"]
        uid_doc = {
            "schema_version": 1,
            "datasets": [
                {"name": source["name"], "source_index": index, "ordered_uids": [f"{source['name']}-{i}" for i in range(counts[source["name"]])]}
                for index, source in enumerate(workload["datasets"])
            ],
        }
        canonical = (json.dumps(uid_doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
        workload["validation_eligibility"]["ordered_eligible_uid_sha256"] = hashlib.sha256(canonical).hexdigest()
    test_manifest = tmp_path / "stage123.test.yaml"
    test_manifest.write_text(yaml.safe_dump(manifest, sort_keys=False))
    manifest_tool = ROOT / "scripts/experiment_manifest.py"
    normalized = tmp_path / "stage123.normalized.json"
    rendered = subprocess.run(["python3", str(manifest_tool), "render", str(test_manifest), "--format", "json"], text=True, capture_output=True, check=True)
    normalized.write_text(rendered.stdout)
    report = tmp_path / "machine.json"; report.write_text('{"ok":true}\n')
    policy = tmp_path / "policy.json"; policy.write_text('{"policy":true}\n')
    budget = tmp_path / "budget.json"; budget.write_text('{"ok":true,"decision":"pass"}\n')
    receipt = tmp_path / "preflight-receipt.json"
    subprocess.run([
        "python3", str(ROOT / "scripts/stage123_preflight_receipt.py"), "issue",
        "--normalized-manifest", str(normalized), "--report", str(report), "--policy", str(policy),
        "--budget-result", str(budget), "--output", str(receipt),
    ], check=True, capture_output=True, text=True)
    env = os.environ.copy()
    env.update(
        {
            "TMUX": "/tmp/fake-tmux",
            "CALIBRATION_RUNNER": str(fake_runner),
            "CALIBRATION_REPORT_ROOT": str(tmp_path / "report"),
            "CALIBRATION_HISTORY_ROOT": str(tmp_path / "history"),
            "CALIBRATION_PREDICTION_ROOT": str(tmp_path / "prediction"),
            "CALIBRATION_QUEUE_SCRATCH": str(tmp_path / "scratch"),
            "CALIBRATION_QUEUE_POLL_SECONDS": "0",
            "CALIBRATION_MANIFEST": str(test_manifest),
            "CALIBRATION_NORMALIZED_MANIFEST": str(normalized),
            "CALIBRATION_PREFLIGHT_REPORT": str(report),
            "CALIBRATION_PREFLIGHT_RECEIPT": str(receipt),
            "CALIBRATION_PREFLIGHT_POLICY": str(policy),
        }
    )
    return env


def test_direct_runner_requires_preflight_before_side_effects(tmp_path: Path) -> None:
    root = tmp_path / "calibration"
    env = {
        **os.environ,
        "CALIBRATION_ROLE": "bootstrap",
        "REP_INDEX": "0",
        "CALIBRATION_ROOT": str(root),
        "ALLOW_CODE_OPERATIONAL_CALIBRATION": "1",
    }
    result = subprocess.run(["bash", str(RUNNER), "stage1"], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "CALIBRATION_NORMALIZED_MANIFEST required" in result.stderr
    assert not root.exists()


def _stage1_identity_env(tmp_path: Path) -> tuple[dict[str, str], Path, Path, Path]:
    descriptor_path = ROOT / "recipe/on_policy_wdl_sft/code_task/calibration_workload_descriptor.py"
    spec = __import__("importlib.util").util.spec_from_file_location("identity_descriptor", descriptor_path)
    descriptor = __import__("importlib.util").util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(descriptor)
    model = tmp_path / "format-sft"; model.mkdir(); (model / "config.json").write_text('{"model_type":"qwen3"}\n'); (model / "weights.bin").write_bytes(b"weights")
    provenance = model / "format_cold_start_source.json"; provenance.write_text(json.dumps({"target_dir": str(model)}) + "\n")
    source_manifest = json.loads(subprocess.run(["python3", str(ROOT / "scripts/experiment_manifest.py"), "render", str(ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml"), "--format", "json"], text=True, capture_output=True, check=True).stdout)
    source_manifest["paths"]["stage1_init_model"] = str(model); source_manifest["paths"]["stage1_init_provenance"] = str(provenance)
    source = source_manifest["calibration_workloads"]["stage1"]["model_sources"][0]
    source["path"] = str(model); source["artifact_sha256"] = descriptor.artifact_sha256(model)
    source["provenance"]["path"] = str(provenance); source["provenance"]["sha256"] = descriptor.file_sha256(provenance)
    normalized = tmp_path / "normalized.json"; normalized.write_text(json.dumps(source_manifest, sort_keys=True) + "\n")
    report = tmp_path / "report.json"; report.write_text('{}\n'); policy = tmp_path / "policy.json"; policy.write_text('{}\n'); budget = tmp_path / "budget.json"; budget.write_text('{"ok":true,"decision":"pass"}\n')
    receipt = tmp_path / "receipt.json"
    subprocess.run(["python3", str(ROOT / "scripts/stage123_preflight_receipt.py"), "issue", "--normalized-manifest", str(normalized), "--report", str(report), "--policy", str(policy), "--budget-result", str(budget), "--output", str(receipt)], check=True, capture_output=True, text=True)
    marker = tmp_path / "tmux-called"; fake_bin = tmp_path / "bin"; fake_bin.mkdir(); tmux = fake_bin / "tmux"; tmux.write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n"); tmux.chmod(0o700)
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "ALLOW_CODE_OPERATIONAL_CALIBRATION": "1", "CALIBRATION_ROLE": "bootstrap", "REP_INDEX": "0", "CALIBRATION_ROOT": str(tmp_path / "calibration"), "CALIBRATION_NORMALIZED_MANIFEST": str(normalized), "CALIBRATION_PREFLIGHT_REPORT": str(report), "CALIBRATION_PREFLIGHT_RECEIPT": str(receipt), "CALIBRATION_PREFLIGHT_POLICY": str(policy), "CALIBRATION_EXPECTED_PROFILE_HASH": source_manifest["resource_profile"]["sha256"], "CALIBRATION_PREFLIGHT_RECEIPT_MAX_AGE_SECONDS": "3600", "STAGE1_INIT_MODEL_PATH": str(tmp_path / "attacker-model"), "STAGE1_INIT_PROVENANCE_PATH": str(tmp_path / "attacker.json")}
    return env, model, provenance, marker


def test_stage1_identity_tamper_fails_before_tmux(tmp_path: Path) -> None:
    env, model, _provenance, marker = _stage1_identity_env(tmp_path)
    (model / "weights.bin").write_bytes(b"tampered")
    result = subprocess.run(["bash", str(RUNNER), "stage1"], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode != 0 and "Stage1 model artifact hash mismatch" in result.stderr
    assert not marker.exists()


def test_stage1_provenance_tamper_fails_before_tmux(tmp_path: Path) -> None:
    env, _model, provenance, marker = _stage1_identity_env(tmp_path)
    provenance.write_text('{"target_dir":"/wrong"}\n')
    result = subprocess.run(["bash", str(RUNNER), "stage1"], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode != 0 and "hash mismatch" in result.stderr
    assert not marker.exists()


def test_stage1_ignores_external_identity_override(tmp_path: Path) -> None:
    env, _model, _provenance, marker = _stage1_identity_env(tmp_path)
    result = subprocess.run(["bash", str(RUNNER), "stage1"], cwd=ROOT, env=env, text=True, capture_output=True)
    assert "attacker-model" not in result.stderr
    assert marker.exists()


def test_stage2_does_not_require_stage1_identity(tmp_path: Path) -> None:
    env, _model, _provenance, marker = _stage1_identity_env(tmp_path)
    env.pop("STAGE1_INIT_MODEL_PATH"); env.pop("STAGE1_INIT_PROVENANCE_PATH")
    result = subprocess.run(["bash", str(RUNNER), "stage2"], cwd=ROOT, env=env, text=True, capture_output=True)
    assert "Stage1" not in result.stderr
    assert marker.exists()


def test_queue_runs_bootstrap_then_freezes_contract_then_acceptance(tmp_path: Path) -> None:
    log = tmp_path / "order.log"
    fake = tmp_path / "fake_runner.py"
    _fake_runner(fake, log)
    result = subprocess.run(
        ["bash", str(QUEUE)],
        cwd=ROOT,
        env=_queue_env(tmp_path, fake),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    lines = log.read_text().splitlines()
    manifest = yaml.safe_load((ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml").read_text())
    required = "|".join([manifest["paths"]["base_model"], *[item["path"] for item in manifest["calibration_workloads"]["stage1"]["datasets"]]])
    expected_bootstrap = [f"bootstrap {phase} {rep} {required}" for phase in ("stage1", "stage2", "stage3") for rep in range(6)]
    expected_acceptance = [f"acceptance {phase} {rep} {required}" for phase in ("stage1", "stage2", "stage3") for rep in range(3)]
    assert lines == expected_bootstrap + expected_acceptance
    history = tmp_path / "history/trusted_history.json"
    contract = tmp_path / "prediction/prediction_contract.json"
    assert history.is_file()
    assert contract.is_file()
    history_doc = json.loads(history.read_text())
    first = history_doc["runs"][0]
    assert first["metrics"]["validation_elapsed_seconds"] == 100
    assert first["sampled_decoding_semantic_hash"]
    assert first["phase_topology_hash"]
    assert first["scorer_hash"]
    assert first["timeout_policy_hash"]
    assert first["max_response_length"] == 8192
    assert first["outcome_schema_version"] == 2
    assert len(first["workload_descriptor_sha256"]) == 64
    assert first["metrics"]["submitted_item_count"] == 1379
    assert first["metrics"]["response_length_p95_tokens"] == 100
    assert json.loads(contract.read_text())["algorithm_version"] == "stage123_history_conformal_v1"


def test_stage12_queue_builds_phase_scoped_history_and_contract(tmp_path: Path) -> None:
    log = tmp_path / "order.log"; fake = tmp_path / "fake_runner.py"; _fake_runner(fake, log)
    env = _queue_env(tmp_path, fake); env["CALIBRATION_AUTHORIZATION_SCOPE"] = "stage12_producer"
    result = subprocess.run(["bash", str(QUEUE)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr + result.stdout
    lines = log.read_text().splitlines()
    manifest = yaml.safe_load((ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml").read_text())
    required = "|".join([manifest["paths"]["base_model"], *[item["path"] for item in manifest["calibration_workloads"]["stage1"]["datasets"]]])
    assert lines == [f"bootstrap {phase} {rep} {required}" for phase in ("stage1", "stage2") for rep in range(6)] + [f"acceptance {phase} {rep} {required}" for phase in ("stage1", "stage2") for rep in range(3)]
    history = json.loads((tmp_path / "history/trusted_history.json").read_text())
    contract = json.loads((tmp_path / "prediction/prediction_contract.json").read_text())
    assert history["phase_scope"] == ["stage1", "stage2"]
    assert {run["phase"] for run in history["runs"]} == {"stage1", "stage2"}
    assert [phase["phase"] for phase in contract["phases"]] == ["stage1", "stage2"]


def test_queue_stops_on_first_failed_calibration_rep(tmp_path: Path) -> None:
    log = tmp_path / "order.log"
    fake = tmp_path / "fake_runner.py"
    _fake_runner(fake, log, fail_on="bootstrap:stage1:2")
    result = subprocess.run(
        ["bash", str(QUEUE)],
        cwd=ROOT,
        env=_queue_env(tmp_path, fake),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    manifest = yaml.safe_load((ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml").read_text())
    required = "|".join([manifest["paths"]["base_model"], *[item["path"] for item in manifest["calibration_workloads"]["stage1"]["datasets"]]])
    assert log.read_text().splitlines() == [f"bootstrap stage1 {rep} {required}" for rep in range(3)]
    assert not (tmp_path / "prediction/prediction_contract.json").exists()


def test_queue_rejects_diagnostic_only_calibration_roots(tmp_path: Path) -> None:
    fake = tmp_path / "fake_runner.py"
    _fake_runner(fake, tmp_path / "order.log")
    for name in ("af1a407f", "baaa596b_v2", "7c1ed4e1_v3"):
        env = _queue_env(tmp_path, fake)
        env["CALIBRATION_REPORT_ROOT"] = str(tmp_path / name / "report")
        result = subprocess.run(["bash", str(QUEUE)], cwd=ROOT, env=env, text=True, capture_output=True, check=False)
        assert result.returncode != 0
        assert "diagnostic-only calibration root is ineligible" in result.stderr


def test_stage1_and_stage3_share_canonical_reward_manager_timeout() -> None:
    script = (ROOT / "recipe/on_policy_wdl_sft/code_task/run_s1_code_base.sh").read_text()
    assert 'export REWARD_TIMEOUT=${REWARD_TIMEOUT:-$CODE_REWARD_MANAGER_TIMEOUT}' in script
    assert 'REWARD_TIMEOUT must equal CODE_REWARD_MANAGER_TIMEOUT' in script
    assert 'export REWARD_TIMEOUT=${REWARD_TIMEOUT:-60}' not in script
