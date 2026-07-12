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


def test_acceptance_requires_frozen_history_and_matching_prediction_contract() -> None:
    text = RUNNER.read_text()
    assert 'if [ "$CALIBRATION_ROLE" = acceptance ]; then' in text
    assert 'CALIBRATION_HISTORY_INDEX required for acceptance' in text
    assert 'CALIBRATION_PREDICTION_CONTRACT required for acceptance' in text
    assert 'check_calibration_prediction_contract.py' in text
    assert 'prediction contract history_index_sha256 mismatch' in text
    assert 'prediction contract manifest_sha256 mismatch' in text
    assert 'CALIBRATION_PREDICTION_CONTRACT_SHA256 mismatch' in text


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
    handle.write(f"{{role}} {{phase}} {{rep}}\\n")
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
    expected_bootstrap = [f"bootstrap {phase} {rep}" for phase in ("stage1", "stage2", "stage3") for rep in range(6)]
    expected_acceptance = [f"acceptance {phase} {rep}" for phase in ("stage1", "stage2", "stage3") for rep in range(3)]
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
    assert log.read_text().splitlines() == ["bootstrap stage1 0", "bootstrap stage1 1", "bootstrap stage1 2"]
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
