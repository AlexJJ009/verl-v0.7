from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def canonical_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def workload(phase):
    return {"phase": phase, "outcome_schema_version": 2, "descriptor": phase}


def v2_metrics(elapsed, rss):
    return {
        "validation_elapsed_seconds": elapsed,
        "peak_rss_gib": rss,
        "response_length_p50_tokens": 100,
        "response_length_p95_tokens": 200,
        "truncation_rate": 0.0,
        "scorer_latency_p50_seconds": 1.0,
        "scorer_latency_p95_seconds": 2.0,
        "scorer_timeout_rate": 0.0,
        "submitted_item_count": 1422,
    }


def load():
    p = ROOT / "scripts/check_code_task_operational_calibration.py"
    s = importlib.util.spec_from_file_location("calcheck", p)
    m = importlib.util.module_from_spec(s)
    assert s.loader
    s.loader.exec_module(m)
    return m


def phase_prediction(phase, elapsed_interval=(90, 120), rss_interval=(90, 120), gpu_interval=(0.30, 0.80)):
    return {
        "status": "deployable",
        "decision": "deployable",
        "cohort_size": 6,
        "features": workload(phase),
        "predictions": {
            "validation_elapsed_seconds": {"point": 100, "interval": list(elapsed_interval), "loo_coverage": 0.9},
            "peak_rss_gib": {"point": 100, "interval": list(rss_interval), "loo_coverage": 0.9},
            "response_length_p50_tokens": {"point": 100, "interval": [90, 110], "loo_coverage": 0.9},
            "response_length_p95_tokens": {"point": 200, "interval": [190, 210], "loo_coverage": 0.9},
            "scorer_latency_p50_seconds": {"point": 1, "interval": [0.5, 1.5], "loo_coverage": 0.9},
            "scorer_latency_p95_seconds": {"point": 2, "interval": [1.5, 2.5], "loo_coverage": 0.9},
            "truncation_rate": {"point": 0, "interval": [0, 0.01]},
            "scorer_timeout_rate": {"point": 0, "interval": [0, 0.01]},
            "all_gpu_idle_fraction_during_validation": {"interval": list(gpu_interval)},
        },
    }


def fixture(tmp_path):
    manifest = {
        "manifest_sha256": "rendered-manifest",
        "resource_profile": {"sha256": "profile-hash"},
        "calibration_receipt_max_age_seconds": 86400,
        "calibration_workloads": {phase: workload(phase) for phase in ("stage1", "stage2", "stage3")},
    }
    contract = {
        "decision": "deployable",
        "phases": [{"phase": name, **phase_prediction(name)} for name in ("stage1", "stage2", "stage3")],
    }
    phases = []
    for name in ("stage1", "stage2", "stage3"):
        reps = []
        for i in range(3):
            artifacts = {}
            for kind in ("status", "resources", "metrics", "generation", "timeline"):
                path = tmp_path / f"{name}-{i}-{kind}.json"
                path.write_text(json.dumps({"kind": kind}))
                artifacts[kind] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            scorer = {
                "sample_count": 1422,
                "scorer_latency_seconds_p50": 0.2,
                "scorer_latency_seconds_p95": 2.0,
                "timeout_count": 0,
                "timeout_rate": 0.0,
                "invalid_score_count": 0,
                "invalid_score_rate": 0.0,
                "valid_score_count": 1422,
                "valid_score_rate": 1.0,
                "valid_scores_per_minute": 775.6,
                "score_distribution": {"-1": 1000, "1": 422},
                "status_distribution": {"wrong_answer": 1000, "passed": 422},
            }
            reps.append(
                {
                    "warmup": False,
                    "status": {"returncode": 0, "timed_out": False},
                    "metrics": {"complete_validation_metrics": True, **v2_metrics([105, 110, 115][i], [105, 110, 115][i])},
                    "scorer": scorer,
                    "timeline": {"rollout_elapsed_seconds": 80, "post_generation_elapsed_seconds": 30, "timeline_elapsed_seconds": 110},
                    "resources": {
                        "peak_rss_gib": [105, 110, 115][i],
                        "gpu_wait_fraction": [0.52, 0.55, 0.58][i],
                        "measurement_started": True,
                        "measurement_window": "validation_rollout_readiness_to_completion",
                        "gpu_sample_interval_seconds": 0.2,
                        "gpu_sample_count": 540,
                    },
                    "artifacts": artifacts,
                }
            )
        phases.append(
            {
                "phase": name,
                "profile_hash": "profile-hash",
                "workload_descriptor_sha256": canonical_sha(workload(name)),
                "outcome_schema_version": 2,
                "optimized": True,
                "predicted": {
                    "validation_elapsed_seconds": 100,
                    "peak_rss_gib": 100,
                    "response_length_p50_tokens": 100,
                    "response_length_p95_tokens": 200,
                    "scorer_latency_p50_seconds": 1,
                    "scorer_latency_p95_seconds": 2,
                    "truncation_rate": 0,
                    "scorer_timeout_rate": 0,
                    "all_gpu_idle_fraction_during_validation": 0.5,
                },
                "observed": {
                    "complete_validation_metrics": True,
                    "validation_elapsed_seconds": 110,
                    "peak_rss_gib": 110,
                    "response_length_p50_tokens": 100,
                    "response_length_p95_tokens": 200,
                    "scorer_latency_p50_seconds": 1,
                    "scorer_latency_p95_seconds": 2,
                    "truncation_rate": 0,
                    "scorer_timeout_rate": 0,
                    "all_gpu_idle_fraction_during_validation": 0.55,
                    "maximum_validation_elapsed_seconds": 115,
                },
                "predictor_repetitions": [],
                "repetitions": reps,
            }
        )
    datasets = []
    for name, expected in load().VALIDATION_DATASETS.items():
        path = tmp_path / f"{name}.parquet"
        path.write_bytes(name.encode())
        datasets.append({"name": name, "path": str(path), **expected})
    report = {
        "evidence_class": "infrastructure_calibration",
        "manifest_sha256": "rendered-manifest",
        "decision": "candidate",
        "prediction_contract_decision": "deployable",
        "contract": {
            "val_max_samples": -1,
            "validation_deadline_seconds": 1800,
            "predictor_repetitions": {"stage1": 0, "stage2": 0, "stage3": 0},
        },
        "input_bindings": {"resource_profile": {"identity": "local-l40s", "sha256": "profile-hash"}},
        "validation_data": {"scope": "full", "datasets": datasets, "total_rows": 1422},
        "phases": phases,
        "queue_identity": "stage123-formal",
    }
    return manifest, report, contract


def patch_dataset_hashes(module, report):
    real_sha = module.file_sha256
    module.file_sha256 = lambda path: next(
        (item["sha256"] for item in report["validation_data"]["datasets"] if item["path"] == str(path)),
        real_sha(path),
    )


def test_deployable_candidate_and_blocked_boundaries(tmp_path):
    m = load()
    manifest, report, contract = fixture(tmp_path)
    patch_dataset_hashes(m, report)
    assert m.check(report, manifest, contract=contract)["ok"]
    report["phases"][0]["observed"]["maximum_validation_elapsed_seconds"] = 1801
    result = m.check(report, manifest, contract=contract)
    assert not result["ok"] and result["decision"] == "blocked"


def test_failed_result_does_not_create_or_overwrite_receipt(tmp_path, monkeypatch):
    m = load()
    manifest, report, contract = fixture(tmp_path)
    patch_dataset_hashes(m, report)
    report["phases"][0]["observed"]["maximum_validation_elapsed_seconds"] = 1801
    manifest_path = tmp_path / "manifest.json"; manifest_path.write_text(json.dumps(manifest))
    report_path = tmp_path / "report.json"; report_path.write_text(json.dumps(report))
    history_path = tmp_path / "history.json"; history_path.write_text("{}")
    receipt_path = tmp_path / "receipt.json"; receipt_path.write_text("preserve-existing")
    monkeypatch.setattr(m, "check_file_binding", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(m, "check", lambda *_args, **_kwargs: {"ok": False, "decision": "blocked", "failures": ["blocked"], "inconclusive_reasons": []})
    monkeypatch.setattr(sys, "argv", [
        "checker", "--report", str(report_path), "--manifest", str(manifest_path),
        "--history-index", str(history_path), "--receipt", str(receipt_path),
    ])
    assert m.main() == 1
    assert receipt_path.read_text() == "preserve-existing"


def test_assembler_must_not_self_declare_deployable(tmp_path):
    m = load()
    manifest, report, contract = fixture(tmp_path)
    patch_dataset_hashes(m, report)
    report["decision"] = "deployable"
    result = m.check(report, manifest, contract=contract)
    assert not result["ok"]
    assert "assembler report decision must be candidate" in result["failures"]


def test_repetition_and_artifact_tampering_are_blocked(tmp_path):
    m = load()
    manifest, report, contract = fixture(tmp_path)
    patch_dataset_hashes(m, report)
    report["phases"][0]["predictor_repetitions"] = [report["phases"][0]["repetitions"][0]]
    assert not m.check(report, manifest, contract=contract)["ok"]
    manifest, report, contract = fixture(tmp_path)
    patch_dataset_hashes(m, report)
    Path(report["phases"][0]["repetitions"][0]["artifacts"]["metrics"]["path"]).write_text("tampered")
    assert not m.check(report, manifest, contract=contract)["ok"]


def test_full_validation_contract_rejections(tmp_path):
    m = load()
    manifest, report, contract = fixture(tmp_path)
    patch_dataset_hashes(m, report)
    mutations = (
        lambda r: r["validation_data"].__setitem__("scope", "sampled"),
        lambda r: r["validation_data"]["datasets"].pop(),
        lambda r: r["validation_data"]["datasets"][0].__setitem__("rows", 64),
        lambda r: r["validation_data"]["datasets"][0].__setitem__("sha256", "bad"),
        lambda r: r["contract"].__setitem__("val_max_samples", 64),
    )
    for mutate in mutations:
        _, candidate, contract = fixture(tmp_path)
        patch_dataset_hashes(m, candidate)
        mutate(candidate)
        assert not m.check(candidate, manifest, contract=contract)["ok"]


def test_timeout_deadline_and_prediction_error_are_blocked(tmp_path):
    m = load()
    manifest, report, contract = fixture(tmp_path)
    patch_dataset_hashes(m, report)
    report["phases"][0]["repetitions"][1]["status"]["timed_out"] = True
    assert not m.check(report, manifest, contract=contract)["ok"]


def test_invalid_resource_measurement_window_is_blocked(tmp_path):
    m = load()
    manifest, report, contract = fixture(tmp_path)
    patch_dataset_hashes(m, report)
    report["phases"][0]["repetitions"][0]["resources"]["measurement_started"] = False
    assert not m.check(report, manifest, contract=contract)["ok"]


def test_coarse_or_incomplete_gpu_sampling_is_blocked(tmp_path):
    m = load()
    manifest, report, contract = fixture(tmp_path)
    patch_dataset_hashes(m, report)
    report["phases"][0]["repetitions"][0]["resources"]["gpu_sample_interval_seconds"] = 1
    assert not m.check(report, manifest, contract=contract)["ok"]
    _, report, contract = fixture(tmp_path)
    patch_dataset_hashes(m, report)
    report["phases"][0]["repetitions"][0]["resources"]["gpu_sample_count"] = 10
    assert not m.check(report, manifest, contract=contract)["ok"]
    _, report, contract = fixture(tmp_path)
    patch_dataset_hashes(m, report)
    report["phases"][0]["repetitions"][0]["resources"]["measurement_window"] = "process_start_to_exit"
    assert not m.check(report, manifest, contract=contract)["ok"]
    _, report, contract = fixture(tmp_path)
    patch_dataset_hashes(m, report)
    report["phases"][0]["repetitions"][1]["metrics"]["validation_elapsed_seconds"] = 1801
    assert not m.check(report, manifest, contract=contract)["ok"]
    _, report, contract = fixture(tmp_path)
    patch_dataset_hashes(m, report)
    report["phases"][0]["repetitions"][1]["resources"]["peak_rss_gib"] = 121
    assert not m.check(report, manifest, contract=contract)["ok"]


def test_elapsed_rss_all_values_and_median_must_fit_contract_interval(tmp_path):
    m = load()
    manifest, report, contract = fixture(tmp_path)
    patch_dataset_hashes(m, report)
    contract["phases"][0]["predictions"]["validation_elapsed_seconds"]["interval"] = [100, 110]
    result = m.check(report, manifest, contract=contract)
    assert not result["ok"]
    assert any("validation_elapsed_seconds acceptance value outside prediction interval" in failure for failure in result["failures"])


def test_v2_continuous_outcome_point_error_and_interval_are_blocked(tmp_path):
    m = load()
    for metric in ("response_length_p50_tokens", "response_length_p95_tokens", "scorer_latency_p50_seconds", "scorer_latency_p95_seconds"):
        manifest, report, contract = fixture(tmp_path)
        patch_dataset_hashes(m, report)
        for rep in report["phases"][0]["repetitions"]:
            rep["metrics"][metric] = contract["phases"][0]["predictions"][metric]["point"] * 1.25
        result = m.check(report, manifest, contract=contract)
        assert not result["ok"]
        assert any(metric in failure for failure in result["failures"])


@pytest.mark.parametrize(
    ("metric", "value", "message"),
    [("truncation_rate", 0.02, "truncation rate exceeds 1%"), ("scorer_timeout_rate", 0.11, "scorer timeout rate exceeds 10%")],
)
def test_v2_rate_thresholds_are_blocked(tmp_path, metric, value, message):
    m = load()
    manifest, report, contract = fixture(tmp_path)
    patch_dataset_hashes(m, report)
    contract["phases"][0]["predictions"][metric]["interval"] = [0, value]
    for rep in report["phases"][0]["repetitions"]:
        rep["metrics"][metric] = value
    result = m.check(report, manifest, contract=contract)
    assert not result["ok"]
    assert any(message in failure for failure in result["failures"])


def test_missing_v2_metric_and_workload_binding_are_blocked(tmp_path):
    m = load()
    manifest, report, contract = fixture(tmp_path)
    patch_dataset_hashes(m, report)
    del report["phases"][0]["repetitions"][0]["metrics"]["response_length_p95_tokens"]
    assert not m.check(report, manifest, contract=contract)["ok"]
    manifest, report, contract = fixture(tmp_path)
    patch_dataset_hashes(m, report)
    report["phases"][0]["workload_descriptor_sha256"] = "bad"
    assert not m.check(report, manifest, contract=contract)["ok"]


def test_gpu_idle_uses_measured_interval_overlap(tmp_path):
    m = load()
    manifest, report, contract = fixture(tmp_path)
    patch_dataset_hashes(m, report)
    contract["phases"][0]["predictions"]["all_gpu_idle_fraction_during_validation"]["interval"] = [0.0, 0.30]
    result = m.check(report, manifest, contract=contract)
    assert not result["ok"]
    assert any("GPU idle measured interval does not overlap" in failure for failure in result["failures"])


def test_prediction_contract_hash_and_decision_mismatch_are_blocked(tmp_path):
    m = load()
    manifest, report, contract = fixture(tmp_path)
    patch_dataset_hashes(m, report)
    report["prediction_contract_sha256"] = "old"
    report["input_bindings"]["prediction_contract"] = {"path": "contract.json", "sha256": "old"}
    result = m.check(report, manifest, contract=contract, hashes={"prediction_contract": "new"})
    assert not result["ok"]
    assert "prediction contract hash mismatch" in result["failures"]
    report["prediction_contract_sha256"] = "new"
    report["input_bindings"]["prediction_contract"]["sha256"] = "new"
    report["prediction_contract_decision"] = "blocked"
    result = m.check(report, manifest, contract=contract, hashes={"prediction_contract": "new"})
    assert not result["ok"]
    assert "prediction contract decision mismatch" in result["failures"]


def test_cli_writes_canonical_receipt_and_detects_preflight_tamper(tmp_path):
    m = load()
    manifest, report, contract = fixture(tmp_path)
    for phase in report["phases"]:
        phase["observed"]["validation_elapsed_seconds"] = 100
        phase["observed"]["peak_rss_gib"] = 100
        phase["observed"]["maximum_validation_elapsed_seconds"] = 100
        for rep in phase["repetitions"]:
            rep["metrics"]["validation_elapsed_seconds"] = 100
            rep["resources"]["peak_rss_gib"] = 100
    manifest.update(
        {
            "validation_dataset_hashes": {"HumanEval+": "h", "MBPP+": "m", "LiveCodeBench": "l"},
            "sampled_decoding_semantic_hash": "sem",
            "resource_profile_hash": "profile-hash",
            "phase_topology_hash": "topology",
            "scorer_hash": "scorer",
            "timeout_policy_hash": "timeout",
            "max_response_length": 8192,
            "calibration_workloads": {phase: workload(phase) for phase in ("stage1", "stage2", "stage3")},
        }
    )
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "report.json"
    contract_path = tmp_path / "prediction_contract.json"
    history_path = tmp_path / "history.json"
    policy_path = tmp_path / "policy.json"
    preflight_path = tmp_path / "preflight.json"
    receipt_path = tmp_path / "receipt.json"
    history_runs = []
    for phase in ("stage1", "stage2", "stage3"):
        for index, (elapsed, rss, idle) in enumerate(
            zip([100, 100, 100, 100, 100, 100], [100, 100, 100, 100, 100, 100], [0.08, 0.20, 0.31, 0.44, 0.58, 0.70])
        ):
            artifact_root = tmp_path / "history-artifacts" / "bootstrap" / phase / f"rep_{index}"
            artifact_root.mkdir(parents=True)
            bindings = {}
            for kind in ("status", "resources", "metrics", "generation", "timeline"):
                path = artifact_root / f"{phase}.{kind}.json"
                path.write_text(json.dumps({"phase": phase, "rep": index, "kind": kind}) + "\n")
                bindings[kind] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            history_runs.append(
                {
                    "run_id": f"{phase}-{index}",
                    "phase": phase,
                    "evidence_role": "bootstrap_history",
                    "artifact_root": str(artifact_root),
                    "artifact_bindings": bindings,
                    "completed_at": f"2026-07-{index + 1:02d}T00:00:00Z",
                    "release_gate_passed": True,
                    "content_addressed": True,
                    "artifacts_readable": True,
                    "validation_dataset_hashes": {"HumanEval+": "h", "MBPP+": "m", "LiveCodeBench": "l"},
                    "sampled_decoding_semantic_hash": "sem",
                    "resource_profile_hash": "profile-hash",
                    "phase_topology_hash": "topology",
                    "scorer_hash": "scorer",
                    "timeout_policy_hash": "timeout",
                    "max_response_length": 8192,
                    "workload_descriptor_sha256": canonical_sha(workload(phase)),
                    "outcome_schema_version": 2,
                    "metrics": {**v2_metrics(elapsed, rss), "all_gpu_idle_fraction_during_validation": idle},
                }
            )
    history_path.write_text(json.dumps({"cutoff_utc": "2026-07-20T00:00:00Z", "runs": history_runs}, sort_keys=True) + "\n")
    policy_path.write_text('{"policy":"v1"}\n')
    preflight_path.write_text('{"decision":"passed"}\n')
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    write_contract = subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/check_calibration_prediction_contract.py"),
            "--contract",
            str(contract_path),
            "--manifest",
            str(manifest_path),
            "--history-index",
            str(history_path),
            "--write",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert write_contract.returncode == 0, write_contract.stdout + write_contract.stderr
    contract = json.loads(contract_path.read_text())
    report["prediction_contract_sha256"] = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    report["prediction_contract_decision"] = contract["decision"]
    report["input_bindings"].update(
        {
            "manifest": {"path": str(manifest_path), "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()},
            "prediction_contract": {"path": str(contract_path), "sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest()},
            "history_index": {"path": str(history_path), "sha256": hashlib.sha256(history_path.read_bytes()).hexdigest()},
            "policy": {"path": str(policy_path), "sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest()},
            "preflight_receipt": {"path": str(preflight_path), "sha256": hashlib.sha256(preflight_path.read_bytes()).hexdigest()},
        }
    )
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")
    patch_dataset_hashes(m, report)
    argv = [
            "check_code_task_operational_calibration.py",
            "--report",
            str(report_path),
            "--manifest",
            str(manifest_path),
            "--contract",
            str(contract_path),
            "--policy",
            str(policy_path),
            "--history-index",
            str(history_path),
            "--preflight-receipt",
            str(preflight_path),
            "--receipt",
            str(receipt_path),
        ]
    old_argv = sys.argv
    stdout = io.StringIO()
    try:
        sys.argv = argv
        with redirect_stdout(stdout):
            code = m.main()
    finally:
        sys.argv = old_argv
    assert code == 0, stdout.getvalue()
    receipt = json.loads(receipt_path.read_text())
    assert receipt["decision"] == "deployable"
    assert receipt["ttl_seconds"] == 86400
    assert receipt_path.read_bytes().endswith(b"\n")
    preflight_path.write_text('{"decision":"tampered"}\n')
    stdout = io.StringIO()
    try:
        sys.argv = argv
        with redirect_stdout(stdout):
            code = m.main()
    finally:
        sys.argv = old_argv
    assert code == 1
    assert "preflight receipt hash mismatch" in stdout.getvalue()
