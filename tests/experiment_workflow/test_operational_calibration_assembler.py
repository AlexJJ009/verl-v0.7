from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load():
    path = ROOT / "scripts/assemble_code_task_operational_calibration.py"
    spec = importlib.util.spec_from_file_location("calassemble", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_validation_provenance_requires_all_three_full_files(tmp_path, monkeypatch):
    module = load()
    paths = {}
    hashes = {}
    rows = {}
    for name, expected in module.VALIDATION_DATASETS.items():
        path = tmp_path / f"{name}.parquet"
        path.write_bytes(name.encode())
        paths[name] = path
        hashes[path] = expected["sha256"]
        rows[path] = expected["rows"]
    monkeypatch.setattr(module, "sha256", lambda path: hashes[path])
    monkeypatch.setattr(module, "parquet_rows", lambda path: rows[path])

    result = module.validation_provenance(paths)
    assert result["scope"] == "full"
    assert result["total_rows"] == 1422
    assert [item["name"] for item in result["datasets"]] == ["HumanEval+", "MBPP+", "LiveCodeBench"]

    rows[paths["HumanEval+"]] = 64
    try:
        module.validation_provenance(paths)
    except ValueError as exc:
        assert "expected full validation" in str(exc)
    else:
        raise AssertionError("sampled validation data was accepted")


def test_scorer_evidence_reports_latency_rates_throughput_and_distributions(tmp_path):
    module = load()
    root = tmp_path / "rep1"
    generation = root / "stage2/logs/validation/run/0.jsonl"
    generation.parent.mkdir(parents=True)
    generation.write_text(
        "\n".join(
            [
                '{"score": 1.0, "code_reward_latency_seconds": 1.0, "code_reward_timeout": 0, "code_reward_status": "passed"}',
                '{"score": -1.0, "code_reward_latency_seconds": 3.0, "code_reward_timeout": 1, "code_reward_status": "timeout"}',
            ]
        )
        + "\n"
    )
    path, evidence = module.load_scorer_evidence(root, "stage2", 12.0)
    assert path == generation
    assert evidence["sample_count"] == 2
    assert evidence["scorer_latency_seconds_p50"] == 2.0
    assert evidence["scorer_latency_seconds_p95"] == pytest.approx(2.9)
    assert evidence["timeout_rate"] == 0.5
    assert evidence["valid_scores_per_minute"] == 10.0
    assert evidence["score_distribution"] == {"-1": 1, "1": 1}


def test_content_sha256_is_stable_for_model_directories(tmp_path):
    module = load()
    model = tmp_path / "model"
    (model / "nested").mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "nested/weights.bin").write_bytes(b"weights")
    first = module.content_sha256(model)
    (model / "nested/weights.bin").touch()
    assert module.content_sha256(model) == first
    (model / "nested/weights.bin").write_bytes(b"changed")
    assert module.content_sha256(model) != first


@pytest.mark.parametrize(
    ("phase", "trainer_elapsed", "ready", "generated", "complete", "expected_prep"),
    [
        ("stage1", 1058.550787596032, 11424107.405553587, 11425162.633636873, 11425165.421732359, 0.534608824),
        ("stage2", 88.79418030567467, 11431533.714413784, 11431607.524541158, 11431614.295717072, 8.212877017),
        ("stage3", 119.36518903635442, 11433294.035499893, 11433410.522387052, 11433412.75362399, 0.647064939),
    ],
)
def test_validation_timeline_uses_readiness_to_metrics_interval(
    tmp_path, phase, trainer_elapsed, ready, generated, complete, expected_prep
):
    module = load()
    root = tmp_path / "rep"
    path = root / f"{phase}.validation_timeline.jsonl"
    root.mkdir()
    path.write_text(
        "\n".join(
            [
                f'{{"event":"validation_ready","monotonic_seconds":{ready}}}',
                f'{{"event":"generation_complete","monotonic_seconds":{generated}}}',
                f'{{"event":"metrics_complete","monotonic_seconds":{complete}}}',
            ]
        )
        + "\n"
    )
    _, timeline = module.load_validation_timeline(root, phase, trainer_elapsed)
    assert timeline["validation_elapsed_seconds"] == pytest.approx(complete - ready)
    assert timeline["timeline_elapsed_seconds"] == pytest.approx(complete - ready)
    assert timeline["trainer_validation_elapsed_seconds"] == trainer_elapsed
    assert timeline["pre_readiness_elapsed_seconds"] == pytest.approx(expected_prep, abs=1e-6)


def test_validation_timeline_rejects_trainer_timer_shorter_than_canonical_interval(tmp_path):
    module = load()
    root = tmp_path / "rep"
    root.mkdir()
    (root / "stage2.validation_timeline.jsonl").write_text(
        '\n'.join([
            '{"event":"validation_ready","monotonic_seconds":10}',
            '{"event":"generation_complete","monotonic_seconds":14}',
            '{"event":"metrics_complete","monotonic_seconds":16}',
        ]) + '\n'
    )
    with pytest.raises(ValueError, match="does not contain canonical validation interval"):
        module.load_validation_timeline(root, "stage2", 5.99)


def test_load_rep_preserves_resource_measurement_contract(tmp_path, monkeypatch):
    module = load()
    root = tmp_path / "rep"
    (root / "stage2/logs/metrics/OnPolicyWDLSFT-CodeTask").mkdir(parents=True)
    (root / "stage2/logs/validation/run").mkdir(parents=True)
    (root / "stage2.status.json").write_text('{"returncode":0,"timed_out":false}')
    (root / "stage2.resources.json").write_text(
        '{"peak_rss_gib":1,"gpu_wait_fraction":0.1,"gpu_sample_count":10,'
        '"gpu_idle_sample_count":1,"gpu_idle_threshold_pct":2,"memory_source":"cgroup",'
        '"measurement_started":true,"measurement_window":"validation_rollout_readiness_to_completion",'
        '"gpu_sample_interval_seconds":0.2,"readiness_wait_seconds":5}'
    )
    (root / "stage2.validation_timeline.jsonl").write_text(
        '\n'.join([
            '{"event":"validation_ready","monotonic_seconds":10}',
            '{"event":"generation_complete","monotonic_seconds":11}',
            '{"event":"metrics_complete","monotonic_seconds":12}',
        ]) + '\n'
    )
    monkeypatch.setattr(module, "load_metrics", lambda *_: (tmp_path / "metrics", {"timing_s/testing": 2.5, **{k: 0 for k in module.CORE_METRICS}}))
    monkeypatch.setattr(module, "load_scorer_evidence", lambda *_: (tmp_path / "generation", {}))
    monkeypatch.setattr(module, "sha256", lambda _: "hash")
    rep = module.load_rep(root, "stage2", True)
    assert rep["metrics"]["validation_elapsed_seconds"] == 2
    assert rep["metrics"]["trainer_validation_elapsed_seconds"] == 2.5
    assert rep["resources"]["measurement_started"] is True
    assert rep["resources"]["gpu_sample_interval_seconds"] == 0.2


def test_aggregate_phase_uses_frozen_contract_and_preserves_acceptance_raw_values(monkeypatch):
    module = load()
    outcome_metrics = {
        "response_length_p50_tokens": 100,
        "response_length_p95_tokens": 200,
        "truncation_rate": 0.0,
        "scorer_latency_p50_seconds": 1.0,
        "scorer_latency_p95_seconds": 2.0,
        "scorer_timeout_rate": 0.0,
    }
    reps = {
        "predictor": {
            "warmup": True,
            "metrics": {"validation_elapsed_seconds": 999, "complete_validation_metrics": True, **outcome_metrics},
            "resources": {"peak_rss_gib": 999, "gpu_wait_fraction": 0.99},
        },
        "r1": {
            "warmup": False,
            "metrics": {"validation_elapsed_seconds": 100, "complete_validation_metrics": True, **outcome_metrics},
            "resources": {"peak_rss_gib": 100, "gpu_wait_fraction": 0.52},
        },
        "r2": {
            "warmup": False,
            "metrics": {"validation_elapsed_seconds": 110, "complete_validation_metrics": True, **outcome_metrics},
            "resources": {"peak_rss_gib": 130, "gpu_wait_fraction": 0.55},
        },
        "r3": {
            "warmup": False,
            "metrics": {"validation_elapsed_seconds": 120, "complete_validation_metrics": True, **outcome_metrics},
            "resources": {"peak_rss_gib": 110, "gpu_wait_fraction": 0.58},
        },
    }

    def fake_load_rep(root, phase, warmup, workload=None):
        return reps[root.name]

    monkeypatch.setattr(module, "load_rep", fake_load_rep)
    result = module.aggregate_phase(
        "stage1",
        "profile",
        [Path("predictor")],
        [Path("r1"), Path("r2"), Path("r3")],
        {"path": "model", "sha256": "model"},
        {
            "status": "deployable",
            "decision": "deployable",
            "cohort_size": 6,
            "eligible_run_ids": ["h1"],
            "failures": [],
            "predictions": {
                "validation_elapsed_seconds": {"point": 111, "interval": [90, 130]},
                "peak_rss_gib": {"point": 112, "interval": [90, 140]},
                "response_length_p50_tokens": {"point": 100, "interval": [90, 110]},
                "response_length_p95_tokens": {"point": 200, "interval": [190, 210]},
                "scorer_latency_p50_seconds": {"point": 1, "interval": [0.5, 1.5]},
                "scorer_latency_p95_seconds": {"point": 2, "interval": [1.5, 2.5]},
                "truncation_rate": {"point": 0, "interval": [0, 0.01]},
                "scorer_timeout_rate": {"point": 0, "interval": [0, 0.01]},
                "all_gpu_idle_fraction_during_validation": {"interval": [0.5, 0.6]},
            },
        },
        {"phase": "stage1", "outcome_schema_version": 2},
    )
    assert result["predicted"]["validation_elapsed_seconds"] == 111
    assert result["observed"]["validation_elapsed_seconds"] == 110
    assert result["observed"]["peak_rss_gib"] == 110
    assert result["acceptance_repetition_values"]["peak_rss_gib"] == [100, 130, 110]
    assert result["acceptance_aggregate"]["method"] == "median_of_three_valid_repetitions"
    assert result["prediction_intervals"]["all_gpu_idle_fraction_during_validation"] == [0.5, 0.6]
    assert result["outcome_schema_version"] == 2
    assert result["workload_descriptor_sha256"]


def test_queue_native_layout_has_zero_predictors_and_exact_acceptance_roots(tmp_path):
    module = load()
    for phase in module.PHASES:
        for index in range(3):
            (tmp_path / "acceptance" / phase / f"rep_{index}").mkdir(parents=True)
    predictors, measured = module.resolve_queue_roots(tmp_path)
    assert predictors == {phase: [] for phase in module.PHASES}
    assert [path.name for path in measured["stage2"]] == ["rep_0", "rep_1", "rep_2"]


@pytest.mark.parametrize("mutation", ["missing", "extra", "legacy"])
def test_queue_native_layout_rejects_missing_extra_and_legacy_roots(tmp_path, mutation):
    module = load()
    for phase in module.PHASES:
        for index in range(3):
            (tmp_path / "acceptance" / phase / f"rep_{index}").mkdir(parents=True)
    if mutation == "missing":
        (tmp_path / "acceptance/stage1/rep_2").rmdir()
    elif mutation == "extra":
        (tmp_path / "acceptance/stage2/rep_3").mkdir()
    else:
        (tmp_path / "stage3/rep0_predictor").mkdir(parents=True)
    with pytest.raises(ValueError):
        module.resolve_queue_roots(tmp_path)
