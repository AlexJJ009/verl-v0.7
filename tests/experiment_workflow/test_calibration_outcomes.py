from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load():
    path = ROOT / "scripts/calibration_outcomes.py"
    spec = importlib.util.spec_from_file_location("calibration_outcomes", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def workload(count=3):
    uid_doc = {"schema_version": 1, "datasets": [{"name": "test", "source_index": 0, "ordered_uids": [chr(97 + i) for i in range(count)]}]}
    uid_hash = hashlib.sha256((json.dumps(uid_doc, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    return {
        "datasets": [{"name": "test", "row_count": count}],
        "validation_eligibility": {
            "submitted_prompt_count": count,
            "per_dataset_eligible_counts": {"test": count},
            "ordered_eligible_uid_sha256": uid_hash,
        },
    }


def row(uid, count, eos, reason, latency, timeout=0):
    return {
        "uid": uid,
        "data_source": "test",
        "response_token_count": count,
        "response_eos_present": eos,
        "response_finish_reason": reason,
        "code_reward_latency_seconds": latency,
        "code_reward_timeout": timeout,
    }


def write(path, rows):
    path.write_text("\n".join(json.dumps(item) for item in rows) + "\n")


def test_outcomes_use_nearest_rank_and_submitted_denominator(tmp_path):
    module = load()
    path = tmp_path / "generation.jsonl"
    write(path, [row("a", 10, True, "stop", 1), row("b", 20, True, "stop", 3, 1), row("c", 8192, False, "length", 2)])
    result = module.load_generation_outcomes(path, workload())
    assert result == {
        "submitted_item_count": 3,
        "response_length_p50_tokens": 20.0,
        "response_length_p95_tokens": 8192.0,
        "truncated_item_count": 1,
        "truncation_rate": 1 / 3,
        "truncation_by_dataset": {
            "test": {"submitted_item_count": 3, "truncated_item_count": 1, "truncation_rate": 1 / 3}
        },
        "scorer_latency_p50_seconds": 2.0,
        "scorer_latency_p95_seconds": 3.0,
        "scorer_timeout_rate": 1 / 3,
    }


@pytest.mark.parametrize(
    ("rows", "code"),
    [
        ([row("a", 10, False, "unknown", 1)], "submitted_row_count"),
        ([row("a", 10, False, "unknown", 1)] * 3, "stable_uid"),
        ([row("a", 10, False, "unknown", 1), row("b", 20, True, "stop", 1), row("c", 30, True, "stop", 1)], "finish_reason"),
        ([row("a", 10, False, "stop", 1), row("b", 20, True, "stop", 1), row("c", 30, True, "stop", 1)], "stop_without_eos"),
    ],
)
def test_outcomes_fail_closed_on_incomplete_evidence(tmp_path, rows, code):
    module = load()
    path = tmp_path / "generation.jsonl"
    write(path, rows)
    with pytest.raises(module.OutcomeValidationError) as raised:
        module.load_generation_outcomes(path, workload())
    assert raised.value.code == code
    assert raised.value.context["path"] == str(path)


def test_outcomes_reject_wrong_uid_order_and_dataset_identity(tmp_path):
    module = load(); path = tmp_path / "generation.jsonl"
    rows = [row("b", 10, True, "stop", 1), row("a", 20, True, "stop", 1), row("c", 30, True, "stop", 1)]
    write(path, rows)
    with pytest.raises(module.OutcomeValidationError) as raised:
        module.load_generation_outcomes(path, workload())
    assert raised.value.code == "uid_order_hash"


def test_outcomes_record_exact_truncation_counts_by_dataset(tmp_path):
    module = load(); path = tmp_path / "generation.jsonl"
    rows = [row("a", 8192, False, "length", 1), row("b", 20, True, "stop", 1), row("c", 30, True, "stop", 1)]
    write(path, rows)
    result = module.load_generation_outcomes(path, workload())
    assert result["truncated_item_count"] == 1
    assert result["truncation_rate"] == pytest.approx(1 / 3)
    assert result["truncation_by_dataset"]["test"] == {
        "submitted_item_count": 3,
        "truncated_item_count": 1,
        "truncation_rate": 1 / 3,
    }
    rows[0]["data_source"] = "unknown"
    write(path, rows)
    with pytest.raises(module.OutcomeValidationError) as raised:
        module.load_generation_outcomes(path, workload())
    assert raised.value.code == "data_source"
    assert raised.value.context["data_source"] == "unknown"
