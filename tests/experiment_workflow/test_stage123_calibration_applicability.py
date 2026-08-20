import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/stage123_calibration_applicability.py"
SPEC = importlib.util.spec_from_file_location("stage123_calibration_applicability", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_capacity_diff_is_empty_for_equal_snapshots() -> None:
    snapshot = {"profile": {"MAX_RESPONSE_LENGTH": "8192"}, "models": {"base_model": "qwen"}}
    assert MODULE.diff_snapshots(snapshot, snapshot) == []


def test_capacity_diff_fails_closed_on_changed_or_missing_field() -> None:
    source = {"profile": {"MAX_RESPONSE_LENGTH": "8192", "TRAIN_PROMPT_BSZ": "64"}}
    candidate = {"profile": {"MAX_RESPONSE_LENGTH": "4096"}}
    differences = MODULE.diff_snapshots(source, candidate)
    assert {item["field"] for item in differences} == {
        "profile.MAX_RESPONSE_LENGTH",
        "profile.TRAIN_PROMPT_BSZ",
    }


def test_source_calibration_requires_zero_training_and_cleanup() -> None:
    calibration = {
        "decision": "passed",
        "workload_identity": {"run_ids": MODULE.PRIMARY_RUN_IDS},
        "phase_evidence": [
            {
                "phase": phase,
                "status": "passed",
                "repetitions": [{"training_steps": 0, "status": "passed", "cleanup": {"resources_released": True}}],
            }
            for phase in ("stage1", "stage2", "stage3")
        ],
    }
    MODULE.validate_source_calibration(calibration)
    calibration["phase_evidence"][1]["repetitions"][0]["training_steps"] = 1
    try:
        MODULE.validate_source_calibration(calibration)
    except ValueError as error:
        assert "training" in str(error)
    else:
        raise AssertionError("training-bearing calibration must be rejected")
