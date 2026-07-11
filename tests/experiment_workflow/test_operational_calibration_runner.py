from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_code_task_operational_calibration.sh"


def test_controlled_termination_requires_complete_validation_evidence() -> None:
    text = RUNNER.read_text()
    for key in (
        "timing_s/testing",
        "val-core/HumanEval+/acc/pass@1",
        "val-core/MBPP+/acc/pass@1",
        "val-core/LiveCodeBench/acc/pass@1",
    ):
        assert key in text
    assert 'if [ "\\$controlled_termination" = true ] && validation_complete; then rc=0; fi' in text
    assert 'controlled_termination_after_complete_validation' in text
    assert "docker kill '$CONTAINER'" in text
    assert "docker stop --time" not in text
