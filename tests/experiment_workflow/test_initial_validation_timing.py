from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "verl/trainer/ppo/ray_trainer.py"


def test_initial_validation_records_testing_wall_time() -> None:
    text = TRAINER.read_text()
    assert 'with marked_timer("testing", initial_validation_timing):' in text
    assert 'val_metrics["timing_s/testing"] = initial_validation_timing["testing"]' in text
    assert text.index('with marked_timer("testing", initial_validation_timing):') < text.index(
        'if self.config.trainer.get("val_only", False):'
    )
