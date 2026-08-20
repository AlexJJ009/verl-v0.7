# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
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


def test_recording_observer_preserves_validation_phase_order() -> None:
    spec = importlib.util.spec_from_file_location("ray_trainer_timing", TRAINER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    observer = module.RecordingValidationObserver()
    observer.record("batch_started", batch_index=1, total_batches=1)
    observer.record("generation_complete", batch_index=1, total_batches=1)
    observer.record("metrics_complete", elapsed_seconds=1.0)
    assert [item["event"] for item in observer.events] == ["batch_started", "generation_complete", "metrics_complete"]
