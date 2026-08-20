# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    RecordingValidationObserver,
    build_response_telemetry,
    build_validation_generation_samples,
    validation_sample_identities,
)
from verl.utils.tracking import ValidationGenerationsLogger


def test_build_validation_generation_samples_keeps_reward_metadata():
    samples = build_validation_generation_samples(
        inputs=["prompt-1", "prompt-2"],
        outputs=["response-1", "response-2"],
        ground_truths=["42", "43"],
        scores=[-1.0, 1.0],
        data_sources=["gsm8k", "gsm8k"],
        sample_uids=["uid-1", "uid-2"],
        reward_extra_infos_dict={
            "acc": [0.0, 1.0],
            "pred": ["[NO_BOXED]", "43"],
            "verification_method": ["no_answer", "string_match"],
            "has_eos": [True, True],
        },
    )

    assert len(samples) == 2
    assert samples[0]["sample_index"] == 0
    assert samples[0]["ground_truth"] == "42"
    assert samples[0]["pred"] == "[NO_BOXED]"
    assert samples[1]["verification_method"] == "string_match"


def test_build_response_telemetry_uses_native_ids_mask_and_eos():
    result = build_response_telemetry(
        torch.tensor([[10, 2, 0, 0], [11, 12, 13, 14], [20, 21, 0, 0]]),
        torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0]]),
        eos_token_id=2,
        max_response_length=4,
    )
    assert result == {
        "response_token_count": [2, 4, 2],
        "response_eos_present": [True, False, False],
        "response_finish_reason": ["stop", "length", "unknown"],
    }


def test_build_response_telemetry_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same rank-2 shape"):
        build_response_telemetry(
            torch.ones(2, 4, dtype=torch.long),
            torch.ones(2, 3, dtype=torch.long),
            eos_token_id=2,
            max_response_length=4,
        )


def test_validation_generations_logger_uses_row_per_sample_table_shape():
    logger = ValidationGenerationsLogger()
    columns, rows = logger._tabularize_samples(
        [
            {
                "sample_index": 3,
                "data_source": "gsm8k",
                "uid": "uid-1",
                "input": "prompt",
                "output": "response",
                "ground_truth": "42",
                "score": -1.0,
                "verification_method": "no_answer",
            },
            {
                "sample_index": 4,
                "data_source": "gsm8k",
                "uid": "uid-2",
                "input": "prompt-2",
                "output": "response-2",
                "ground_truth": "7",
                "score": 1.0,
                "verification_method": "string_match",
            },
        ],
        step=12,
    )

    assert columns[:8] == [
        "step",
        "sample_index",
        "data_source",
        "uid",
        "input",
        "output",
        "ground_truth",
        "score",
    ]
    assert "verification_method" in columns
    assert len(rows) == 2
    assert rows[0][columns.index("step")] == 12
    assert rows[0][columns.index("sample_index")] == 3
    assert rows[1][columns.index("score")] == 1.0


def test_dump_generations_writes_stable_uid(tmp_path):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.global_steps = 3
    trainer._dump_generations(
        inputs=["prompt"],
        outputs=["response"],
        gts=["answer"],
        scores=[1.0],
        reward_extra_infos_dict={},
        dump_path=str(tmp_path),
        data_sources=["HumanEval+"],
        sample_uids=["he-1"],
    )
    import json

    row = json.loads((tmp_path / "3.jsonl").read_text())
    assert row["uid"] == "he-1"
    assert row["data_source"] == "HumanEval+"


def test_dump_generations_serializes_numpy_reward_metadata(tmp_path):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.global_steps = 1
    trainer._dump_generations(
        inputs=["prompt"],
        outputs=["response"],
        gts=[{"expected": np.int64(1)}],
        scores=[np.float32(-1.0)],
        reward_extra_infos_dict={
            "format_contract_success": [np.bool_(False)],
            "nested": [{"runtime_ok": np.bool_(True)}],
        },
        dump_path=str(tmp_path),
    )
    import json

    row = json.loads((tmp_path / "1.jsonl").read_text())
    assert row["score"] == -1.0
    assert row["gts"] == {"expected": 1}
    assert row["format_contract_success"] is False
    assert row["nested"] == {"runtime_ok": True}


def test_source_uid_survives_repeat_pad_and_unpad_in_order():
    proto = DataProto.from_dict(
        tensors={"dummy": torch.tensor([[1], [2], [3]])},
        non_tensors={
            "uid": np.array(["request-a", "request-b", "request-c"]),
            "source_uid": np.array(["source-a", "source-b", "source-c"]),
        },
    )
    repeated = proto.repeat(repeat_times=2, interleave=True)
    padded, pad_size = pad_dataproto_to_divisor(repeated, 4)
    restored = unpad_dataproto(padded, pad_size)
    assert restored.non_tensor_batch["uid"].tolist() == [
        "request-a",
        "request-a",
        "request-b",
        "request-b",
        "request-c",
        "request-c",
    ]
    assert restored.non_tensor_batch["source_uid"].tolist() == [
        "source-a",
        "source-a",
        "source-b",
        "source-b",
        "source-c",
        "source-c",
    ]


def test_validation_identity_falls_back_to_legacy_uid():
    identities = validation_sample_identities({"uid": np.array(["request-a", "request-b"])})
    assert identities.tolist() == ["request-a", "request-b"]


def test_validation_identity_prefers_stable_source_uid_when_present():
    identities = validation_sample_identities(
        {
            "uid": np.array(["request-a", "request-b"]),
            "source_uid": np.array(["source-a", "source-b"]),
        }
    )
    assert identities.tolist() == ["source-a", "source-b"]


def test_recording_validation_observer_preserves_stable_identity_and_events():
    observer = RecordingValidationObserver()
    identities = validation_sample_identities({"uid": np.array(["request-a"]), "source_uid": np.array(["source-a"])})
    observer.record("batch_started", sample_identities=identities.tolist())
    observer.record("generation_complete", batch_index=1, total_batches=1)
    observer.record("metrics_complete", elapsed_seconds=1.25)
    assert observer.events == [
        {"event": "batch_started", "sample_identities": ["source-a"]},
        {"event": "generation_complete", "batch_index": 1, "total_batches": 1},
        {"event": "metrics_complete", "elapsed_seconds": 1.25},
    ]


def test_maybe_log_val_generations_prints_subset_and_logs_full_tracking(capsys):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.global_steps = 7
    logged = {}

    class _StubValidationLogger:
        def log(self, loggers, samples, step):
            logged["loggers"] = loggers
            logged["samples"] = samples
            logged["step"] = step

    trainer.validation_generations_logger = _StubValidationLogger()
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "log_val_generations": 1,
                "log_val_generations_to_tracking": -1,
                "logger": ["wandb", "file"],
            }
        }
    )

    trainer._maybe_log_val_generations(
        [
            {
                "sample_index": 0,
                "data_source": "gsm8k",
                "uid": "uid-1",
                "input": "prompt-1",
                "output": "response-1",
                "ground_truth": "42",
                "score": -1.0,
                "verification_method": "no_answer",
            },
            {
                "sample_index": 1,
                "data_source": "gsm8k",
                "uid": "uid-2",
                "input": "prompt-2",
                "output": "response-2",
                "ground_truth": "43",
                "score": 1.0,
                "verification_method": "string_match",
            },
        ]
    )

    out = capsys.readouterr().out
    assert "Validation generations at step 7" in out
    assert out.count("[validation sample") == 1
    assert "verification_method:" in out
    assert logged["loggers"] == ["wandb", "file"]
    assert len(logged["samples"]) == 2
    assert logged["step"] == 7
