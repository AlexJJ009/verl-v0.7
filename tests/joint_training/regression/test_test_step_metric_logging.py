import numpy as np
import torch

from verl.trainer.ppo.ray_trainer import extract_test_step_metrics_for_logging


def test_extract_test_step_metrics_for_logging_keeps_scalar_train_and_val_metrics():
    metrics = {
        "actor/grad_norm": torch.tensor(1.25),
        "critic/advantages/min": np.float32(-0.5),
        "training/global_step": 5,
        "val-core/openai/gsm8k/acc/mean@1": 0.75,
        "misc/non_scalar_list": [1, 2, 3],
        "misc/non_scalar_tensor": torch.ones(2),
    }

    summary = extract_test_step_metrics_for_logging(metrics)

    assert list(summary.keys()) == [
        "training/global_step",
        "actor/grad_norm",
        "critic/advantages/min",
        "val-core/openai/gsm8k/acc/mean@1",
    ]
    assert summary["actor/grad_norm"] == 1.25
    assert summary["critic/advantages/min"] == -0.5
    assert "misc/non_scalar_list" not in summary
    assert "misc/non_scalar_tensor" not in summary
