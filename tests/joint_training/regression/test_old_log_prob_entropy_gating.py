# SPDX-License-Identifier: Apache-2.0

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils import tensordict_utils as tu


class _FakeBatch:
    def to_tensordict(self):
        return TensorDict({"input_ids": torch.ones(1, 1, dtype=torch.long)}, batch_size=[1])


def test_compute_old_log_prob_skips_entropy_when_disabled(monkeypatch):
    import verl.trainer.ppo.ray_trainer as ray_trainer_module

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.use_legacy_worker_impl = "disable"
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "calculate_entropy": False,
                    "entropy_coeff": 0.0,
                }
            }
        }
    )

    seen = {}

    def fake_compute_log_prob(batch_td):
        seen["calculate_entropy"] = tu.get(batch_td, "calculate_entropy")
        return tu.get_tensordict(
            {"log_probs": torch.zeros(1, 1, dtype=torch.float32)},
            non_tensor_dict={"metrics": {"mfu": 0.0}},
        )

    trainer.actor_rollout_wg = type("_WG", (), {"compute_log_prob": staticmethod(fake_compute_log_prob)})()

    monkeypatch.setattr(ray_trainer_module, "left_right_2_no_padding", lambda td: td)
    monkeypatch.setattr(ray_trainer_module, "no_padding_2_padding", lambda tensor, td: tensor)

    old_log_prob, old_log_prob_mfu = RayPPOTrainer._compute_old_log_prob(trainer, _FakeBatch())

    assert seen["calculate_entropy"] is False
    assert "entropys" not in old_log_prob.batch
    assert torch.equal(old_log_prob.batch["old_log_probs"], torch.zeros(1, 1, dtype=torch.float32))
    assert old_log_prob_mfu == 0.0
