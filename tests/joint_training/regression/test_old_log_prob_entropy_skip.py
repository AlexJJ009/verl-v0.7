# SPDX-License-Identifier: Apache-2.0

import torch
from omegaconf import OmegaConf

from verl import DataProto


class _FakeActorRolloutWG:
    def __init__(self):
        self.seen_flags = []

    def compute_log_prob(self, batch: DataProto) -> DataProto:
        self.seen_flags.append(batch.meta_info.get("calculate_entropy"))
        return DataProto.from_dict(tensors={"old_log_probs": torch.zeros(1, 2)})


def test_compute_old_log_prob_skips_entropy_when_disabled_and_restores_meta_flag():
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.use_legacy_worker_impl = "legacy"
    trainer.actor_rollout_wg = _FakeActorRolloutWG()
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

    batch = DataProto.from_dict(tensors={"input_ids": torch.ones(1, 2, dtype=torch.long)})
    batch.meta_info["calculate_entropy"] = True

    old_log_prob, old_log_prob_mfu = trainer._compute_old_log_prob(batch)

    assert trainer.actor_rollout_wg.seen_flags == [False]
    assert old_log_prob_mfu == 0
    assert "entropys" not in old_log_prob.batch
    assert batch.meta_info["calculate_entropy"] is True
