import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.utils.device import get_device_name
from verl.workers.critic import dp_critic as dp_critic_module
from verl.workers.critic.dp_critic import DataParallelPPOCritic


class MockCriticModel(nn.Module):
    def __init__(self, vocab_size=128, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False, **kwargs):
        hidden_states = self.embedding(input_ids)
        logits = self.value_head(hidden_states)

        class MockOutput:
            def __init__(self, logits):
                self.logits = logits

        return MockOutput(logits)


class TestDynamicBatchSyncGroupForCritic(unittest.TestCase):
    def setUp(self):
        if get_device_name() == "cuda":
            self.device = torch.device("cuda:0")
        elif get_device_name() == "npu":
            self.device = torch.device("npu:0")
        else:
            self.device = torch.device("cpu")

        self.config = OmegaConf.create(
            {
                "model": {"use_remove_padding": False},
                "ulysses_sequence_parallel_size": 1,
                "ppo_mini_batch_size": 4,
                "ppo_micro_batch_size_per_gpu": 2,
                "ppo_epochs": 1,
                "ppo_max_token_len_per_gpu": 64,
                "cliprange_value": 0.2,
                "loss_agg_mode": "token-mean",
                "grad_clip": 1.0,
                "use_dynamic_bsz": True,
            }
        )

        self.model = MockCriticModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.dp_group = object()
        self.critic = DataParallelPPOCritic(
            config=self.config,
            critic_module=self.model,
            critic_optimizer=self.optimizer,
            dp_group=self.dp_group,
        )

    def _build_compute_values_data(self):
        batch_size = 4
        seq_len = 8
        response_len = 4

        batch = TensorDict(
            {
                "input_ids": torch.randint(0, 128, (batch_size, seq_len), device=self.device),
                "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=self.device),
                "position_ids": torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1),
                "responses": torch.randint(0, 128, (batch_size, response_len), device=self.device),
                "response_mask": torch.ones(batch_size, response_len, dtype=torch.float32, device=self.device),
            },
            batch_size=[batch_size],
        )

        return DataProto(
            batch=batch,
            meta_info={"micro_batch_size": 2, "max_token_len": seq_len, "use_dynamic_bsz": True},
        )

    def _build_update_critic_data(self):
        batch_size = 4
        seq_len = 8
        response_len = 4

        batch = TensorDict(
            {
                "input_ids": torch.randint(0, 128, (batch_size, seq_len), device=self.device),
                "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=self.device),
                "position_ids": torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1),
                "responses": torch.randint(0, 128, (batch_size, response_len), device=self.device),
                "response_mask": torch.ones(batch_size, response_len, dtype=torch.float32, device=self.device),
                "values": torch.zeros(batch_size, response_len, dtype=torch.float32, device=self.device),
                "returns": torch.ones(batch_size, response_len, dtype=torch.float32, device=self.device),
            },
            batch_size=[batch_size],
        )

        return DataProto(
            batch=batch,
            meta_info={"global_token_num": [seq_len] * batch_size},
        )

    def test_compute_values_uses_dp_group_for_dynamic_batch(self):
        data = self._build_compute_values_data()

        with patch(
            "verl.workers.critic.dp_critic.prepare_dynamic_batch", wraps=dp_critic_module.prepare_dynamic_batch
        ) as mocked:
            values = self.critic.compute_values(data)

        self.assertEqual(values.shape, (4, 4))
        self.assertIs(mocked.call_args.kwargs["dp_group"], self.dp_group)
        self.assertTrue(mocked.call_args.kwargs["same_micro_num_in_dp"])

    def test_update_critic_uses_dp_group_for_dynamic_batch(self):
        data = self._build_update_critic_data()

        with patch(
            "verl.workers.critic.dp_critic.prepare_dynamic_batch", wraps=dp_critic_module.prepare_dynamic_batch
        ) as mocked:
            metrics = self.critic.update_critic(data)

        self.assertIn("critic/vf_loss", metrics)
        self.assertIs(mocked.call_args.kwargs["dp_group"], self.dp_group)
        self.assertTrue(mocked.call_args.kwargs["same_micro_num_in_dp"])


if __name__ == "__main__":
    unittest.main()
