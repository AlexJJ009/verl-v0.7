import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto


class _DummyTokenizer:
    eos_token_id = 7
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return "decoded"


class _FakeRolloutManager:
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = len(prompts)
        if prompts.batch is not None and "input_ids" in prompts.batch.keys():
            prompts_out = prompts.batch["input_ids"].clone()
        else:
            prompts_out = torch.tensor([[11, 12, 13]], dtype=torch.long).repeat(batch_size, 1)
        response_length = 3
        responses = torch.full((batch_size, response_length), 5, dtype=torch.long)
        seq = torch.cat([prompts_out, responses], dim=1)
        attention_mask = torch.ones_like(seq)
        position_ids = torch.arange(seq.shape[1], dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
        batch = TensorDict(
            {
                "prompts": prompts_out,
                "responses": responses,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch)


class _FakeRewardLoopManager:
    def __init__(self):
        self.seen_batches = []

    def compute_rm_score(self, batch: DataProto) -> DataProto:
        self.seen_batches.append(batch)
        assert len(batch) == 1
        assert "data_source" in batch.non_tensor_batch
        assert batch.non_tensor_batch["data_source"].tolist() == ["gsm8k"]
        assert "reward_model" in batch.non_tensor_batch

        rm_scores = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        rm_scores[:, -1] = 1.0
        return DataProto(
            batch=TensorDict({"rm_scores": rm_scores}, batch_size=len(batch)),
            non_tensor_batch={"acc": np.array([1.0])},
            meta_info={"reward_extra_keys": ["acc"]},
        )


def test_validate_preserves_reward_metadata_for_colocated_rm():
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer._is_joint_training = False
    trainer.async_rollout_mode = False
    trainer.use_rm = False
    trainer.tokenizer = _DummyTokenizer()
    trainer.async_rollout_manager = _FakeRolloutManager()
    trainer.reward_loop_manager = _FakeRewardLoopManager()
    trainer.checkpoint_manager = type(
        "_CheckpointManager",
        (),
        {"sleep_replicas": lambda self: None, "update_weights": lambda self, eval_only=False: None},
    )()
    trainer.global_steps = 0
    trainer._maybe_log_val_generations = lambda **kwargs: None
    trainer._val_metrics_update = lambda data_sources, sample_uids, reward_extra_infos_dict, sample_turns: {"ok": 1.0}
    trainer.config = OmegaConf.create(
        {
            "data": {"max_response_length": 3},
            "actor_rollout_ref": {
                "rollout": {
                    "name": "hf",
                    "val_kwargs": {"n": 1, "do_sample": True},
                    "agent": {"num_workers": 2},
                }
            },
            "trainer": {"validation_data_dir": None},
        }
    )

    prompts = torch.tensor([[11, 12, 13]], dtype=torch.long)
    test_data = {
        "prompts": prompts.clone(),
        "input_ids": prompts.clone(),
        "attention_mask": torch.ones_like(prompts),
        "position_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "data_source": np.array(["gsm8k"], dtype=object),
        "reward_model": np.array([{"ground_truth": "4"}], dtype=object),
        "extra_info": np.array([{}], dtype=object),
    }
    trainer.val_dataloader = [test_data]

    result = trainer._validate()

    assert result == {"ok": 1.0}
    assert len(trainer.reward_loop_manager.seen_batches) == 1


def test_validate_handles_hf_raw_prompt_batches_without_tensor_gen_batch():
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer._is_joint_training = False
    trainer.async_rollout_mode = False
    trainer.use_rm = False
    trainer.tokenizer = _DummyTokenizer()
    trainer.async_rollout_manager = _FakeRolloutManager()
    trainer.reward_loop_manager = _FakeRewardLoopManager()
    trainer.checkpoint_manager = type(
        "_CheckpointManager",
        (),
        {"sleep_replicas": lambda self: None, "update_weights": lambda self, eval_only=False: None},
    )()
    trainer.global_steps = 0
    trainer._maybe_log_val_generations = lambda **kwargs: None
    trainer._val_metrics_update = lambda data_sources, sample_uids, reward_extra_infos_dict, sample_turns: {"ok": 1.0}
    trainer.config = OmegaConf.create(
        {
            "data": {"max_response_length": 3},
            "actor_rollout_ref": {
                "rollout": {
                    "name": "hf",
                    "val_kwargs": {"n": 1, "do_sample": True},
                    "agent": {"num_workers": 2},
                }
            },
            "trainer": {"validation_data_dir": None},
        }
    )

    test_data = {
        "dummy_tensor": torch.zeros(1, 1, dtype=torch.uint8),
        "raw_prompt": np.array([[{"role": "user", "content": "hello"}]], dtype=object),
        "data_source": np.array(["gsm8k"], dtype=object),
        "reward_model": np.array([{"ground_truth": "4"}], dtype=object),
        "extra_info": np.array([{}], dtype=object),
    }
    trainer.val_dataloader = [test_data]

    result = trainer._validate()

    assert result == {"ok": 1.0}
    assert len(trainer.reward_loop_manager.seen_batches) == 1
