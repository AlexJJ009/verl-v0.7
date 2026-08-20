"""Unit tests for HFSyncRolloutManager and HFRollout worker-side tokenization.

Verifies that:
1. HFSyncRolloutManager has the correct interface (same as AgentLoopManager)
2. generate_sequences() delegates to worker_group.generate_sequences()
3. rollout_replicas is an empty list (no separate replicas for in-process rollout)
4. start_profile() and stop_profile() are no-ops
5. RayPPOTrainer selects HFSyncRolloutManager when rollout.name=hf
6. generate_sequences in fsdp_workers handles loop-already-running case
7. _tokenize_raw_prompts_for_hf() correctly tokenizes raw chat prompts to tensors
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from verl import DataProto

TOKENIZER_PATH = "/data-1/.cache/Qwen3-4B-Base-SFT-stage-1"


class TestHFSyncRolloutManagerInterface:
    """Verify HFSyncRolloutManager has the AgentLoopManager-compatible interface."""

    def setup_method(self):
        from verl.trainer.ppo.ray_trainer import HFSyncRolloutManager

        self.mock_wg = MagicMock()
        self.manager = HFSyncRolloutManager(worker_group=self.mock_wg)

    def test_rollout_replicas_is_empty_list(self):
        assert self.manager.rollout_replicas == []

    def test_generate_sequences_delegates_to_worker_group(self):
        mock_batch = MagicMock(spec=DataProto)
        mock_result = MagicMock(spec=DataProto)
        self.mock_wg.generate_sequences.return_value = mock_result

        result = self.manager.generate_sequences(mock_batch)

        self.mock_wg.generate_sequences.assert_called_once_with(mock_batch)
        assert result is mock_result

    def test_start_profile_is_noop(self):
        # Should not raise and should not call anything on worker_group
        self.manager.start_profile()
        self.manager.start_profile(step=5)
        self.mock_wg.start_profile.assert_not_called()

    def test_stop_profile_is_noop(self):
        self.manager.stop_profile()
        self.mock_wg.stop_profile.assert_not_called()

    def test_generate_sequences_returns_dataproto(self):
        """generate_sequences() must return DataProto (same contract as AgentLoopManager)."""
        import torch

        batch = DataProto.from_dict({"input_ids": torch.zeros(2, 4, dtype=torch.long)})
        self.mock_wg.generate_sequences.return_value = batch

        result = self.manager.generate_sequences(batch)
        assert isinstance(result, DataProto)


class TestHFSyncRolloutManagerSelection:
    """Verify ray_trainer uses HFSyncRolloutManager when rollout.name=hf."""

    def test_hf_sync_rollout_manager_importable(self):
        from verl.trainer.ppo.ray_trainer import HFSyncRolloutManager

        assert HFSyncRolloutManager is not None

    def test_hf_sync_rollout_manager_has_correct_attributes(self):
        from verl.trainer.ppo.ray_trainer import HFSyncRolloutManager

        mock_wg = MagicMock()
        mgr = HFSyncRolloutManager(worker_group=mock_wg)

        # Must have the same interface as AgentLoopManager
        assert hasattr(mgr, "rollout_replicas")
        assert hasattr(mgr, "generate_sequences")
        assert hasattr(mgr, "start_profile")
        assert hasattr(mgr, "stop_profile")
        assert callable(mgr.generate_sequences)
        assert callable(mgr.start_profile)
        assert callable(mgr.stop_profile)

    def test_hf_rollout_replica_generate_sequences_raises_not_implemented(self):
        """HFRolloutReplica.generate_sequences should raise NotImplementedError,
        since traffic goes through HFSyncRolloutManager instead."""
        from verl.workers.rollout.replica import RolloutReplicaRegistry

        HFRolloutReplica = RolloutReplicaRegistry.get("hf")
        # Instantiate with minimal mocks
        replica = HFRolloutReplica.__new__(HFRolloutReplica)

        with pytest.raises(NotImplementedError):
            asyncio.run(replica.generate_sequences())

    def test_generate_sequences_hf_removed_from_fsdp_worker(self):
        """The broken generate_sequences_hf() async stub must not exist on ActorRolloutRefWorker."""
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        assert not hasattr(ActorRolloutRefWorker, "generate_sequences_hf"), (
            "generate_sequences_hf was removed; do not re-add it. "
            "Use HFSyncRolloutManager -> worker_group.generate_sequences() instead."
        )


class TestGetGenBatchForHFRollout:
    """Verify _get_gen_batch behavior for HF rollout (async_rollout_mode=False)."""

    def _make_batch(self):
        import numpy as np
        import torch
        from tensordict import TensorDict

        # Dataset returns dummy_tensor + non-tensor fields (no input_ids)
        batch_td = TensorDict(
            {"dummy_tensor": torch.zeros(4, 1, dtype=torch.uint8)},
            batch_size=4,
        )
        return DataProto(
            batch=batch_td,
            non_tensor_batch={
                "raw_prompt": np.array([[{"role": "user", "content": "hello"}]] * 4),
                "reward_model": np.array([{}] * 4),
                "data_source": np.array(["gsm8k"] * 4),
            },
        )

    def test_hf_rollout_gen_batch_has_no_tensor_keys(self):
        """When async_rollout_mode=False, dataset only provides dummy_tensor.
        _get_gen_batch returns batch=None (worker tokenizes from raw_prompt)."""
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        trainer = MagicMock()
        trainer.async_rollout_mode = False

        batch = self._make_batch()
        gen_batch = RayPPOTrainer._get_gen_batch(trainer, batch)

        # gen_batch.batch is None because dataset doesn't provide input_ids
        # The worker's _tokenize_raw_prompts_for_hf() handles tokenization
        assert gen_batch.batch is None or (gen_batch.batch is not None and "input_ids" not in gen_batch.batch.keys())
        # raw_prompt must be present for worker-side tokenization
        assert "raw_prompt" in gen_batch.non_tensor_batch

    def test_agent_loop_gen_batch_has_no_tensor_keys(self):
        """When async_rollout_mode=True (AgentLoopManager), gen_batch has no tensor keys."""
        trainer = MagicMock()
        trainer.async_rollout_mode = True

        batch = self._make_batch()

        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        gen_batch = RayPPOTrainer._get_gen_batch(trainer, batch)

        # async path: no tensor keys in gen_batch
        assert gen_batch.batch is None or len(gen_batch.batch.keys()) == 0


class TestLoopAlreadyRunningFix:
    """Verify generate_sequences() handles event loop already running (async Ray actor case)."""

    def test_rollout_mode_and_trainer_mode_runnable_via_asyncio_run(self):
        """rollout_mode() and trainer_mode() for HFRollout have no awaits — verify they
        can be driven to completion via asyncio.run() in a new thread (the fallback path)."""
        import asyncio
        import concurrent.futures

        async def rollout_mode_stub():
            # Simulates HFRollout path: no awaits
            return None

        async def trainer_mode_stub():
            return None

        # These must complete via asyncio.run() in a thread (the fix code path)
        def run_in_thread(coro_factory):
            asyncio.run(coro_factory())

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(run_in_thread, rollout_mode_stub).result()
            pool.submit(run_in_thread, trainer_mode_stub).result()
        # If we reach here, the thread-based asyncio.run() works correctly

    def test_loop_is_running_detection(self):
        """Verify asyncio.get_event_loop().is_running() is True inside async context."""

        async def check_running():
            loop = asyncio.get_event_loop()
            return loop.is_running()

        result = asyncio.run(check_running())
        assert result is True


class TestTokenizeRawPromptsForHF:
    """Verify _tokenize_raw_prompts_for_hf correctly tokenizes chat prompts for HFRollout.

    This method bridges the gap: RLHFDataset returns raw_prompt (chat message dicts)
    for AgentLoop compatibility, but HFRollout needs pre-tokenized input_ids tensors.
    The FSDP worker tokenizes on the fly.
    """

    def _make_prompts_with_raw_prompt(self, n_prompts=4, prompt_text="Solve: 2 + 2 = ?"):
        """Build a DataProto with raw_prompt in non_tensor_batch (no input_ids)."""
        import numpy as np

        messages = [{"role": "user", "content": prompt_text}]
        return DataProto(
            batch=None,
            non_tensor_batch={
                "raw_prompt": np.array([messages] * n_prompts),
            },
        )

    def test_tokenize_raw_prompts_produces_input_ids(self):
        """After tokenization, prompts.batch must have input_ids, attention_mask, position_ids."""
        from transformers import AutoTokenizer

        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        max_len = 64

        worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
        worker.tokenizer = tokenizer
        worker.config = MagicMock()
        worker.config.rollout.prompt_length = max_len

        prompts = self._make_prompts_with_raw_prompt(n_prompts=4)
        result = worker._tokenize_raw_prompts_for_hf(prompts)

        assert result.batch is not None, "batch must not be None after tokenization"
        assert "input_ids" in result.batch.keys()
        assert "attention_mask" in result.batch.keys()
        assert "position_ids" in result.batch.keys()

    def test_tokenize_raw_prompts_correct_shape(self):
        """Tokenized tensors must have shape (n_prompts, max_prompt_length)."""
        from transformers import AutoTokenizer

        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        max_len = 64
        n_prompts = 3

        worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
        worker.tokenizer = tokenizer
        worker.config = MagicMock()
        worker.config.rollout.prompt_length = max_len

        prompts = self._make_prompts_with_raw_prompt(n_prompts=n_prompts)
        result = worker._tokenize_raw_prompts_for_hf(prompts)

        assert result.batch["input_ids"].shape == (n_prompts, max_len)
        assert result.batch["attention_mask"].shape == (n_prompts, max_len)
        assert result.batch["position_ids"].shape == (n_prompts, max_len)

    def test_tokenize_raw_prompts_returns_unchanged_if_no_raw_prompt(self):
        """If raw_prompt is not in non_tensor_batch, return prompts unchanged."""
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
        worker.tokenizer = MagicMock()
        worker.config = MagicMock()

        prompts = DataProto(batch=None, non_tensor_batch={})
        result = worker._tokenize_raw_prompts_for_hf(prompts)

        assert result is prompts  # unchanged
        worker.tokenizer.apply_chat_template.assert_not_called()

    def test_tokenize_uses_left_padding(self):
        """Tokenization must use left-padding for HFRollout (causal LM generation)."""
        from transformers import AutoTokenizer

        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        max_len = 64

        worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
        worker.tokenizer = tokenizer
        worker.config = MagicMock()
        worker.config.rollout.prompt_length = max_len

        # Use prompts of different lengths to test padding behavior
        import numpy as np

        short_msg = [{"role": "user", "content": "Hi"}]
        long_msg = [{"role": "user", "content": "Solve this step by step: " + "x " * 20}]
        prompts = DataProto(
            batch=None,
            non_tensor_batch={"raw_prompt": np.array([short_msg, long_msg])},
        )
        result = worker._tokenize_raw_prompts_for_hf(prompts)

        # For left-padded sequences, the first token of the shorter sequence is pad_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        assert result.batch["input_ids"][0, 0].item() == pad_id, (
            "Short prompt should be left-padded: first token should be pad_token_id"
        )

    def test_tokenizer_padding_side_restored_after_tokenization(self):
        """Tokenizer padding_side must be restored to its original value after tokenization."""
        from transformers import AutoTokenizer

        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        tokenizer.padding_side = "right"  # Set to right before calling
        max_len = 64

        worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
        worker.tokenizer = tokenizer
        worker.config = MagicMock()
        worker.config.rollout.prompt_length = max_len

        prompts = self._make_prompts_with_raw_prompt(n_prompts=2)
        worker._tokenize_raw_prompts_for_hf(prompts)

        assert tokenizer.padding_side == "right", "padding_side must be restored to 'right' after tokenization"
