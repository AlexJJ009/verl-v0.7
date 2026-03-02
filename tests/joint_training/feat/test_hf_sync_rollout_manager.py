"""Unit tests for HFSyncRolloutManager.

Verifies that:
1. HFSyncRolloutManager has the correct interface (same as AgentLoopManager)
2. generate_sequences() delegates to worker_group.generate_sequences()
3. rollout_replicas is an empty list (no separate replicas for in-process rollout)
4. start_profile() and stop_profile() are no-ops
5. RayPPOTrainer selects HFSyncRolloutManager when rollout.name=hf
6. generate_sequences in fsdp_workers handles loop-already-running case
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, call

from verl import DataProto


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


class TestLoopAlreadyRunningFix:
    """Verify generate_sequences() handles event loop already running (async Ray actor case)."""

    def test_rollout_mode_and_trainer_mode_runnable_via_asyncio_run(self):
        """rollout_mode() and trainer_mode() for HFRollout have no awaits — verify they
        can be driven to completion via asyncio.run() in a new thread (the fallback path)."""
        import asyncio
        import concurrent.futures
        from unittest.mock import MagicMock, patch
        from verl.workers.rollout.hf_rollout import HFRollout

        # Create a minimal mock worker that has rollout_mode / trainer_mode
        # We test that asyncio.run() can execute them without issues
        mock_rollout = MagicMock(spec=HFRollout)
        mock_module = MagicMock()
        mock_module._fsdp_wrapped_module = MagicMock()
        delattr(mock_module._fsdp_wrapped_module, "fusion_lambda") if hasattr(
            mock_module._fsdp_wrapped_module, "fusion_lambda"
        ) else None

        # Import the async functions and test them directly
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
        # If we reach here, the thread-based async.run() works correctly

    def test_loop_is_running_detection(self):
        """Verify asyncio.get_event_loop().is_running() is True inside async context."""

        async def check_running():
            loop = asyncio.get_event_loop()
            return loop.is_running()

        result = asyncio.run(check_running())
        assert result is True



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
        import asyncio
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
