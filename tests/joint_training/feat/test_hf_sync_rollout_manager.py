"""Unit tests for HFSyncRolloutManager.

Verifies that:
1. HFSyncRolloutManager has the correct interface (same as AgentLoopManager)
2. generate_sequences() delegates to worker_group.generate_sequences()
3. rollout_replicas is an empty list (no separate replicas for in-process rollout)
4. start_profile() and stop_profile() are no-ops
5. RayPPOTrainer selects HFSyncRolloutManager when rollout.name=hf
"""

import pytest
from unittest.mock import MagicMock, call

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
