# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.ray_trainer import assign_dynamic_permutation_sample_ids
from verl.utils.seqlen_balancing import prepare_dynamic_batch


def _expanded_batch():
    base = DataProto(
        batch=TensorDict({"row_value": torch.tensor([10, 20])}, batch_size=[2]),
        non_tensor_batch={},
    )
    return base.repeat(repeat_times=3, interleave=True)


def test_sample_ids_are_assigned_after_expansion_and_follow_reordering():
    batch = _expanded_batch()
    assign_dynamic_permutation_sample_ids(batch, global_step=7)
    expected = torch.arange(6, dtype=torch.int64) + (7 << 32)
    assert torch.equal(batch.batch["dynperm_sample_id"], expected)

    order = torch.tensor([5, 0, 3, 1, 4, 2])
    batch.reorder(order)
    assert torch.equal(batch.batch["dynperm_sample_id"], expected[order])
    assert torch.equal(batch.batch["row_value"], torch.tensor([20, 10, 20, 10, 20, 10]))


def test_sample_ids_replay_from_restored_step_and_change_across_steps():
    first = _expanded_batch()
    replay = _expanded_batch()
    next_step = _expanded_batch()
    assign_dynamic_permutation_sample_ids(first, global_step=9)
    assign_dynamic_permutation_sample_ids(replay, global_step=9)
    assign_dynamic_permutation_sample_ids(next_step, global_step=10)
    assert torch.equal(first.batch["dynperm_sample_id"], replay.batch["dynperm_sample_id"])
    assert not torch.equal(first.batch["dynperm_sample_id"], next_step.batch["dynperm_sample_id"])


def test_duplicate_assignment_fails_closed():
    batch = _expanded_batch()
    assign_dynamic_permutation_sample_ids(batch, global_step=1)
    with pytest.raises(ValueError, match="already exists"):
        assign_dynamic_permutation_sample_ids(batch, global_step=1)


def test_dynamic_microbatch_reordering_preserves_tensor_identity_alignment():
    lengths = torch.tensor([6, 2, 5, 3])
    attention_mask = torch.arange(6).unsqueeze(0) < lengths.unsqueeze(1)
    data = DataProto(
        batch=TensorDict(
            {
                "input_ids": torch.arange(4).unsqueeze(1).expand(-1, 6),
                "attention_mask": attention_mask,
                "row_value": torch.tensor([10, 20, 30, 40]),
            },
            batch_size=[4],
        ),
        non_tensor_batch={},
    )
    assign_dynamic_permutation_sample_ids(data, global_step=3)
    expected_by_row = {
        int(row): int(sample_id)
        for row, sample_id in zip(data.batch["row_value"], data.batch["dynperm_sample_id"], strict=True)
    }
    micro_batches, _ = prepare_dynamic_batch(data, max_token_len=8, same_micro_num_in_dp=False)
    for micro_batch in micro_batches:
        for row, sample_id in zip(micro_batch.batch["row_value"], micro_batch.batch["dynperm_sample_id"], strict=True):
            assert int(sample_id) == expected_by_row[int(row)]
