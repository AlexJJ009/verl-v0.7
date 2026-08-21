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

import math

import pytest
import torch

from verl import DataProto
from verl.models.joint_model.dynamic_permutation import (
    target_preserving_dynamic_permutation,
    validate_dynamic_permutation_rho,
)
from verl.trainer.ppo.ray_trainer import assign_dynamic_permutation_sample_ids
from verl.workers.actor.dp_actor import (
    build_dynamic_permutation_training_context,
    dynamic_permutation_actor_update_index,
)
from verl.workers.config import WeakLogitPermutationConfig


def _identity(shape):
    rows = math.prod(shape)
    return (
        torch.arange(rows, dtype=torch.int64).reshape(shape),
        torch.arange(rows, dtype=torch.int64).reshape(shape) + 11,
        torch.ones(shape, dtype=torch.bool),
    )


@pytest.mark.parametrize("rho", [float("nan"), float("inf"), -0.1, 1.1, None, True])
def test_rho_validation_fails_closed(rho):
    with pytest.raises(ValueError, match="rho"):
        validate_dynamic_permutation_rho(rho)


def test_rho_zero_is_exact_object_and_rng_no_op():
    logits = torch.randn(2, 3, 11, requires_grad=True)
    targets = torch.randint(0, 11, (2, 3))
    sample_ids, positions, active = _identity((2, 3))
    rng_before = torch.random.get_rng_state().clone()
    output, telemetry = target_preserving_dynamic_permutation(
        logits,
        targets,
        sample_ids,
        positions,
        active,
        rho=0,
        base_seed=42,
        global_step=7,
        actor_update_index=3,
    )
    assert output is logits
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert telemetry.selected_coordinates == 0


@pytest.mark.parametrize("rho", [0.25, 0.5, 1.0])
def test_target_multiset_entropy_coverage_determinism_and_autograd(rho):
    vocab = 17
    logits = torch.randn(2, 4, vocab, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[0, 1, 8, 16], [3, 6, 9, 12]])
    sample_ids, positions, active = _identity((2, 4))
    output, telemetry = target_preserving_dynamic_permutation(
        logits,
        targets,
        sample_ids,
        positions,
        active,
        rho=rho,
        base_seed=42,
        global_step=7,
        actor_update_index=3,
        row_chunk_size=2,
        audit_rows=8,
        entropy_atol=1e-12,
    )
    replay, _ = target_preserving_dynamic_permutation(
        logits,
        targets,
        sample_ids,
        positions,
        active,
        rho=rho,
        base_seed=42,
        global_step=7,
        actor_update_index=3,
        row_chunk_size=3,
    )
    assert torch.equal(output, replay)
    assert torch.equal(output.gather(-1, targets.unsqueeze(-1)), logits.gather(-1, targets.unsqueeze(-1)))
    assert torch.equal(torch.sort(output, dim=-1).values, torch.sort(logits, dim=-1).values)
    assert torch.allclose(
        torch.distributions.Categorical(logits=output).entropy(),
        torch.distributions.Categorical(logits=logits).entropy(),
        atol=1e-12,
        rtol=0,
    )
    expected = math.floor(rho * (vocab - 1))
    assert telemetry.selected_per_row == expected
    assert telemetry.selected_coordinates == logits.shape[0] * logits.shape[1] * expected
    assert telemetry.fixed_points == 0
    output.square().sum().backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == logits.numel()


def test_step_resamples_and_inactive_rows_are_untouched():
    logits = torch.arange(3 * 13, dtype=torch.float64).reshape(3, 13)
    targets = torch.tensor([0, 6, 12])
    sample_ids, positions, active = _identity((3,))
    active[1] = False
    first, _ = target_preserving_dynamic_permutation(
        logits,
        targets,
        sample_ids,
        positions,
        active,
        rho=1,
        base_seed=9,
        global_step=4,
        actor_update_index=0,
    )
    second, _ = target_preserving_dynamic_permutation(
        logits,
        targets,
        sample_ids,
        positions,
        active,
        rho=1,
        base_seed=9,
        global_step=5,
        actor_update_index=0,
    )
    assert torch.equal(first[1], logits[1])
    assert not torch.equal(first[0], second[0])


def test_row_reordering_and_chunking_do_not_change_mapping():
    logits = torch.randn(5, 19)
    targets = torch.tensor([1, 3, 5, 7, 9])
    sample_ids = torch.tensor([101, 102, 103, 104, 105])
    positions = torch.tensor([11, 12, 13, 14, 15])
    active = torch.ones(5, dtype=torch.bool)
    expected, _ = target_preserving_dynamic_permutation(
        logits,
        targets,
        sample_ids,
        positions,
        active,
        rho=0.5,
        base_seed=7,
        global_step=8,
        actor_update_index=2,
        row_chunk_size=1,
    )
    order = torch.tensor([3, 0, 4, 1, 2])
    reordered, _ = target_preserving_dynamic_permutation(
        logits[order],
        targets[order],
        sample_ids[order],
        positions[order],
        active[order],
        rho=0.5,
        base_seed=7,
        global_step=8,
        actor_update_index=2,
        row_chunk_size=4,
    )
    inverse = torch.argsort(order)
    assert torch.equal(expected, reordered[inverse])


def test_single_selected_coordinate_fails_closed():
    logits = torch.randn(1, 5)
    with pytest.raises(ValueError, match="exactly one"):
        target_preserving_dynamic_permutation(
            logits,
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([2]),
            torch.tensor([True]),
            rho=0.25,
            base_seed=0,
            global_step=0,
            actor_update_index=0,
        )


def test_shape_and_target_validation():
    logits = torch.randn(2, 7)
    with pytest.raises(ValueError, match="shape"):
        target_preserving_dynamic_permutation(
            logits,
            torch.tensor([1]),
            torch.arange(2),
            torch.arange(2),
            torch.ones(2, dtype=torch.bool),
            rho=1,
            base_seed=0,
            global_step=0,
            actor_update_index=0,
        )
    with pytest.raises(ValueError, match="outside"):
        target_preserving_dynamic_permutation(
            logits,
            torch.tensor([1, 7]),
            torch.arange(2),
            torch.arange(2),
            torch.ones(2, dtype=torch.bool),
            rho=1,
            base_seed=0,
            global_step=0,
            actor_update_index=0,
        )


def test_actor_training_context_aligns_response_targets_and_masks():
    input_ids = torch.tensor([[10, 11, 12, 3, 4, 5], [20, 21, 22, 6, 7, 8]])
    responses = torch.tensor([[3, 4, 5], [6, 7, 8]])
    response_mask = torch.tensor([[1, 1, 1], [1, 0, 1]])
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 1, 2, 3, 4]])
    context = build_dynamic_permutation_training_context(
        input_ids=input_ids,
        position_ids=position_ids,
        responses=responses,
        response_mask=response_mask,
        sample_ids=torch.tensor([100, 200]),
    )
    assert torch.equal(context["target_ids"][:, 2:5], responses)
    assert torch.equal(context["active_mask"][:, 2:5], response_mask.bool())
    assert not context["active_mask"][:, :2].any()
    assert not context["active_mask"][:, 5:].any()
    assert torch.equal(context["sample_ids"][:, 0], torch.tensor([100, 200]))
    assert torch.equal(context["token_positions"], position_ids)


def test_actor_training_context_rejects_misaligned_sample_ids():
    with pytest.raises(ValueError, match="dynperm_sample_id"):
        build_dynamic_permutation_training_context(
            input_ids=torch.ones(2, 4, dtype=torch.long),
            position_ids=torch.arange(4).expand(2, -1),
            responses=torch.ones(2, 2, dtype=torch.long),
            response_mask=torch.ones(2, 2, dtype=torch.long),
            sample_ids=torch.ones(2, 1, dtype=torch.long),
        )


def test_actor_update_index_is_shared_by_microbatches_and_advances_after_update():
    assert dynamic_permutation_actor_update_index(0, 0, 3) == 0
    assert dynamic_permutation_actor_update_index(0, 1, 3) == 1
    assert dynamic_permutation_actor_update_index(1, 0, 3) == 3
    with pytest.raises(ValueError, match="invalid"):
        dynamic_permutation_actor_update_index(0, 3, 3)


def test_trainer_assigns_global_stable_dynperm_sample_ids_after_rollout_expansion():
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.ones(4, 3, dtype=torch.long),
            "responses": torch.ones(4, 2, dtype=torch.long),
        }
    )
    assign_dynamic_permutation_sample_ids(batch, global_step=17)
    assert torch.equal(
        batch.batch["dynperm_sample_id"],
        torch.tensor([(17 << 32) + row for row in range(4)], dtype=torch.int64),
    )
    with pytest.raises(ValueError, match="already exists"):
        assign_dynamic_permutation_sample_ids(batch, global_step=18)


@pytest.mark.parametrize("rho", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_actor_config_rejects_invalid_rho(rho):
    with pytest.raises(ValueError, match="rho"):
        WeakLogitPermutationConfig(rho=rho)


def test_actor_config_accepts_numeric_partial_and_endpoint_doses():
    for rho in (0, 0.25, 0.5, 1):
        assert WeakLogitPermutationConfig(enabled=True, rho=rho).rho == float(rho)
