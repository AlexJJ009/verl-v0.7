# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import asyncio

import pytest
from omegaconf import OmegaConf

from verl.experimental.reward_loop.reward_loop import RewardLoopManager, RewardLoopWorker, split_near_equal


class _Batch:
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        return _Batch(self.values[item])


def test_split_near_equal_supports_non_divisible_validation_batches():
    chunks = split_near_equal(_Batch(range(1379)), 8)
    assert [len(chunk) for chunk in chunks] == [173, 173, 173, 172, 172, 172, 172, 172]
    assert [value for chunk in chunks for value in chunk.values] == list(range(1379))


@pytest.mark.asyncio
async def test_compute_score_batch_bounds_concurrency_and_preserves_order():
    worker = RewardLoopWorker.__new__(RewardLoopWorker)
    worker.max_concurrency = 3
    in_flight = 0
    peak_in_flight = 0

    async def compute_score(item):
        nonlocal in_flight, peak_in_flight
        value = item.values[0]
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        try:
            await asyncio.sleep((10 - value) * 0.001)
            return {"reward_score": value}
        finally:
            in_flight -= 1

    worker.compute_score = compute_score
    outputs = await worker.compute_score_batch(_Batch(range(10)))

    assert peak_in_flight == 3
    assert [output["reward_score"] for output in outputs] == list(range(10))


@pytest.mark.asyncio
async def test_compute_score_batch_propagates_exceptions():
    worker = RewardLoopWorker.__new__(RewardLoopWorker)
    worker.max_concurrency = 2
    cancelled = asyncio.Event()

    async def compute_score(item):
        if item.values[0] == 1:
            raise RuntimeError("reward failed")
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return {"reward_score": item.values[0]}

    worker.compute_score = compute_score

    with pytest.raises(RuntimeError, match="reward failed"):
        await worker.compute_score_batch(_Batch(range(4)))
    assert cancelled.is_set()


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "4", None])
def test_max_concurrency_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="must be a positive integer"):
        RewardLoopWorker._validate_max_concurrency(value)


@pytest.mark.parametrize("value", [1, 16, 128])
def test_max_concurrency_accepts_positive_integers(value):
    assert RewardLoopWorker._validate_max_concurrency(value) == value


def test_reward_loop_ray_actor_uses_configured_max_concurrency(monkeypatch):
    configured_options = []

    class _RemoteClass:
        def options(self, **kwargs):
            configured_options.append(kwargs)
            return self

        def remote(self, *args):
            return object()

    monkeypatch.setattr(
        "verl.experimental.reward_loop.reward_loop.ray.nodes",
        lambda: [{"NodeID": "01" * 28, "Alive": True, "Resources": {"CPU": 176}}],
    )
    manager = RewardLoopManager.__new__(RewardLoopManager)
    manager.config = OmegaConf.create(
        {"reward": {"num_workers": 2, "max_concurrency_per_worker": 4}}
    )
    manager.reward_router_address = None
    manager.reward_loop_workers_class = _RemoteClass()

    manager._init_reward_loop_workers()

    assert len(manager.reward_loop_workers) == 2
    assert [options["max_concurrency"] for options in configured_options] == [4, 4]
