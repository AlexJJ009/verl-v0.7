# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
import verl.workers.fsdp_workers as fsdp_workers_module
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def test_actor_rollout_ref_worker_actor_ref_model():
    """Test specifying different reference/actor model"""
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8888"

    actor_model_path = os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B-Instruct")
    ref_model_path = os.path.expanduser("~/models/Qwen/Qwen2.5-1.5B-Instruct")
    if not os.path.isdir(actor_model_path) or not os.path.isdir(ref_model_path):
        pytest.skip("requires local Qwen2.5 worker test models under ~/models/Qwen")
    config_str = f"""
    model:
      path: {actor_model_path}
    actor:
      _target_: verl.workers.config.FSDPActorConfig
      strategy: fsdp
      fsdp_config:
        _target_: verl.workers.config.FSDPEngineConfig
        fsdp_size: -1
        forward_prefetch: false
      profiler:
        tool: torch_memory
        save_path: ./mem_snapshots
        tool_config:
          torch_memory:
            _target_: verl.utils.profiler.config.TorchMemoryToolConfig
            trace_alloc_max_entries: 100000
            stack_depth: 32
    ref:
      model:
        path: {ref_model_path}
      fsdp_config:
        _target_: verl.workers.config.FSDPEngineConfig
        fsdp_size: -1
      profiler:
        tool: torch_memory
        save_path: ./mem_snapshots
        tool_config:
          torch_memory:
            _target_: verl.utils.profiler.config.TorchMemoryToolConfig
            trace_alloc_max_entries: 100000
            stack_depth: 32
      log_prob_micro_batch_size: 1
      ulysses_sequence_parallel_size: 1
      entropy_from_logits_with_chunking: false
    """
    dict_conf = OmegaConf.create(config_str)
    actor_rollout_ref_worker = ActorRolloutRefWorker(dict_conf, role="ref")
    actor_rollout_ref_worker.init_model()

    model_config = actor_rollout_ref_worker.ref_module_fsdp._fsdp_wrapped_module.config
    assert model_config.hidden_size == 1536

    # set ref.model to null, fallback to default case where actor is the same as reference
    dict_conf["ref"]["model"] = None
    actor_rollout_ref_worker = ActorRolloutRefWorker(dict_conf, role="ref")
    actor_rollout_ref_worker.init_model()

    model_config = actor_rollout_ref_worker.ref_module_fsdp._fsdp_wrapped_module.config
    assert model_config.hidden_size == 896


def test_trainer_mode_releases_rollout_cache_when_enabled(monkeypatch):
    worker = object.__new__(ActorRolloutRefWorker)
    worker.rollout = SimpleNamespace(release=AsyncMock())
    worker.config = OmegaConf.create({"rollout": {"free_cache_engine": True}})
    worker._is_offload_param = False
    worker.actor_module_fsdp = object()

    monkeypatch.setattr(fsdp_workers_module, "aggressive_empty_cache", lambda force_sync=True: None)
    monkeypatch.setattr(fsdp_workers_module, "set_expandable_segments", lambda enabled: None)
    monkeypatch.setattr(fsdp_workers_module, "log_gpu_memory_usage", lambda *args, **kwargs: None)

    asyncio.run(ActorRolloutRefWorker.trainer_mode(worker))

    worker.rollout.release.assert_awaited_once()


def test_compute_log_prob_respects_calculate_entropy_meta_flag():
    worker = object.__new__(ActorRolloutRefWorker)
    worker._is_actor = True
    worker._is_offload_param = False
    worker.config = OmegaConf.create(
        {
            "rollout": {
                "log_prob_micro_batch_size_per_gpu": 1,
                "log_prob_max_token_len_per_gpu": 16,
                "log_prob_use_dynamic_bsz": False,
                "temperature": 1.0,
            },
            "ref": {},
        }
    )
    worker.actor = SimpleNamespace(
        actor_module=SimpleNamespace(disable_adapter=lambda: nullcontext()),
        compute_log_prob=Mock(return_value={"log_probs": torch.zeros(1, 2)}),
    )
    worker.tokenizer = SimpleNamespace(pad_token_id=0)
    worker.ulysses_sharding_manager = nullcontext()
    worker._world_size = 1

    data = DataProto.from_dict(tensors={"input_ids": torch.ones(1, 2, dtype=torch.long)})
    data.meta_info["calculate_entropy"] = False

    output = ActorRolloutRefWorker.compute_log_prob(worker, data)

    worker.actor.compute_log_prob.assert_called_once()
    assert worker.actor.compute_log_prob.call_args.kwargs["calculate_entropy"] is False
    assert "entropys" not in output.batch


def test_compute_log_prob_honors_calculate_entropy_meta():
    worker = object.__new__(ActorRolloutRefWorker)
    worker._is_actor = True
    worker._is_offload_param = False
    worker._world_size = 1
    worker.config = OmegaConf.create(
        {
            "rollout": {
                "log_prob_micro_batch_size_per_gpu": 1,
                "log_prob_max_token_len_per_gpu": 16,
                "log_prob_use_dynamic_bsz": False,
                "temperature": 1.0,
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 1,
                "log_prob_max_token_len_per_gpu": 16,
                "log_prob_use_dynamic_bsz": False,
            },
        }
    )
    worker.tokenizer = SimpleNamespace(pad_token_id=0)
    worker.ulysses_sharding_manager = nullcontext()

    seen = {}

    def fake_compute_log_prob(data, calculate_entropy):
        seen["calculate_entropy"] = calculate_entropy
        return {"log_probs": torch.zeros(1, 1, dtype=torch.float32)}

    worker.actor = SimpleNamespace(
        actor_module=SimpleNamespace(disable_adapter=lambda: nullcontext()),
        compute_log_prob=fake_compute_log_prob,
    )

    data = DataProto.from_dict(
        tensors={"responses": torch.zeros(1, 1, dtype=torch.long)},
        meta_info={"calculate_entropy": False},
    )

    output = ActorRolloutRefWorker.compute_log_prob(worker, data)

    assert seen["calculate_entropy"] is False
    assert "entropys" not in output.batch
