import argparse
import asyncio
from types import SimpleNamespace

import pytest
import torch
from torch import nn


class _FakeReplica:
    def __init__(self, workers=None):
        self.workers = workers or ["worker-0"]
        self.abort_calls = 0
        self.resume_calls = 0

    async def abort_all_requests(self):
        self.abort_calls += 1

    async def resume_all_requests(self):
        self.resume_calls += 1


class _FakeTrainer:
    world_size = 2

    def __init__(self):
        self.update_calls = []
        self.execute_calls = []

    def update_weights(self, eval_only=False):
        self.update_calls.append(eval_only)
        return ["trainer-update"]

    def execute_checkpoint_engine(self, *args, **kwargs):
        self.execute_calls.append((args, kwargs))
        return ["trainer-exec"]


class _FakeRolloutWorkerGroup:
    world_size = 1

    def __init__(self):
        self.update_calls = 0
        self.execute_calls = []

    def update_weights(self):
        self.update_calls += 1
        return ["rollout-update"]

    def execute_checkpoint_engine(self, *args, **kwargs):
        self.execute_calls.append((args, kwargs))
        return ["rollout-exec"]


def test_checkpoint_manager_propagates_eval_only_for_server_rollouts(monkeypatch):
    from verl.checkpoint_engine.base import CheckpointEngineManager
    import verl.checkpoint_engine.base as checkpoint_base

    trainer = _FakeTrainer()
    rollout_group = _FakeRolloutWorkerGroup()
    replicas = [_FakeReplica(workers=["worker-0"])]

    monkeypatch.setattr(checkpoint_base, "RayWorkerGroup", lambda *args, **kwargs: rollout_group)
    monkeypatch.setattr(checkpoint_base, "RayClassWithInitArgs", lambda *args, **kwargs: None)
    monkeypatch.setattr(checkpoint_base.ray, "get", lambda value: value)
    monkeypatch.setattr(checkpoint_base.CheckpointEngineRegistry, "get", lambda backend: object)
    monkeypatch.setattr(CheckpointEngineManager, "build_process_group", lambda self, rollout: None)

    manager = CheckpointEngineManager(backend="nccl", trainer=trainer, replicas=replicas)

    manager.update_weights(eval_only=True)

    assert trainer.update_calls == [True]
    assert rollout_group.update_calls == 1
    assert replicas[0].abort_calls == 1
    assert replicas[0].resume_calls == 1


def test_joint_vllm_model_registry_registration_is_idempotent():
    from verl.models.joint_model.vllm_registry import register_joint_vllm_model_architectures

    class _FakeRegistry:
        def __init__(self):
            self.models = {}
            self.calls = []

        def register_model(self, arch, model_cls):
            self.calls.append((arch, model_cls))
            self.models[arch] = model_cls

    registry = _FakeRegistry()

    register_joint_vllm_model_architectures(model_registry=registry)
    register_joint_vllm_model_architectures(model_registry=registry)

    assert registry.calls == [
        (
            "QwenJointForCausalLM",
            "verl.models.joint_model.vllm_modeling_joint_qwen3:QwenJointForCausalLM",
        )
    ]


def test_joint_vllm_registry_patches_layer_indexing_for_dual_submodels():
    pytest.importorskip("vllm")

    import vllm.model_executor.models.utils as model_utils
    import vllm.utils as vllm_utils
    import vllm.v1.utils as v1_utils

    from verl.models.joint_model.vllm_registry import patch_joint_vllm_layer_indexing

    patch_joint_vllm_layer_indexing()

    branch0_layer0 = model_utils.extract_layer_index("sub_models.0.model.layers.0.self_attn.attn")
    branch0_layer7 = model_utils.extract_layer_index("sub_models.0.model.layers.7.self_attn.attn")
    branch1_layer0 = model_utils.extract_layer_index("sub_models.1.model.layers.0.self_attn.attn")

    assert branch0_layer0 == 0
    assert branch0_layer7 == 7
    assert branch1_layer0 > branch0_layer7
    assert branch1_layer0 != branch0_layer0
    assert vllm_utils.extract_layer_index("sub_models.1.model.layers.0.self_attn.attn") == branch1_layer0
    assert v1_utils.extract_layer_index("sub_models.1.model.layers.0.self_attn.attn") == branch1_layer0


def test_vllm_async_server_imports_with_legacy_vllm():
    vllm = pytest.importorskip("vllm")

    from packaging import version

    if version.parse(vllm.__version__) >= version.parse("0.11.0"):
        pytest.skip("legacy vLLM compatibility path only applies before 0.11.0")

    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    assert vLLMHttpServer is not None


def test_filter_supported_vllm_cli_config_drops_unknown_keys():
    from verl.workers.rollout.vllm_rollout.vllm_async_server import _filter_supported_vllm_cli_config

    serve_parser = argparse.ArgumentParser(add_help=False)
    serve_parser.add_argument("--dtype")
    serve_parser.add_argument("--enable-sleep-mode", action="store_true")

    filtered, dropped = _filter_supported_vllm_cli_config(
        {
            "dtype": "bfloat16",
            "enable_sleep_mode": True,
            "logprobs_mode": "processed_logprobs",
        },
        serve_parser,
    )

    assert filtered == {
        "dtype": "bfloat16",
        "enable_sleep_mode": True,
    }
    assert dropped == {
        "logprobs_mode": "processed_logprobs",
    }


def test_vllm_zmq_handle_prefers_configured_ipc_dir(monkeypatch, tmp_path):
    from verl.workers.rollout.vllm_rollout.utils import get_zmq_ipc_handle

    monkeypatch.setenv("VERL_ZMQ_IPC_DIR", str(tmp_path))

    handle = get_zmq_ipc_handle("GPU-test")

    assert handle == f"ipc://{tmp_path}/rl-colocate-zmq-GPU-test.sock"
    assert tmp_path.is_dir()


def test_legacy_vllm_serve_parser_rejects_logprobs_mode_via_filter():
    vllm = pytest.importorskip("vllm")

    from packaging import version

    if version.parse(vllm.__version__) >= version.parse("0.11.0"):
        pytest.skip("legacy vLLM compatibility path only applies before 0.11.0")

    from verl.workers.rollout.vllm_rollout.vllm_async_server import (
        _build_vllm_cli_parser,
        _filter_supported_vllm_cli_config,
    )

    _, _, serve_parser = _build_vllm_cli_parser()

    filtered, dropped = _filter_supported_vllm_cli_config(
        {
            "dtype": "bfloat16",
            "logprobs_mode": "processed_logprobs",
        },
        serve_parser,
    )

    assert filtered == {"dtype": "bfloat16"}
    assert dropped == {"logprobs_mode": "processed_logprobs"}


def test_vllm_async_server_legacy_async_llm_without_reset_mm_cache(monkeypatch):
    pytest.importorskip("vllm")

    import verl.workers.rollout.vllm_rollout.vllm_async_server as server_module
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    created = {}

    class _FakeEngineClient:
        def __init__(self):
            self.collective_calls = []

        async def collective_rpc(self, method=None, kwargs=None, **_):
            self.collective_calls.append((method, kwargs))

        async def get_supported_tasks(self):
            return ()

    class _FakeAsyncLLM:
        @staticmethod
        def from_vllm_config(vllm_config, usage_context, enable_log_requests=None, disable_log_stats=None):
            created["client"] = _FakeEngineClient()
            return created["client"]

    class _FakeEngineArgs:
        enable_log_requests = False
        disable_log_stats = True

        def create_engine_config(self, usage_context=None):
            return SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_master_port=None))

    monkeypatch.setattr(
        server_module.AsyncEngineArgs,
        "from_cli_args",
        staticmethod(lambda args: _FakeEngineArgs()),
    )
    monkeypatch.setattr(server_module, "AsyncLLM", _FakeAsyncLLM)
    monkeypatch.setattr(server_module, "build_app", lambda args, supported_tasks=None: SimpleNamespace(state={}))

    async def _fake_init_app_state(engine_client, *args):
        return None

    async def _fake_run_unvicorn(app, args, server_address):
        return 8123, "server-task"

    monkeypatch.setattr(server_module, "init_app_state", _fake_init_app_state)
    monkeypatch.setattr(server_module, "run_unvicorn", _fake_run_unvicorn)

    server = vLLMHttpServer.__new__(vLLMHttpServer)
    server._dp_master_port = 4321
    server._server_address = "127.0.0.1"
    server.replica_rank = 0
    server.node_rank = 0
    server.model_config = SimpleNamespace(tokenizer=[0, 1, 2])

    asyncio.run(server.run_server(SimpleNamespace()))

    assert server.engine is created["client"]
    assert created["client"].collective_calls == [("monkey_patch_model", {"vocab_size": 3})]
    assert server._server_port == 8123
    assert server._server_task == "server-task"


def test_vllm_async_server_closes_reserved_port_sockets():
    pytest.importorskip("vllm")

    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    class _FakeSock:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    server = vLLMHttpServer.__new__(vLLMHttpServer)
    master_sock = _FakeSock()
    dp_rpc_sock = _FakeSock()
    dp_master_sock = _FakeSock()
    server._master_sock = master_sock
    server._dp_rpc_sock = dp_rpc_sock
    server._dp_master_sock = dp_master_sock

    server._close_reserved_port_sockets()

    assert master_sock.close_calls == 1
    assert dp_rpc_sock.close_calls == 1
    assert dp_master_sock.close_calls == 1
    assert server._master_sock is None
    assert server._dp_rpc_sock is None
    assert server._dp_master_sock is None


def test_vllm_async_server_legacy_async_llm_without_wait_for_requests_to_drain():
    pytest.importorskip("vllm")

    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    class _FakeLegacyAsyncLLM:
        pass

    server = vLLMHttpServer.__new__(vLLMHttpServer)
    server.engine = _FakeLegacyAsyncLLM()

    asyncio.run(server.wait_for_requests_to_drain())


def test_vllm_async_server_generate_omits_empty_multimodal_prompt(monkeypatch):
    pytest.importorskip("vllm")

    import verl.workers.rollout.vllm_rollout.vllm_async_server as server_module
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    captured = {}

    class _FakeConfig(SimpleNamespace):
        def get(self, key, default=None):
            return getattr(self, key, default)

    class _FakeEngine:
        async def list_loras(self):
            return []

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs

            async def _generator():
                yield SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            token_ids=[11, 12],
                            logprobs=None,
                            routed_experts=None,
                            finish_reason="stop",
                        )
                    ]
                )

            return _generator()

    monkeypatch.setattr(server_module, "TokensPrompt", lambda **kwargs: kwargs)

    server = vLLMHttpServer.__new__(vLLMHttpServer)
    server.config = _FakeConfig(
        max_model_len=32,
        response_length=8,
        prompt_length=8,
        repetition_penalty=1.0,
        enable_rollout_routing_replay=False,
    )
    server.model_config = SimpleNamespace(processor=None, lora_rank=0, lora={})
    server.engine = _FakeEngine()

    output = asyncio.run(server.generate(prompt_ids=[1, 2, 3], sampling_params={}, request_id="req-1"))

    assert captured["generate_kwargs"]["prompt"] == {"prompt_token_ids": [1, 2, 3]}
    assert "multi_modal_data" not in captured["generate_kwargs"]["prompt"]
    assert output.token_ids == [11, 12]


def test_patch_vllm_moe_model_weight_loader_ignores_non_moe_joint_models():
    from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

    class _JointLikeModel:
        pass

    patch_vllm_moe_model_weight_loader(_JointLikeModel())


def test_process_vllm_weights_after_loading_falls_back_for_legacy_vllm(monkeypatch):
    pytest.importorskip("vllm")

    import vllm.model_executor.model_loader.loader as loader_module
    import vllm.model_executor.model_loader.utils as loader_utils

    from verl.workers.rollout.vllm_rollout.utils import _process_vllm_weights_after_loading

    calls = []

    monkeypatch.delattr(loader_utils, "process_weights_after_loading", raising=False)
    monkeypatch.setattr(
        loader_module,
        "_process_weights_after_loading",
        lambda model, model_config, target_device: calls.append((model, model_config, target_device)),
    )

    model = object()
    model_config = object()
    device = object()

    _process_vllm_weights_after_loading(model, model_config, device)

    assert calls == [(model, model_config, device)]


def test_patch_transformers_tokenizers_backend_compat_adds_missing_special_tokens_property():
    from verl.workers.rollout.vllm_rollout.utils import patch_transformers_tokenizers_backend_compat

    class _FakeTokenizersBackend:
        SPECIAL_TOKENS_ATTRIBUTES = ["bos_token", "eos_token", "pad_token"]

        def __init__(self):
            self._special_tokens_map = {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<eos>",
            }
            self._extra_special_tokens = ["<extra>", "<bos>"]

    patch_transformers_tokenizers_backend_compat(_FakeTokenizersBackend)
    patch_transformers_tokenizers_backend_compat(_FakeTokenizersBackend)

    tokenizer = _FakeTokenizersBackend()

    assert tokenizer.all_special_tokens_extended == ["<bos>", "<eos>", "<extra>"]


def test_qwen_joint_tokenizer_has_extended_special_tokens_after_compat_patch():
    pytest.importorskip("vllm")

    from pathlib import Path

    from transformers import AutoTokenizer

    from verl.workers.rollout.vllm_rollout.utils import patch_transformers_tokenizers_backend_compat

    tokenizer_path = Path("/data-1/.cache/huggingface/QwenJoint-1.7B")
    if not tokenizer_path.exists():
        pytest.skip("joint tokenizer path is unavailable in this environment")

    patch_transformers_tokenizers_backend_compat()

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)

    assert hasattr(tokenizer, "all_special_tokens_extended")
    assert [str(token) for token in tokenizer.all_special_tokens_extended] == tokenizer.all_special_tokens


class _FakeIntermediateTensors(dict):
    pass


class _FakeSubModel(nn.Module):
    def __init__(self, bias, output_kind="tensor"):
        super().__init__()
        self.bias = bias
        self.output_kind = output_kind
        self.load_calls = []
        self.forward_calls = []

    def load_weights(self, weights):
        weights = list(weights)
        self.load_calls.append(weights)
        return {name for name, _ in weights}

    def compute_logits(self, hidden_states, sampling_metadata=None):
        return hidden_states + self.bias

    def forward(self, input_ids=None, positions=None, intermediate_tensors=None, inputs_embeds=None):
        self.forward_calls.append(
            {
                "input_ids": input_ids,
                "positions": positions,
                "intermediate_tensors": intermediate_tensors,
                "inputs_embeds": inputs_embeds,
            }
        )
        if self.output_kind == "intermediate":
            return _FakeIntermediateTensors(
                {
                    "hidden_states": torch.tensor([self.bias], dtype=torch.float32),
                    "residual": torch.tensor([self.bias + 100], dtype=torch.float32),
                }
            )
        return torch.tensor([[self.bias]], dtype=torch.float32)


def _build_joint_vllm_model(fusion_lambda=0.25, output_kind="tensor"):
    from verl.models.joint_model.vllm_modeling_joint_qwen3 import QwenJointForCausalLM

    model = QwenJointForCausalLM.__new__(QwenJointForCausalLM)
    nn.Module.__init__(model)
    model.fusion_lambda = fusion_lambda
    model.hidden_size = 1
    model._use_model2_only = False
    model.sub_models = nn.ModuleList([_FakeSubModel(1, output_kind), _FakeSubModel(10, output_kind)])
    return model


def test_joint_vllm_model_loads_prefixed_weights_for_rollout_mode():
    model = _build_joint_vllm_model()

    tensor0 = torch.tensor([1.0])
    tensor1 = torch.tensor([2.0])
    loaded = model.load_weights(
        [
            ("sub_models.0.model.layers.0.weight", tensor0),
            ("sub_models.1.model.layers.0.weight", tensor1),
        ]
    )

    assert model._use_model2_only is False
    assert model.sub_models[0].load_calls == [[("model.layers.0.weight", tensor0)]]
    assert model.sub_models[1].load_calls == [[("model.layers.0.weight", tensor1)]]
    assert loaded == {
        "sub_models.0.model.layers.0.weight",
        "sub_models.1.model.layers.0.weight",
    }


def test_joint_vllm_model_loads_unprefixed_weights_for_eval_mode():
    model = _build_joint_vllm_model()

    tensor = torch.tensor([3.0])
    loaded = model.load_weights([("model.layers.0.weight", tensor)])

    assert model._use_model2_only is True
    assert model.sub_models[0].load_calls == []
    assert model.sub_models[1].load_calls == [[("model.layers.0.weight", tensor)]]
    assert loaded == {"model.layers.0.weight"}


def test_joint_vllm_model_computes_fused_and_eval_only_logits():
    model = _build_joint_vllm_model(fusion_lambda=0.25)

    fused = model.compute_logits((torch.tensor([2.0]), torch.tensor([4.0])))
    assert torch.allclose(fused, torch.tensor([5.75]))

    model._use_model2_only = True
    eval_only = model.compute_logits(torch.tensor([4.0]))
    assert torch.allclose(eval_only, torch.tensor([14.0]))


def test_joint_vllm_model_packs_pipeline_intermediates_in_rollout_mode():
    model = _build_joint_vllm_model(output_kind="intermediate")

    output = model.forward(
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0]),
        intermediate_tensors=None,
    )

    assert isinstance(output, _FakeIntermediateTensors)
    assert set(output.keys()) == {
        "sub_model_0_hidden_states",
        "sub_model_0_residual",
        "sub_model_1_hidden_states",
        "sub_model_1_residual",
    }


def test_joint_vllm_model_returns_single_hidden_state_tensor_for_rollout_mode():
    model = _build_joint_vllm_model()

    output = model.forward(
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0]),
        intermediate_tensors=None,
    )

    assert isinstance(output, torch.Tensor)
    assert torch.equal(output, torch.tensor([[1.0, 10.0]]))


def test_joint_vllm_model_computes_fused_logits_from_concatenated_hidden_states():
    model = _build_joint_vllm_model(fusion_lambda=0.25)

    sample_hidden_states = model.forward(
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0]),
        intermediate_tensors=None,
    )[torch.tensor([0])]

    fused = model.compute_logits(sample_hidden_states, None)

    assert torch.allclose(fused, torch.tensor([[6.5]]))


def test_joint_vllm_model_uses_only_model2_forward_in_eval_mode():
    model = _build_joint_vllm_model(output_kind="intermediate")
    model._use_model2_only = True

    output = model.forward(
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0]),
        intermediate_tensors=None,
    )

    assert isinstance(output, _FakeIntermediateTensors)
    assert set(output.keys()) == {"hidden_states", "residual"}
    assert len(model.sub_models[0].forward_calls) == 0
    assert len(model.sub_models[1].forward_calls) == 1
