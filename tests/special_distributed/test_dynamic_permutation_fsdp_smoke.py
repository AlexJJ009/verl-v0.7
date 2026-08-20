# Copyright 2026 The verl authors.
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

"""Candidate-bound 8-GPU FSDP smoke for weak-logit Dynamic Permutation.

This is an executable smoke, not a pytest test.  It is intentionally small and
offline, but exercises the production remove-padding actor path, FlashAttention
2, BF16 FSDP, optimizer behavior, sharded checkpointing, and deterministic
resume.  The Slurm wrapper is the supported entry point.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from verl.models.joint_model.configuration_joint_qwen3 import QwenJointConfig
from verl.models.joint_model.modeling_joint_qwen3 import QwenJointForCausalLM
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config import FSDPActorConfig, PolicyLossConfig, WeakLogitPermutationConfig

SEED = 20260820


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return _digest(raw)


def _model_digest(model: FSDP) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _all_ranks_equal(tensor: torch.Tensor, label: str) -> None:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    for rank_tensor in gathered[1:]:
        torch.testing.assert_close(gathered[0], rank_tensor, atol=0.0, rtol=0.0, msg=label)


def _build_actor(*, freeze_model1: bool, enabled: bool, rho: float):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    config = QwenJointConfig(
        vocab_size=4096,
        hidden_size=384,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=6,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=1024,
        fusion_lambda=0.8,
        freeze_model1=freeze_model1,
        attn_implementation="flash_attention_2",
    )
    model = QwenJointForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    model = FSDP(
        model,
        use_orig_params=True,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)
    actor_config = FSDPActorConfig(
        strategy="fsdp",
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=2,
        ppo_epochs=1,
        use_dynamic_bsz=False,
        use_torch_compile=False,
        use_remove_padding=True,
        ulysses_sequence_parallel_size=1,
        rollout_n=1,
        policy_loss=PolicyLossConfig(loss_mode="wdl_sft"),
        weak_logit_permutation=WeakLogitPermutationConfig(
            enabled=enabled,
            rho=rho,
            seed=SEED,
            row_chunk_size=64,
            audit_invariants=True,
            audit_frequency=1,
            audit_rows=32,
        ),
    )
    return model, optimizer, scheduler, DataParallelPPOActor(actor_config, model, optimizer)


def _micro_batch(*, enabled: bool, rho: float, step: int) -> dict[str, torch.Tensor | float | int | bool]:
    generator = torch.Generator(device="cpu").manual_seed(SEED + step)
    input_ids = torch.randint(1, 4096, (2, 512), generator=generator, dtype=torch.long).cuda()
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, :37] = 0
    response_length = 256
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": (attention_mask.cumsum(dim=-1) - 1).clamp_min(0),
        "responses": input_ids[:, -response_length:],
        "response_mask": torch.ones((2, response_length), device="cuda", dtype=torch.long),
        "dynperm_sample_id": torch.tensor([1001, 2002], device="cuda", dtype=torch.int64),
        "apply_weak_logit_permutation": enabled,
        "dynperm_rho": rho,
        "dynperm_base_seed": SEED,
        "dynperm_global_step": step,
        "dynperm_actor_update_index": 0,
        "dynperm_row_chunk_size": 64,
        "dynperm_audit_rows": 32,
        "dynperm_entropy_atol": 2e-6,
        "dynperm_multiset_atol": 0.0,
        "return_submodel_log_probs": False,
        "submodel_log_prob_grad_indices": [],
    }


def _branch_grad_and_optimizer_state(model: FSDP, optimizer: torch.optim.Optimizer) -> tuple[list[float], list[int]]:
    grad_sums = [torch.zeros((), device="cuda", dtype=torch.float64) for _ in range(2)]
    state_counts = [torch.zeros((), device="cuda", dtype=torch.int64) for _ in range(2)]
    for name, parameter in model.named_parameters():
        for branch in range(2):
            marker = f"sub_models.{branch}."
            if marker in name:
                if parameter.grad is not None:
                    grad_sums[branch] += parameter.grad.detach().double().abs().sum()
                if parameter in optimizer.state and optimizer.state[parameter]:
                    state_counts[branch] += 1
    for value in [*grad_sums, *state_counts]:
        dist.all_reduce(value)
    return [value.item() for value in grad_sums], [int(value.item()) for value in state_counts]


def _one_step(model, optimizer, scheduler, actor, *, enabled: bool, rho: float, step: int) -> dict:
    batch = _micro_batch(enabled=enabled, rho=rho, step=step)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = actor._forward_micro_batch(batch, temperature=1.0)
    log_probs = output["log_probs"]
    _all_ranks_equal(log_probs.detach(), f"cross-rank log-prob mismatch at step {step}")
    loss = -(log_probs * batch["response_mask"]).sum() / batch["response_mask"].sum()
    loss.backward()
    gradients, _ = _branch_grad_and_optimizer_state(model, optimizer)
    optimizer.step()
    scheduler.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    _, optimizer_states = _branch_grad_and_optimizer_state(model, optimizer)
    telemetry = output.get("dynamic_permutation_telemetry") or {}
    if enabled and float(telemetry.get("dynperm/invariant_failures", 0.0)) != 0.0:
        raise AssertionError(f"Dynamic Permutation invariant failure: {telemetry}")
    return {
        "loss": float(loss.detach()),
        "log_probs_sha256": _tensor_digest(log_probs),
        "model_sha256": _model_digest(model),
        "cpu_rng_sha256": _tensor_digest(torch.random.get_rng_state()),
        "cuda_rng_sha256": _tensor_digest(torch.cuda.get_rng_state()),
        "branch_gradient_l1": gradients,
        "branch_optimizer_state_count": optimizer_states,
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "telemetry": telemetry,
    }


@torch.no_grad()
def _validation_digest(model: FSDP) -> str:
    model.eval()
    batch = _micro_batch(enabled=False, rho=0.0, step=91)
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        eval_only=True,
        use_cache=False,
    )
    _all_ranks_equal(output.logits, "cross-rank validation output mismatch")
    model.train()
    return _tensor_digest(output.logits)


def _checkpoint_namespaces(checkpoint: Path, rank: int, world_size: int) -> list[str]:
    state = torch.load(checkpoint / f"model_world_size_{world_size}_rank_{rank}.pt", map_location="cpu")
    local = sorted({key.split(".")[0] + "." + key.split(".")[1] for key in state if key.startswith("sub_models.")})
    all_local: list[list[str] | None] = [None] * world_size
    dist.all_gather_object(all_local, local)
    return sorted({item for items in all_local for item in (items or [])})


def _run_arm(output_root: Path, *, freeze_model1: bool, enabled: bool, rho: float, checkpoint_resume: bool) -> dict:
    model, optimizer, scheduler, actor = _build_actor(freeze_model1=freeze_model1, enabled=enabled, rho=rho)
    results = []
    checkpoint = output_root / "checkpoints" / f"freeze{int(freeze_model1)}"
    manager = FSDPCheckpointManager(model=model, optimizer=optimizer, lr_scheduler=scheduler)
    steps_before_checkpoint = 2 if checkpoint_resume else 3
    for step in range(steps_before_checkpoint):
        results.append(_one_step(model, optimizer, scheduler, actor, enabled=enabled, rho=rho, step=step))

    namespaces: list[str] = []
    resume_match = None
    if checkpoint_resume:
        manager.save_checkpoint(str(checkpoint), global_step=2, max_ckpt_to_keep=1)
        namespaces = _checkpoint_namespaces(checkpoint, dist.get_rank(), dist.get_world_size())
        if namespaces != ["sub_models.0", "sub_models.1"]:
            raise AssertionError(f"checkpoint namespaces are incomplete: {namespaces}")
        uninterrupted = _one_step(model, optimizer, scheduler, actor, enabled=enabled, rho=rho, step=2)
        manager.load_checkpoint(str(checkpoint))
        resumed = _one_step(model, optimizer, scheduler, actor, enabled=enabled, rho=rho, step=2)
        for key in ("loss", "log_probs_sha256", "model_sha256", "cpu_rng_sha256", "cuda_rng_sha256"):
            if uninterrupted[key] != resumed[key]:
                raise AssertionError(f"resume mismatch for {key}: {uninterrupted[key]} != {resumed[key]}")
        results.extend([uninterrupted, resumed])
        resume_match = True

    validation = _validation_digest(model)
    summary = {
        "freeze_model1": freeze_model1,
        "enabled": enabled,
        "rho": rho,
        "steps": results,
        "validation_sha256": validation,
        "checkpoint_namespaces": namespaces,
        "resume_match": resume_match,
    }
    del actor, optimizer, scheduler, manager, model
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return summary


def _assert_noop(disabled: dict, rho_zero: dict) -> None:
    for key in ("validation_sha256",):
        if disabled[key] != rho_zero[key]:
            raise AssertionError(f"rho=0 no-op mismatch for {key}")
    if len(disabled["steps"]) != len(rho_zero["steps"]):
        raise AssertionError("rho=0 comparator step counts differ")
    for index, (plain, zero) in enumerate(zip(disabled["steps"], rho_zero["steps"], strict=True)):
        for key in (
            "loss",
            "log_probs_sha256",
            "model_sha256",
            "cpu_rng_sha256",
            "cuda_rng_sha256",
            "branch_gradient_l1",
            "branch_optimizer_state_count",
        ):
            if plain[key] != zero[key]:
                raise AssertionError(f"rho=0 no-op mismatch at step {index} for {key}")


def _assert_freeze_contract(arm: dict) -> None:
    for step in arm["steps"]:
        model1_grad, model2_grad = step["branch_gradient_l1"]
        model1_states, model2_states = step["branch_optimizer_state_count"]
        if arm["freeze_model1"]:
            if model1_grad != 0.0 or model1_states != 0:
                raise AssertionError("frozen Model1 acquired gradients or optimizer state")
        elif model1_grad <= 0.0 or model1_states <= 0:
            raise AssertionError("trainable Model1 did not acquire gradients and optimizer state")
        if model2_grad <= 0.0 or model2_states <= 0:
            raise AssertionError("Model2 did not acquire gradients and optimizer state")


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    marker = (args.workspace / ".candidate-sha").read_text().strip()
    if marker != args.candidate_sha:
        raise SystemExit(f"candidate marker mismatch: {marker} != {args.candidate_sha}")
    if any(
        not path.is_symlink() and path.lstat().st_mode & 0o222 for path in [args.workspace, *args.workspace.rglob("*")]
    ):
        raise SystemExit("candidate workspace must be read-only")
    if dist.get_world_size() != 8:
        raise SystemExit(f"this gate requires exactly 8 ranks, got {dist.get_world_size()}")

    rank = dist.get_rank()
    receipt: dict = {
        "schema_version": 1,
        "candidate_sha": args.candidate_sha,
        "world_size": dist.get_world_size(),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "dtype": "bfloat16",
        "attention": "flash_attention_2",
        "remove_padding": True,
        "formal_experiment": False,
        "arms": [],
    }
    for freeze in (False, True):
        disabled = _run_arm(args.output, freeze_model1=freeze, enabled=False, rho=0.0, checkpoint_resume=False)
        rho_zero = _run_arm(args.output, freeze_model1=freeze, enabled=True, rho=0.0, checkpoint_resume=False)
        rho_one = _run_arm(args.output, freeze_model1=freeze, enabled=True, rho=1.0, checkpoint_resume=True)
        _assert_noop(disabled, rho_zero)
        _assert_freeze_contract(rho_one)
        baseline_time = _median([item["elapsed_seconds"] for item in rho_zero["steps"]])
        active_time = _median([item["elapsed_seconds"] for item in rho_one["steps"][:3]])
        baseline_memory = max(item["peak_memory_bytes"] for item in rho_zero["steps"])
        active_memory = max(item["peak_memory_bytes"] for item in rho_one["steps"][:3])
        overhead = {
            "median_step_time_ratio": active_time / baseline_time,
            "peak_memory_ratio": active_memory / baseline_memory,
        }
        if overhead["median_step_time_ratio"] > 1.25:
            raise AssertionError(f"step-time overhead exceeds 25%: {overhead}")
        if overhead["peak_memory_ratio"] > 1.15:
            raise AssertionError(f"peak-memory overhead exceeds 15%: {overhead}")
        receipt["arms"].append(
            {
                "freeze_model1": freeze,
                "disabled": disabled,
                "rho_zero": rho_zero,
                "rho_one": rho_one,
                "overhead": overhead,
            }
        )

    receipt["result"] = "PASS"
    receipt["finished_at_epoch"] = time.time()
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        receipt_path = args.output / "gpu_fsdp_smoke_receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"result": "PASS", "receipt": str(receipt_path), "candidate_sha": args.candidate_sha}))
    dist.barrier()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        main()
    finally:
        dist.destroy_process_group()
