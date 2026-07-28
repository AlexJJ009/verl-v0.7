#!/usr/bin/env python3
"""Executable H20 admission probes for the Meituan RLVR candidate image."""

from __future__ import annotations

import argparse
from datetime import timedelta
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import subprocess


EXPECTED_FLASHINFER_BACKEND = "vllm.v1.attention.backends.flashinfer.FlashInferBackend"
EXPECTED_FLASHINFER_BUILDER = "vllm.v1.attention.backends.flashinfer.FlashInferMetadataBuilder"


def emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


def nccl_version_string(version: object) -> str:
    if isinstance(version, (tuple, list)):
        return ".".join(str(part) for part in version)
    if isinstance(version, int):
        return f"{version // 10000}.{version % 10000 // 100}.{version % 100}"
    return str(version)


def validate_versions(observed: dict[str, str], expected: dict[str, str]) -> None:
    mismatches = {
        name: {"expected": wanted, "observed": observed.get(name)}
        for name, wanted in expected.items()
        if observed.get(name) != wanted
    }
    if mismatches:
        raise RuntimeError(f"runtime package/version mismatch: {mismatches}")


def inspect_engine_attention_backend(worker: object) -> list[dict[str, object]]:
    """Run inside each vLLM worker and inspect initialized attention groups."""
    records: list[dict[str, object]] = []
    model_runner = getattr(worker, "model_runner")
    for cache_group_id, groups in enumerate(model_runner.attn_groups):
        for group in groups:
            backend_module, backend_qualname = group.backend.full_cls_name()
            records.append(
                {
                    "cache_group_id": cache_group_id,
                    "backend": f"{backend_module}.{backend_qualname}",
                    "builder_classes": [
                        f"{type(builder).__module__}.{type(builder).__qualname__}"
                        for builder in group.metadata_builders
                    ],
                    "layer_count": len(group.layer_names),
                    "first_layer": group.layer_names[0] if group.layer_names else "",
                    "last_layer": group.layer_names[-1] if group.layer_names else "",
                }
            )
    return records


def validate_engine_attention_records(
    per_worker: list[list[dict[str, object]]], expected_workers: int
) -> None:
    if len(per_worker) != expected_workers:
        raise RuntimeError(
            f"vLLM backend RPC returned {len(per_worker)} workers; expected {expected_workers}"
        )
    for worker_index, records in enumerate(per_worker):
        if not records:
            raise RuntimeError(f"vLLM worker {worker_index} has no initialized attention groups")
        for record in records:
            if record.get("backend") != EXPECTED_FLASHINFER_BACKEND:
                raise RuntimeError(
                    f"vLLM worker {worker_index} uses a non-FlashInfer backend: {record}"
                )
            builders = record.get("builder_classes")
            if not isinstance(builders, list) or not builders:
                raise RuntimeError(
                    f"vLLM worker {worker_index} has no initialized metadata builder: {record}"
                )
            if set(builders) != {EXPECTED_FLASHINFER_BUILDER}:
                raise RuntimeError(
                    f"vLLM worker {worker_index} uses a non-FlashInfer metadata builder: {record}"
                )


def nvidia_inventory() -> tuple[list[dict[str, str]], str]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows: list[dict[str, str]] = []
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        index, name, driver = (part.strip() for part in line.split(",", maxsplit=2))
        rows.append({"index": index, "name": name, "driver": driver})

    summary = subprocess.run(
        ["nvidia-smi"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)+)", summary.stdout)
    if match is None:
        raise RuntimeError("cannot parse CUDA driver API version from nvidia-smi")
    return rows, match.group(1)


def inventory_probe(args: argparse.Namespace) -> None:
    import flash_attn
    import flashinfer
    import torch
    import vllm

    rows, cuda_driver = nvidia_inventory()
    if len(rows) != args.expected_gpus:
        raise RuntimeError(f"expected {args.expected_gpus} GPUs, observed {len(rows)}")
    if any(args.expected_gpu_name.casefold() not in row["name"].casefold() for row in rows):
        raise RuntimeError(f"one or more GPUs are not {args.expected_gpu_name}: {rows}")
    drivers = {row["driver"] for row in rows}
    if len(drivers) != 1:
        raise RuntimeError(f"worker exposes inconsistent NVIDIA drivers: {sorted(drivers)}")
    if args.expected_driver and drivers != {args.expected_driver}:
        raise RuntimeError(f"driver mismatch: expected={args.expected_driver} observed={sorted(drivers)}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != args.expected_gpus:
        raise RuntimeError("PyTorch does not see the expected CUDA device count")

    nccl_runtime = nccl_version_string(torch.cuda.nccl.version())
    observed_versions = {
        "cuda_driver": cuda_driver,
        "cuda_runtime": str(torch.version.cuda),
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "flashinfer": flashinfer.__version__,
        "flash_attn": flash_attn.__version__,
        "nccl_package": metadata.version("nvidia-nccl-cu12"),
        "nccl_runtime": nccl_runtime,
    }
    expected_versions = {
        "cuda_runtime": args.expected_cuda_runtime,
        "torch": args.expected_torch,
        "vllm": args.expected_vllm,
        "flashinfer": args.expected_flashinfer,
        "flash_attn": args.expected_flash_attn,
        "nccl_package": args.expected_nccl,
        "nccl_runtime": args.expected_nccl,
    }
    if args.expected_cuda_driver:
        expected_versions["cuda_driver"] = args.expected_cuda_driver
    validate_versions(observed_versions, expected_versions)

    allocations = []
    for index in range(args.expected_gpus):
        with torch.cuda.device(index):
            value = torch.full((1024,), index + 1, device=f"cuda:{index}", dtype=torch.float32)
            if float(value.sum().item()) != float((index + 1) * 1024):
                raise RuntimeError(f"CUDA allocation/compute failed on device {index}")
            allocations.append(torch.cuda.get_device_name(index))
    torch.cuda.synchronize()

    emit(
        {
            "ok": True,
            "probe": "inventory",
            "gpu_count": len(rows),
            "gpus": rows,
            "allocation_devices": allocations,
            "versions": observed_versions,
        }
    )


def distributed_probe(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    from flash_attn import flash_attn_func

    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(f"launch this mode with torchrun; missing env: {missing}")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != args.expected_world_size:
        raise RuntimeError(f"expected world_size={args.expected_world_size}, observed={world_size}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.timeout_seconds))
    try:
        value = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
        dist.all_reduce(value)
        expected = world_size * (world_size + 1) / 2
        if float(value.item()) != float(expected):
            raise RuntimeError(f"NCCL all_reduce mismatch: expected={expected} observed={value.item()}")

        shape = (1, 16, 2, 64)
        q = torch.randn(shape, device=f"cuda:{local_rank}", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(shape, device=f"cuda:{local_rank}", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(shape, device=f"cuda:{local_rank}", dtype=torch.bfloat16, requires_grad=True)
        output = flash_attn_func(q, k, v, causal=True)
        output.float().square().mean().backward()
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            if tensor.grad is None or not bool(torch.isfinite(tensor.grad).all().item()):
                raise RuntimeError(f"FlashAttention backward produced invalid {name} gradient")
        dist.barrier()
        if rank == 0:
            emit(
                {
                    "ok": True,
                    "probe": "distributed_nccl_flash_attention",
                    "world_size": world_size,
                    "all_reduce": float(value.item()),
                    "flash_attention_shape": shape,
                    "dtype": "bfloat16",
                }
            )
    finally:
        dist.destroy_process_group()


def vllm_probe(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams

    model = Path(args.model).resolve()
    for name in ("config.json", "tokenizer_config.json"):
        if not (model / name).is_file():
            raise FileNotFoundError(f"flat model directory is missing {name}: {model}")
    if not list(model.glob("*.safetensors")):
        raise FileNotFoundError(f"flat model directory has no safetensors: {model}")
    if os.environ.get("VLLM_ATTENTION_BACKEND") != "FLASHINFER":
        raise RuntimeError("VLLM_ATTENTION_BACKEND must be exactly FLASHINFER")
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 is required")

    config = AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    engine = LLM(
        model=str(model),
        tokenizer=str(model),
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    outputs = engine.generate(
        [args.prompt],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens, n=1),
        use_tqdm=False,
    )
    text = outputs[0].outputs[0].text
    if not text:
        raise RuntimeError("vLLM returned an empty generation")
    per_worker = engine.collective_rpc(
        inspect_engine_attention_backend,
        timeout=args.rpc_timeout,
    )
    validate_engine_attention_records(per_worker, args.tensor_parallel_size)
    emit(
        {
            "ok": True,
            "probe": "vllm_flashinfer_generation",
            "model": str(model),
            "model_type": getattr(config, "model_type", ""),
            "tokenizer": type(tokenizer).__name__,
            "tensor_parallel_size": args.tensor_parallel_size,
            "attention_backend_env": os.environ["VLLM_ATTENTION_BACKEND"],
            "attention_backend_workers": per_worker,
            "torch": torch.__version__,
            "vllm": metadata.version("vllm"),
            "output_text": text,
        }
    )


def self_test() -> None:
    if sum(range(1, 9)) != 36:
        raise AssertionError("all-reduce expectation is broken")
    valid_record = {
        "backend": EXPECTED_FLASHINFER_BACKEND,
        "builder_classes": [EXPECTED_FLASHINFER_BUILDER],
    }
    validate_engine_attention_records([[valid_record]], 1)
    invalid_record = {
        "backend": "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend",
        "builder_classes": [EXPECTED_FLASHINFER_BUILDER],
    }
    try:
        validate_engine_attention_records([[invalid_record]], 1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a non-FlashInfer engine backend must fail the verifier")
    validate_versions({"torch": "2.9.1+cu126"}, {"torch": "2.9.1+cu126"})
    try:
        validate_versions({"torch": "2.9.0+cu126"}, {"torch": "2.9.1+cu126"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("a runtime package version mismatch must fail the verifier")
    emit({"ok": True, "probe": "self_test"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("self-test")

    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--expected-gpus", type=int, default=8)
    inventory.add_argument("--expected-gpu-name", default="H20")
    inventory.add_argument("--expected-driver", default="")
    inventory.add_argument("--expected-cuda-driver", default="")
    inventory.add_argument("--expected-cuda-runtime", default="12.6")
    inventory.add_argument("--expected-torch", default="2.9.1+cu126")
    inventory.add_argument("--expected-vllm", default="0.12.0")
    inventory.add_argument("--expected-flashinfer", default="0.5.3")
    inventory.add_argument("--expected-flash-attn", default="2.8.1")
    inventory.add_argument("--expected-nccl", default="2.27.5")

    distributed = subparsers.add_parser("distributed")
    distributed.add_argument("--expected-world-size", type=int, default=8)
    distributed.add_argument("--timeout-seconds", type=int, default=180)

    vllm = subparsers.add_parser("vllm")
    vllm.add_argument("--model", required=True)
    vllm.add_argument("--tensor-parallel-size", type=int, default=8)
    vllm.add_argument("--max-model-len", type=int, default=512)
    vllm.add_argument("--max-tokens", type=int, default=8)
    vllm.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    vllm.add_argument("--rpc-timeout", type=float, default=120.0)
    vllm.add_argument("--enforce-eager", action="store_true")
    vllm.add_argument("--prompt", default="Return only the integer result of 1+1.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "inventory":
        inventory_probe(args)
    elif args.command == "distributed":
        distributed_probe(args)
    elif args.command == "vllm":
        vllm_probe(args)
    else:  # pragma: no cover - argparse owns the command choices
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
