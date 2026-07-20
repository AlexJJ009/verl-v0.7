#!/usr/bin/env python3
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
"""Fail closed unless real SFT samples supervise complete reasoning and answers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reasoning_body(content: str) -> str:
    if "<think>" not in content or "</think>" not in content:
        raise ValueError("assistant target has no complete <think> block")
    return content.split("<think>", 1)[1].split("</think>", 1)[0].strip()


def validate(args: argparse.Namespace) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    config = OmegaConf.create(
        {
            "messages_key": args.messages_key,
            "max_length": args.max_length,
            "truncation": "error",
            "pad_mode": "no_padding",
            "shuffle": True,
            "seed": args.seed,
            "tokenize_whole_message": True,
            "ignore_input_ids_mismatch": False,
        }
    )
    dataset = MultiTurnSFTDataset(
        str(args.dataset), tokenizer, config, max_samples=args.samples
    )
    failures = []
    rows = []
    for index in range(len(dataset)):
        item = dataset[index]
        supervised_text = tokenizer.decode(item["input_ids"][item["loss_mask"].bool()])
        messages = dataset.messages[index]
        assistant_messages = [message for message in messages if message["role"] == "assistant"]
        if not assistant_messages:
            failures.append({"sample_index": index, "failed_checks": ["has_assistant_message"]})
            continue
        checks = {
            "has_think_open": "<think>" in supervised_text,
            "has_think_close": "</think>" in supervised_text,
            "has_answer_open": "<answer>" in supervised_text,
            "has_answer_close": "</answer>" in supervised_text,
            "has_eos": tokenizer.eos_token is None or tokenizer.eos_token in supervised_text,
            "system_user_excluded": all(
                message["content"] not in supervised_text
                for message in messages
                if message["role"] in {"system", "user"}
            ),
            "all_reasoning_present": all(
                reasoning_body(message["content"]) in supervised_text for message in assistant_messages
            ),
        }
        failed_checks = sorted(name for name, passed in checks.items() if not passed)
        rows.append(
            {
                "sample_index": index,
                "sequence_tokens": int(len(item["input_ids"])),
                "supervised_tokens": int(item["loss_mask"].sum()),
                "checks": checks,
            }
        )
        if failed_checks:
            failures.append({"sample_index": index, "failed_checks": failed_checks})

    return {
        "schema_version": 1,
        "ok": not failures and len(dataset) == args.samples,
        "model": str(args.model),
        "dataset": str(args.dataset),
        "dataset_sha256": sha256(args.dataset),
        "seed": args.seed,
        "requested_samples": args.samples,
        "validated_samples": len(dataset),
        "tokenize_whole_message": True,
        "ignore_input_ids_mismatch": False,
        "failures": failures,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--messages-key", default="messages")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    result = validate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("ok", "validated_samples", "failures")}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
