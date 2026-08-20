#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Create an immutable, explicit Code Cold Start Model1 selection receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from packaging.version import Version


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(path: Path) -> dict:
    required = ("config.json", "tokenizer_config.json", "chat_template.jinja")
    for filename in required:
        if not (path / filename).is_file():
            raise FileNotFoundError(path / filename)
    config = json.loads((path / "config.json").read_text())
    transformers_version = config.get("transformers_version")
    if (
        config.get("model_type") == "qwen3"
        and transformers_version
        and Version("4.57.2") < Version(transformers_version) < Version("5.0.0")
    ):
        raise ValueError("Qwen3 Model1 config would trigger the Transformers Mistral-regex false positive")
    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"no safetensors files under {path}")
    return {
        "model_path": str(path),
        "config_sha256": file_sha256(path / "config.json"),
        "tokenizer_config_sha256": file_sha256(path / "tokenizer_config.json"),
        "chat_template_sha256": file_sha256(path / "chat_template.jinja"),
        "weight_files": [{"path": str(item), "sha256": file_sha256(item)} for item in weights],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/data-1/model_weights/code_task/qwen3_1p7b_cold_start_cotmask_v3_steps"),
    )
    parser.add_argument("--review-note", required=True)
    parser.add_argument("--allow-below-format-threshold", action="store_true")
    args = parser.parse_args()
    selection_path = args.artifact_root / "model1_selection.json"
    current = json.loads(selection_path.read_text())
    candidates = current.get("candidates", [])
    matches = [candidate for candidate in candidates if int(candidate["step"]) == args.step]
    if len(matches) != 1:
        raise ValueError(f"step {args.step} is not a unique evaluated candidate")
    candidate = matches[0]
    format_gate_override = not bool(candidate["passed_format_gate"])
    if format_gate_override and not args.allow_below_format_threshold:
        raise ValueError("selected candidate did not pass the pre-registered format gate")
    payload = {
        "schema_version": 2,
        "selected_step": args.step,
        "selection_policy": "manual_format_gate_override" if format_gate_override else "earliest_format_gate_pass",
        "format_gate_override": format_gate_override,
        "review_note": args.review_note,
        "candidate": candidate,
        "identity": model_identity(Path(candidate["model_path"])),
        "supersedes_unselected_receipt_sha256": file_sha256(selection_path),
    }
    temporary = selection_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(selection_path)
    print(selection_path)


if __name__ == "__main__":
    main()
