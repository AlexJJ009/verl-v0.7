#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Create the explicit Model1 selection receipt after reviewing math cold-start candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(path: Path) -> dict:
    config = path / "config.json"
    if not config.is_file():
        raise FileNotFoundError(config)
    weight_files = sorted(path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"no safetensors files under {path}")
    return {
        "model_path": str(path),
        "config_sha256": file_sha256(config),
        "weight_files": [{"path": str(item), "sha256": file_sha256(item)} for item in weight_files],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/data-2/model_weights/math_task/qwen3_1p7b_cold_start_v1"),
    )
    parser.add_argument("--review-note", required=True)
    parser.add_argument(
        "--allow-below-format-threshold",
        action="store_true",
        help="Allow an explicitly reviewed candidate that did not reach the pre-registered format threshold.",
    )
    args = parser.parse_args()
    candidates_path = args.artifact_root / "cold_start_candidates.json"
    candidates = json.loads(candidates_path.read_text())["candidates"]
    matches = [candidate for candidate in candidates if candidate["step"] == args.step]
    if len(matches) != 1:
        raise ValueError(f"step {args.step} is not a unique evaluated candidate")
    candidate = matches[0]
    format_gate_override = not candidate["passed_format_gate"]
    if format_gate_override and not args.allow_below_format_threshold:
        raise ValueError("selected candidate did not pass the pre-registered format gate")
    payload = {
        "schema_version": 1,
        "selected_step": args.step,
        "selection_policy": "manual_format_gate_override" if format_gate_override else "earliest_format_gate_pass",
        "format_gate_override": format_gate_override,
        "review_note": args.review_note,
        "candidate": candidate,
        "identity": model_identity(Path(candidate["model_path"])),
    }
    output = args.artifact_root / "model1_selection.json"
    if output.exists():
        raise FileExistsError(f"selection is immutable and already exists: {output}")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pause_marker = args.artifact_root / "PAUSE_FOR_MODEL1_SELECTION"
    if pause_marker.exists():
        pause_marker.unlink()
    print(output)


if __name__ == "__main__":
    main()
