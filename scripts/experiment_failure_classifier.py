#!/usr/bin/env python3
"""Classify experiment terminal failures into one deterministic primary reason."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


RULES = (
    ("ray_memory_kill", re.compile(r"Killed due to the node running low on memory|memory monitor.*killed", re.I)),
    ("gpu_oom", re.compile(r"CUDA out of memory|torch\.OutOfMemoryError|OutOfMemoryError.*GPU", re.I)),
    ("model_provenance_failure", re.compile(r"provenance.*(?:mismatch|missing|invalid)|model.*hash mismatch", re.I)),
    ("config_path_failure", re.compile(r"No such file or directory|FileNotFoundError|path.*(?:missing|mismatch)|config.*(?:invalid|mismatch)", re.I)),
    ("dependency_failure", re.compile(r"ModuleNotFoundError|ImportError|command not found|missing dependency", re.I)),
    ("release_failure", re.compile(r"release (?:hook|gate).*(?:failed|error)|DB/W&B release.*not verify", re.I)),
    ("scorer_timeout", re.compile(r"Reward computation timed out|scorer.*timeout|timed out.*data_source", re.I)),
    ("early_queue_exit", re.compile(r"queue.*(?:exited|stopped).*(?:early|before)|stopped before configured final checkpoint", re.I)),
)


def classify(text: str) -> dict[str, object]:
    matches = [name for name, pattern in RULES if pattern.search(text)]
    return {
        "schema_version": 1,
        "primary_reason": matches[0] if matches else "unknown",
        "secondary_signals": matches[1:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = classify(args.path.read_text(encoding="utf-8", errors="replace"))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
