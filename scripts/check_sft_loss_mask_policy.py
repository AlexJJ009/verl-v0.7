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
"""Enforce fail-closed SFT chat-template and loss-mask policy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "tests/special_sanity/sft_input_ids_mismatch_allowlist.json"
COLD_START_LAUNCHERS = (
    ROOT / "recipe/on_policy_wdl_sft/format_cold_start/run_sft_math_qwen3_1p7b_format.sh",
    ROOT / "recipe/on_policy_wdl_sft/format_cold_start/run_sft_code_qwen3_1p7b_kodcode_format.sh",
)
V3_COLD = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/math_qwen3_1p7b_cold_start_cotmask_v3.yaml"
INVALIDATED = (
    ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/math_qwen3_1p7b_cold_start.yaml",
    ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/math_qwen3_1p7b_cold_start_lr5e6_v2.yaml",
    ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/math_qwen3_1p7b_stage123_lr5e6_v2.yaml",
)


def mismatch_overrides(root: Path) -> set[str]:
    observed = set()
    for directory in ("recipe", "examples", "tests"):
        directory_path = root / directory
        if not directory_path.exists():
            continue
        for path in directory_path.rglob("*.sh"):
            if "ignore_input_ids_mismatch=True" in path.read_text(encoding="utf-8", errors="replace"):
                observed.add(path.relative_to(root).as_posix())
    return observed


def mismatch_override_digests(root: Path) -> dict[str, str]:
    import hashlib

    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in mismatch_overrides(root)}


def check(root: Path = ROOT) -> list[str]:
    failures = []
    allowlist = json.loads((root / ALLOWLIST.relative_to(ROOT)).read_text())
    allowed = {entry["path"]: entry["sha256"] for entry in allowlist["entries"]}
    observed = mismatch_override_digests(root)
    if set(observed) != set(allowed):
        failures.append(
            f"ignore_input_ids_mismatch=True inventory changed: observed={sorted(observed)} allowed={sorted(allowed)}"
        )
    for path in sorted(set(observed) & set(allowed)):
        if observed[path] != allowed[path]:
            failures.append(f"allowlisted mismatch override changed and requires re-audit: {path}")

    for launcher in COLD_START_LAUNCHERS:
        launcher_text = (root / launcher.relative_to(ROOT)).read_text()
        for required in ('"data.tokenize_whole_message=True"', '"data.ignore_input_ids_mismatch=False"'):
            if required not in launcher_text:
                failures.append(f"cold-start launcher missing {required}: {launcher.relative_to(ROOT)}")

    cold = yaml.safe_load((root / V3_COLD.relative_to(ROOT)).read_text())
    if cold["execution"].get("requires_whole_message_loss_mask") is not True:
        failures.append("V3 cold-start manifest must require whole-message loss-mask preflight")
    if cold["execution"].get("launch_allowed") is not False:
        failures.append("V3 launch must remain disabled until explicit admission")
    if "loss_mask_preflight_receipt" not in cold["paths"]:
        failures.append("V3 cold-start manifest has no loss-mask preflight receipt path")

    for path in INVALIDATED:
        manifest = yaml.safe_load((root / path.relative_to(ROOT)).read_text())
        launch_allowed = manifest.get("launch_allowed", manifest.get("execution", {}).get("launch_allowed"))
        if launch_allowed is not False or "invalidated" not in manifest["status"]:
            failures.append(f"invalid historical manifest is not fail-closed: {path.relative_to(ROOT)}")
    return failures


def main() -> int:
    failures = check()
    print(json.dumps({"ok": not failures, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
