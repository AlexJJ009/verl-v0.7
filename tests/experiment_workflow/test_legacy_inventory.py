# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_tool():
    path = ROOT / "scripts/experiment_manifest.py"
    spec = importlib.util.spec_from_file_location("inventory_manifest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_live_inventory_is_explicit_and_stage123_is_manifest_native(tmp_path: Path):
    tool = load_tool()
    root = ROOT / "recipe/on_policy_wdl_sft"
    entries = []
    manifest = root / "experiment_manifest/stage123.yaml"
    native = {
        "code_task/run_code_task_qwen3_1p7b_stage123_queue.sh",
        "code_task/monitor_code_task_qwen3_1p7b_stage123_notify.sh",
    }
    for path in tool.runnable_paths(root):
        rel = path.relative_to(root).as_posix()
        entries.append("manifest-native" if rel in native and manifest.is_file() else "legacy")
    assert entries.count("manifest-native") == 2
    assert "legacy" in entries


def test_inventory_report_contains_no_silent_classification(tmp_path: Path):
    import subprocess

    output = tmp_path / "inventory.json"
    result = subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/experiment_manifest.py"),
            "inventory",
            "--root",
            str(ROOT / "recipe/on_policy_wdl_sft"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    report = json.loads(output.read_text())
    assert report["entries"]
    assert all(
        item["classification"] in {"manifest-native", "legacy-traceable", "legacy-unresolved"}
        and item["evidence_paths"]
        for item in report["entries"]
    )
