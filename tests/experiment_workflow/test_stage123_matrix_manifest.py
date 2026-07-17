from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/stage123_matrix_manifest.py"
MANIFEST = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/stage123_model2_kl_split_stage3.yaml"


def test_matrix_manifest_validates_six_runs_in_dependency_order() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOL), "render", str(MANIFEST)],
        check=True,
        text=True,
        capture_output=True,
    )
    manifest = json.loads(result.stdout)

    assert [run["id"] for run in manifest["runs"]] == [
        "frac25-stage2-nokl",
        "frac25-stage2-m2kl",
        "frac25-stage3-nokl-model1",
        "frac25-stage3-m2kl-model1",
        "frac25-stage3-nokl-model2",
        "frac25-stage3-m2kl-model2",
    ]
    assert manifest["validation"]["n"] == 3
    assert manifest["validation"]["primary_metric"] == "val-core/model2/HumanEval+/acc/mean@3"
    assert manifest["hypotheses"]["primary"].startswith("model2_only_kl")
    assert manifest["decision_policy"]["minimum_effect_pp"] == 1.0
