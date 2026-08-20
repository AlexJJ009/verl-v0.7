# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "docs/joint_training/constraints/principles/engineering_rule_catalog.md"


def tool():
    path = ROOT / "scripts/check_engineering_rule_catalog.py"
    spec = importlib.util.spec_from_file_location("rule_catalog", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_real_catalog_passes():
    assert tool().check(CATALOG) == []


def test_missing_trigger_action_or_failure_is_line_specific(tmp_path: Path):
    path = tmp_path / "bad.md"
    path.write_text(
        "# Rules\n\n## ER-001\n\n- Scope: x\n- Enforcement tier: judgment-only\n- Evidence source: source.md\n- When:\n- Do:\n- Otherwise:\n"
    )
    failures = tool().check(path)
    assert any("line 3" in item and "missing When" in item for item in failures)
    assert any("missing Do" in item for item in failures)
    assert any("missing Otherwise" in item for item in failures)
