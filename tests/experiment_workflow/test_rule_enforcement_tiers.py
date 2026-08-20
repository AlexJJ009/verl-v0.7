# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from test_engineering_rule_catalog import tool


def test_machine_checkable_rule_requires_checker_test_failure_and_reachability(tmp_path: Path):
    path = tmp_path / "bad.md"
    path.write_text(
        "# Rules\n\n## ER-001\n\n- Scope: x\n- Enforcement tier: machine-check\n- Evidence source: source.md\n- When: x\n- Do: y\n- Otherwise: z\n"
    )
    failures = tool().check(path)
    assert len([item for item in failures if "machine-enforced rule lacks" in item]) == 3


def test_judgment_rule_does_not_require_fake_checker(tmp_path: Path):
    path = tmp_path / "good.md"
    path.write_text(
        "# Rules\n\n## ER-001\n\n- Scope: x\n- Enforcement tier: judgment-only\n- Evidence source: source.md\n- When: x\n- Do: y\n- Otherwise: z\n"
    )
    assert tool().check(path) == []
