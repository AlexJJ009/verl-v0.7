# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs/joint_training/reports/engineering_rule_migration.md"


def test_every_migration_source_exists_and_disposition_is_known():
    text = REPORT.read_text()
    rows = [line for line in text.splitlines() if line.startswith("| ER-")]
    assert rows
    allowed = {"retained", "reworded", "replaced-by-gate", "project-local", "global-candidate", "dropped-unverifiable"}
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert cells[1] in allowed
        sources = re.findall(r"`([^`]+)`", cells[2])
        assert sources and all((ROOT / source).exists() for source in sources)
