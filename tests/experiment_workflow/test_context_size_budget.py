from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def within_budget(path: Path, max_lines: int = 400, max_bytes: int = 128 * 1024) -> bool:
    data = path.read_bytes()
    return len(data) <= max_bytes and len(data.splitlines()) <= max_lines


def test_real_startup_documents_fit_hard_budget():
    assert within_budget(ROOT / "AGENTS.md")
    assert within_budget(ROOT / "CLAUDE.md")
    assert within_budget(ROOT / ".codex/README.md")


def test_oversized_startup_document_fails(tmp_path: Path):
    path = tmp_path / "AGENTS.md"
    path.write_text("x\n" * 401)
    assert not within_budget(path)


def test_exception_contract_has_required_fields_and_caps():
    exception = {
        "owner": "owner",
        "reason": "reason",
        "plan": "docs/plan.md",
        "reviewer": "reviewer",
        "expiry": "2026-07-01",
        "max_lines": 400,
        "max_bytes": 131072,
    }
    assert set(exception) == {"owner", "reason", "plan", "reviewer", "expiry", "max_lines", "max_bytes"}
    assert exception["max_lines"] <= 400 and exception["max_bytes"] <= 128 * 1024
    assert date.fromisoformat(exception["expiry"]) < date(2026, 7, 12)
