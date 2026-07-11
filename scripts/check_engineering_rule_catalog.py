#!/usr/bin/env python3
"""Validate trigger/action/failure engineering rule records."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


REQUIRED = ("Scope", "Enforcement tier", "Evidence source", "When", "Do", "Otherwise")
TIERS = {"structural", "machine-check", "judgment-only"}


def check(path: Path) -> list[str]:
    failures = []
    lines = path.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if re.fullmatch(r"## ER-\d{3}", line)]
    ids = [lines[i][3:] for i in starts]
    if len(ids) != len(set(ids)):
        failures.append("duplicate rule ID")
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        block = lines[start:end]; fields = {}
        for offset, line in enumerate(block, start=start + 1):
            match = re.match(r"- ([^:]+):\s*(.*)", line)
            if match:
                fields[match.group(1)] = (match.group(2), offset + 1)
        for name in REQUIRED:
            if not fields.get(name, ("", 0))[0].strip():
                failures.append(f"line {start + 1}: {ids[position]} missing {name}")
        tier = fields.get("Enforcement tier", ("", 0))[0]
        if tier not in TIERS:
            failures.append(f"line {fields.get('Enforcement tier', ('', start + 1))[1]}: invalid enforcement tier")
        if tier in {"structural", "machine-check"}:
            checker = fields.get("Checker", ("", start + 1))[0]
            for token in ("test `", "failure `", "reachability `"):
                if token not in checker:
                    failures.append(f"line {start + 1}: {ids[position]} machine-enforced rule lacks {token.strip()}")
    if not starts:
        failures.append("no engineering rules found")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("catalog", type=Path); args = parser.parse_args()
    failures = check(args.catalog)
    for failure in failures: print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
