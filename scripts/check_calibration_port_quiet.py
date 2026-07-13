#!/usr/bin/env python3
"""Fail when a listening TCP socket occupies a controlled calibration port."""

from __future__ import annotations

import os
from pathlib import Path


def controlled_ports(spec: str) -> set[int]:
    ports: set[int] = set()
    for item in spec.split(","):
        low, high = (int(value) for value in item.split("-", 1))
        if not 1024 <= low <= high <= 65535:
            raise ValueError(f"invalid controlled port range: {item}")
        ports.update(range(low, high + 1))
    return ports


def listening_ports(paths: tuple[Path, ...] = (Path("/proc/net/tcp"), Path("/proc/net/tcp6"))) -> set[int]:
    ports: set[int] = set()
    for path in paths:
        try:
            lines = path.read_text().splitlines()[1:]
        except FileNotFoundError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 4 or fields[3] != "0A":
                continue
            try:
                ports.add(int(fields[1].rsplit(":", 1)[1], 16))
            except (IndexError, ValueError):
                continue
    return ports


def main() -> int:
    domains = controlled_ports(os.environ["CALIBRATION_PORT_DOMAINS"])
    busy = sorted(domains & listening_ports())
    if busy:
        raise SystemExit(f"controlled calibration ports are busy: {busy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
