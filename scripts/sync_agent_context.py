#!/usr/bin/env python3
"""Compatibility wrapper for the server-wide agent context sync tool."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def find_tool() -> Path:
    env_tool = os.environ.get("AGENT_CONTEXT_SYNC_TOOL")
    if env_tool:
        return Path(env_tool).expanduser().resolve()

    here = Path(__file__).resolve()
    candidates = []
    for parent in here.parents:
        candidates.append(parent / "agent-tools" / "sync_agent_context.py")
        candidates.append(parent.parent / "agent-tools" / "sync_agent_context.py")
    candidates.extend(
        [
            Path.home() / "agent-tools" / "sync_agent_context.py",
            Path.home() / ".local" / "share" / "agent-tools" / "sync_agent_context.py",
            Path("/data-1/agent-tools/sync_agent_context.py"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.resolve() != here:
            return candidate.resolve()
    raise SystemExit("agent context sync tool not found; set AGENT_CONTEXT_SYNC_TOOL=/path/to/sync_agent_context.py")


def main() -> None:
    tool = find_tool()
    os.execv(sys.executable, [sys.executable, str(tool), *sys.argv[1:]])


if __name__ == "__main__":
    main()
