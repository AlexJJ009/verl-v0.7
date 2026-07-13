#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ -n "${TMUX:-}" ] || { echo "ERROR: calibration queue must run inside tmux" >&2; exit 1; }
exec python3 "$REPO/scripts/experiment_execution_core.py" queue "$@"
