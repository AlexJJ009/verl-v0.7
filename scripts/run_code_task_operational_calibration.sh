#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
echo "WARNING: legacy calibration runner delegates to experiment_execution_core.py" >&2
exec python3 "$REPO/scripts/experiment_execution_core.py" phase "$@"
