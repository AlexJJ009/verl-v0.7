#!/usr/bin/env bash
# Stable relative entry for a colleague already inside the admitted AFO worker.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENTRY="${REPO_ROOT}/recipe/on_policy_wdl_sft/rebuttal_rlvr/run_colleague.sh"

if [ ! -f "$ENTRY" ]; then
    echo "ERROR: recipe colleague entry is missing: $ENTRY" >&2
    exit 2
fi
exec bash "$ENTRY" "$@"
