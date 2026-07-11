#!/usr/bin/env bash
set -euo pipefail
log=${1:?log path required}
shift
mkdir -p "$(dirname "$log")"
"$@" 2>&1 | tee -a "$log"
exit "${PIPESTATUS[0]}"
