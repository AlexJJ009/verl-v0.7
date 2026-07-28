#!/usr/bin/env bash
# One-command sequential six-cell run for one allocated eight-H20 worker.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${ROOT:?ROOT must point to the colleague persistent Meituan root}"

for rl_seed in 20260727 20260728 20260729; do
    for experiment in R01 R02; do
        echo "=== ${experiment} paired RLVR seed ${rl_seed} ==="
        bash "${SCRIPT_DIR}/run_colleague.sh" "$experiment" "$rl_seed"
    done
done
