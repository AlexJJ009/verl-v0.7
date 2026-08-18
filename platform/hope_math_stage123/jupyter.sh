#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
exec bash "${REPO_ROOT}/recipe/on_policy_wdl_sft/math_task/meituan/jupyter.sh"
