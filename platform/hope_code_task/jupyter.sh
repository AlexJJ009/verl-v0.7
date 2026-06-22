#!/usr/bin/env bash
# Thin platform shim for Meituan AFO code-task launches.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
exec bash "${REPO_ROOT}/recipe/on_policy_wdl_sft/code_task/meituan/jupyter.sh"
