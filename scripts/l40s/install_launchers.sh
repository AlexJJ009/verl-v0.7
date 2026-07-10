#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
INSTALL_ROOT=${INSTALL_ROOT:-/data-1/verl07}

mkdir -p "$INSTALL_ROOT"
ln -sfn "$REPO_ROOT" "$INSTALL_ROOT/verl"
ln -sfn "$REPO_ROOT/scripts/l40s/run_train.sh" "$INSTALL_ROOT/run_train.sh"

printf 'repo: %s -> %s\n' "$INSTALL_ROOT/verl" "$(readlink -f "$INSTALL_ROOT/verl")"
printf 'launcher: %s -> %s\n' "$INSTALL_ROOT/run_train.sh" "$(readlink -f "$INSTALL_ROOT/run_train.sh")"
