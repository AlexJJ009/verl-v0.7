#!/usr/bin/env bash
# Resolve per-run manifest provenance, then invoke the manifest-native release hook.
set -euo pipefail

REPO=${REPO:-${VERL_REPO_ROOT:-/data-1/code/verl}}
: "${RUN_PREFIX:?RUN_PREFIX required}"
: "${EXPERIMENT_NORMALIZED_MANIFEST:?EXPERIMENT_NORMALIZED_MANIFEST required}"
: "${EXPERIMENT_PREFLIGHT_RECEIPT:?EXPERIMENT_PREFLIGHT_RECEIPT required}"
: "${EXPERIMENT_DEPLOYABILITY_RECEIPT:?EXPERIMENT_DEPLOYABILITY_RECEIPT required}"

EXPERIMENT_RUN_PROVENANCE=$(python3 - "$EXPERIMENT_NORMALIZED_MANIFEST" "$RUN_PREFIX" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); print(next(x for x in d['runs'] if x['run_prefix']==sys.argv[2])['provenance_file'])
PY
)
export EXPERIMENT_RUN_PROVENANCE
exec bash "$REPO/scripts/manifest_code_task_training_release_hook.sh"
