#!/usr/bin/env bash
# One-command colleague handoff for render/preflight or Hope submission.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HANDOFF_ENV=${HANDOFF_ENV:-"${SCRIPT_DIR}/handoff.env"}
HANDOFF_REGISTRY="${SCRIPT_DIR}/handoff_registry.json"

python3 - "$HANDOFF_REGISTRY" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert value["schema_version"] == 1
assert value["initializations"]["R01"]["model_name"] == "R01_ORDINARY_SFT_4B_AM1P4M"
assert value["initializations"]["R01"]["sft_dataset"] == "AM-1.4M"
assert value["initializations"]["R02"]["sft_dataset"] == "AM-1.4M"
assert value["downstream_rlvr"]["dataset_id"] == "hendrycks_math_7500"
assert value["downstream_rlvr"]["shared_by_arms"] is True
assert value["storage_layout"]["boundary_variable"] == "ROOT"
assert value["storage_layout"]["controlled_roots"] == ["DATASET_ROOT", "MODEL_ROOT", "STATE_ROOT"]
assert value["gates"]["G1b"] == "passed"
assert value["gates"]["G3"].endswith("pending")
assert value["gates"]["G4"].endswith("pending")
PY

if [ ! -f "$HANDOFF_ENV" ]; then
    echo "ERROR: handoff config is missing: $HANDOFF_ENV" >&2
    echo "ACTION: copy ${SCRIPT_DIR}/handoff.env.example to handoff.env and fill the Meituan-owned paths." >&2
    exit 2
fi

# shellcheck disable=SC1090
source "$HANDOFF_ENV"

: "${HANDOFF_ACTION:?HANDOFF_ACTION must be render or submit}"
: "${ROOT:?ROOT is required}"
: "${DATASET_ROOT:?DATASET_ROOT is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${STATE_ROOT:?STATE_ROOT is required}"
: "${HANDOFF_BUNDLE_ROOT:?HANDOFF_BUNDLE_ROOT is required}"
: "${HANDOFF_MANIFEST:=${HANDOFF_BUNDLE_ROOT}/approved-batch.json}"
: "${HANDOFF_RENDER_OUTPUT:=${HANDOFF_BUNDLE_ROOT}/rendered}"
: "${HANDOFF_SCRATCH_ROOT:=/tmp/rebuttal_rlvr_submit}"
: "${R01_MODEL_NAME:=R01_ORDINARY_SFT_4B_AM1P4M}"
: "${R01_MODEL_PATH:=${MODEL_ROOT}/${R01_MODEL_NAME}}"
: "${HOPE_SEMANTICS_RECEIPT:=${HANDOFF_BUNDLE_ROOT}/g3/hope-semantics.json}"
: "${G3_ADMISSION_RECEIPT:=${HANDOFF_BUNDLE_ROOT}/g3/g3-admission.json}"
: "${SUBMISSION_LEDGER:=${HANDOFF_BUNDLE_ROOT}/submission/global-submission-ledger.jsonl}"
: "${MAX_ACTIVE_JOBS:=8}"

case "$HANDOFF_ACTION" in
    render|submit) ;;
    *) echo "ERROR: HANDOFF_ACTION must be render or submit" >&2; exit 2 ;;
esac

for value in "$ROOT" "$DATASET_ROOT" "$MODEL_ROOT" "$STATE_ROOT" "$HANDOFF_BUNDLE_ROOT" "$HANDOFF_MANIFEST" "$HANDOFF_RENDER_OUTPUT" "$HANDOFF_SCRATCH_ROOT" "$R01_MODEL_PATH"; do
    case "$value" in
        /*) ;;
        *) echo "ERROR: handoff paths must be absolute: $value" >&2; exit 2 ;;
    esac
done

python3 - "$ROOT" "$DATASET_ROOT" "$MODEL_ROOT" "$STATE_ROOT" "$R01_MODEL_PATH" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
dataset_root, model_root, state_root, model_path = map(Path, sys.argv[2:])
for label, path in (
    ("DATASET_ROOT", dataset_root),
    ("MODEL_ROOT", model_root),
    ("STATE_ROOT", state_root),
):
    try:
        relative = path.resolve().relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {label} must be below ROOT") from exc
    if not relative.parts:
        raise SystemExit(f"ERROR: {label} must be a strict child of ROOT")
try:
    model_relative = model_path.resolve().relative_to(model_root.resolve())
except ValueError as exc:
    raise SystemExit("ERROR: R01_MODEL_PATH must be below MODEL_ROOT") from exc
if not model_relative.parts:
    raise SystemExit("ERROR: R01_MODEL_PATH must name a concrete directory below MODEL_ROOT")
PY

if [ ! -f "$HANDOFF_MANIFEST" ]; then
    echo "ERROR: approved batch manifest is missing: $HANDOFF_MANIFEST" >&2
    exit 2
fi
python3 - "$HANDOFF_MANIFEST" "$ROOT" "$DATASET_ROOT" "$MODEL_ROOT" "$STATE_ROOT" <<'PY'
import json
from pathlib import Path
import sys

manifest_path, root, dataset_root, model_root, state_root = sys.argv[1:]
manifest = json.loads(Path(manifest_path).read_text())
expected = {
    "root": root,
    "dataset_root": dataset_root,
    "model_root": model_root,
    "state_root": state_root,
}
for index, job in enumerate(manifest.get("jobs", [])):
    mismatches = [field for field, wanted in expected.items() if job.get(field) != wanted]
    if mismatches:
        raise SystemExit(
            f"ERROR: manifest job {index} differs from handoff controlled roots: {mismatches}"
        )
PY
if [ ! -d "$R01_MODEL_PATH" ]; then
    echo "ERROR: registered R01 slot is unresolved: ${R01_MODEL_NAME}" >&2
    echo "EXPECTED_PATH: ${R01_MODEL_PATH}" >&2
    exit 2
fi

args=(
    python3 "${SCRIPT_DIR}/submit_manifest.py"
    --manifest "$HANDOFF_MANIFEST"
    --render-output "$HANDOFF_RENDER_OUTPUT"
    --scratch-root "$HANDOFF_SCRATCH_ROOT"
    --max-active-jobs "$MAX_ACTIVE_JOBS"
)

if [ "$HANDOFF_ACTION" = "render" ]; then
    args+=(--render-only)
else
    : "${HOPE_SEMANTICS_RECEIPT:?submit requires Meituan G3 HOPE_SEMANTICS_RECEIPT}"
    : "${G3_ADMISSION_RECEIPT:?submit requires Meituan G3_ADMISSION_RECEIPT}"
    : "${SUBMISSION_LEDGER:?submit requires the G3-bound SUBMISSION_LEDGER}"
    args+=(
        --submit
        --hope-semantics-receipt "$HOPE_SEMANTICS_RECEIPT"
        --g3-admission-receipt "$G3_ADMISSION_RECEIPT"
        --submission-ledger "$SUBMISSION_LEDGER"
    )
fi

cd "$REPO_ROOT"
exec "${args[@]}"
