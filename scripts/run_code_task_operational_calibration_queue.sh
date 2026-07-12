#!/usr/bin/env bash
# Serial AC-05/19/23 operational calibration queue: bootstrap history, freeze contract, acceptance.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MANIFEST=${CALIBRATION_MANIFEST:-$REPO/recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml}
MANIFEST_TOOL=${CALIBRATION_MANIFEST_TOOL:-$REPO/scripts/experiment_manifest.py}
CONTRACT_TOOL=${CALIBRATION_CONTRACT_TOOL:-$REPO/scripts/check_calibration_prediction_contract.py}
RUNNER=${CALIBRATION_RUNNER:-$REPO/scripts/run_code_task_operational_calibration.sh}
CALIBRATION_AUTHORIZATION_SCOPE=${CALIBRATION_AUTHORIZATION_SCOPE:-full}
case "$CALIBRATION_AUTHORIZATION_SCOPE" in
  full) PHASES=(stage1 stage2 stage3) ;;
  stage12_producer) PHASES=(stage1 stage2) ;;
  *) echo "ERROR: CALIBRATION_AUTHORIZATION_SCOPE must be full or stage12_producer" >&2; exit 1 ;;
esac

if [ "${1:-}" = "--sandbox-dry-run" ]; then
  scratch=$(mktemp -d /data-1/tmp/verl_agent_scratch/calibration-sandbox.XXXXXX)
  trap 'rm -rf "$scratch"' EXIT
  normalized="$scratch/stage123.normalized.json"
  python3 "$MANIFEST_TOOL" render "$MANIFEST" --format json > "$normalized"
  python3 - "$normalized" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(json.dumps({"ok":True,"manifest_sha256":d["manifest_sha256"],"phases":{p:{"model_provenance_class":d["calibration_workloads"][p]["model_provenance_class"],"model_sources":d["calibration_workloads"][p]["model_sources"]} for p in ("stage1","stage2","stage3")}},sort_keys=True))
PY
  exit 0
fi

[ -n "${TMUX:-}" ] || { echo "ERROR: calibration queue must run inside tmux" >&2; exit 1; }

SCRATCH=${CALIBRATION_QUEUE_SCRATCH:-/data-1/tmp/verl_agent_scratch/experiment_workflow/calibration_queue}
mkdir -p "$SCRATCH"
NORMALIZED_MANIFEST=${CALIBRATION_NORMALIZED_MANIFEST:-$SCRATCH/stage123.normalized.json}
python3 "$MANIFEST_TOOL" render "$MANIFEST" --format json > "$NORMALIZED_MANIFEST"
export CALIBRATION_NORMALIZED_MANIFEST="$NORMALIZED_MANIFEST"
: "${CALIBRATION_PREFLIGHT_REPORT:?CALIBRATION_PREFLIGHT_REPORT required}"
: "${CALIBRATION_PREFLIGHT_RECEIPT:?CALIBRATION_PREFLIGHT_RECEIPT required}"
: "${CALIBRATION_PREFLIGHT_POLICY:?CALIBRATION_PREFLIGHT_POLICY required}"
export CALIBRATION_EXPECTED_PROFILE_HASH=${CALIBRATION_EXPECTED_PROFILE_HASH:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["resource_profile"]["sha256"])' "$NORMALIZED_MANIFEST")}
export CALIBRATION_PREFLIGHT_RECEIPT_MAX_AGE_SECONDS=${CALIBRATION_PREFLIGHT_RECEIPT_MAX_AGE_SECONDS:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["preflight"]["receipt_max_age_seconds"])' "$NORMALIZED_MANIFEST")}

manifest_get() {
  python3 - "$NORMALIZED_MANIFEST" "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}

export STAGE1_INIT_MODEL_PATH=$(manifest_get paths.stage1_init_model)
export STAGE1_INIT_PROVENANCE_PATH=$(manifest_get paths.stage1_init_provenance)

algorithm=$(manifest_get calibration_policy.algorithm)
[ "$algorithm" = stage123_history_conformal_v1 ] || { echo "ERROR: unsupported calibration algorithm: $algorithm" >&2; exit 1; }
BOOTSTRAP_REPS=$(manifest_get calibration_policy.bootstrap_repetitions_per_phase)
ACCEPTANCE_REPS=$(manifest_get calibration_policy.acceptance_repetitions_per_phase)
DEADLINE=$(manifest_get calibration_policy.validation_deadline_seconds)
[ "$BOOTSTRAP_REPS" = 6 ] || { echo "ERROR: bootstrap_repetitions_per_phase must be 6" >&2; exit 1; }
[ "$ACCEPTANCE_REPS" = 3 ] || { echo "ERROR: acceptance_repetitions_per_phase must be 3" >&2; exit 1; }
[ "$DEADLINE" = 1800 ] || { echo "ERROR: validation_deadline_seconds must be 1800" >&2; exit 1; }

REPORT_ROOT=${CALIBRATION_REPORT_ROOT:-$(manifest_get calibration_policy.roots.report)}
HISTORY_ROOT=${CALIBRATION_HISTORY_ROOT:-$(manifest_get calibration_policy.roots.history)}
PREDICTION_ROOT=${CALIBRATION_PREDICTION_ROOT:-$(manifest_get calibration_policy.roots.prediction)}
case "$REPORT_ROOT" in
  */af1a407f/report|*/baaa596b_v2/report|*/7c1ed4e1_v3/report)
    echo "ERROR: diagnostic-only calibration root is ineligible: $REPORT_ROOT" >&2
    exit 1
    ;;
esac
HISTORY_INDEX=${CALIBRATION_HISTORY_INDEX:-$HISTORY_ROOT/trusted_history.json}
PREDICTION_CONTRACT=${CALIBRATION_PREDICTION_CONTRACT:-$PREDICTION_ROOT/prediction_contract.json}
mkdir -p "$REPORT_ROOT" "$HISTORY_ROOT" "$PREDICTION_ROOT"

status_path() {
  local role=$1 phase=$2 rep=$3
  printf '%s/%s/%s/rep_%s/%s.status.json\n' "$REPORT_ROOT" "$role" "$phase" "$rep" "$phase"
}

status_ok() {
  local path=$1
  python3 - "$path" <<'PY' >/dev/null 2>&1
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
data = json.loads(path.read_text())
if data.get("returncode") != 0 or data.get("timed_out") is True or data.get("fatal_termination") is True:
    raise SystemExit(1)
PY
}

status_failed() {
  local path=$1
  [ -f "$path" ] && ! status_ok "$path"
}

wait_for_status() {
  local role=$1 phase=$2 rep=$3 session=$4 path
  path=$(status_path "$role" "$phase" "$rep")
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep "${CALIBRATION_QUEUE_POLL_SECONDS:-5}"
  done
  status_ok "$path" || { echo "ERROR: calibration failed role=$role phase=$phase rep=$rep status=$path" >&2; exit 1; }
}

run_missing_rep() {
  local role=$1 phase=$2 rep=$3 path session container
  path=$(status_path "$role" "$phase" "$rep")
  if status_ok "$path"; then
    echo "skip existing role=$role phase=$phase rep=$rep"
    return 0
  fi
  if status_failed "$path"; then
    echo "ERROR: existing failed calibration status blocks queue: $path" >&2
    exit 1
  fi
  session="code_calibration_${role}_${phase}_rep${rep}"
  container="code-calibration-${role}-${phase}-rep${rep}"
  echo "start role=$role phase=$phase rep=$rep"
  env \
    ALLOW_CODE_OPERATIONAL_CALIBRATION="${ALLOW_CODE_OPERATIONAL_CALIBRATION:-1}" \
    CALIBRATION_ROLE="$role" \
    REP_INDEX="$rep" \
    CALIBRATION_ROOT="$REPORT_ROOT" \
    CALIBRATION_MANIFEST="$MANIFEST" \
    CALIBRATION_NORMALIZED_MANIFEST="$NORMALIZED_MANIFEST" \
    CALIBRATION_PREFLIGHT_REPORT="$CALIBRATION_PREFLIGHT_REPORT" \
    CALIBRATION_PREFLIGHT_RECEIPT="$CALIBRATION_PREFLIGHT_RECEIPT" \
    CALIBRATION_PREFLIGHT_POLICY="$CALIBRATION_PREFLIGHT_POLICY" \
    CALIBRATION_EXPECTED_PROFILE_HASH="$CALIBRATION_EXPECTED_PROFILE_HASH" \
    CALIBRATION_PREFLIGHT_RECEIPT_MAX_AGE_SECONDS="$CALIBRATION_PREFLIGHT_RECEIPT_MAX_AGE_SECONDS" \
    CALIBRATION_TMUX_NAME="$session" \
    CALIBRATION_CONTAINER_NAME="$container" \
    CALIBRATION_HISTORY_INDEX="$HISTORY_INDEX" \
    CALIBRATION_PREDICTION_CONTRACT="$PREDICTION_CONTRACT" \
    CALIBRATION_HISTORY_SHA256="${CALIBRATION_HISTORY_SHA256:-}" \
    CALIBRATION_PREDICTION_CONTRACT_SHA256="${CALIBRATION_PREDICTION_CONTRACT_SHA256:-}" \
    STAGE1_INIT_MODEL_PATH="$STAGE1_INIT_MODEL_PATH" \
    STAGE1_INIT_PROVENANCE_PATH="$STAGE1_INIT_PROVENANCE_PATH" \
    "$RUNNER" "$phase"
  wait_for_status "$role" "$phase" "$rep" "$session"
}

generate_history_snapshot() {
  if [ -f "$HISTORY_INDEX" ]; then
    echo "reuse frozen history snapshot: $HISTORY_INDEX"
    return 0
  fi
  CALIBRATION_REPO="$REPO" CALIBRATION_PHASES="${PHASES[*]}" python3 - "$REPORT_ROOT" "$HISTORY_INDEX" "$NORMALIZED_MANIFEST" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.environ["CALIBRATION_REPO"] + "/scripts")
from calibration_timing import load_validation_timing
from calibration_outcomes import load_generation_outcomes
from check_calibration_prediction_contract import canonical_json_sha256

report_root = Path(sys.argv[1])
history_path = Path(sys.argv[2])
manifest = json.loads(Path(sys.argv[3]).read_text())
profile_hash = manifest["resource_profile"]["sha256"]
semantics = manifest["semantics"]
dataset_hashes = semantics["validation_dataset_hashes"]
sampled_hash = semantics["sampled_decoding_semantic_hash"]
scorer_hash = semantics["scorer_hash"]
timeout_hash = semantics["timeout_policy_hash"]
topology_hashes = semantics["phase_topology_hashes"]
max_response_length = manifest["resource_profile"]["max_response_length"]
runs = []
phases = os.environ["CALIBRATION_PHASES"].split()
if phases not in (["stage1", "stage2"], ["stage1", "stage2", "stage3"]):
    raise SystemExit(f"invalid CALIBRATION_PHASES: {phases}")
for phase in phases:
    for rep in range(6):
        root = report_root / "bootstrap" / phase / f"rep_{rep}"
        status_path = root / f"{phase}.status.json"
        resources_path = root / f"{phase}.resources.json"
        metrics_paths = list((root / phase / "logs/metrics/OnPolicyWDLSFT-CodeTask").glob("*.jsonl"))
        generation_paths = list((root / phase / "logs/validation").glob("*/*.jsonl"))
        timeline_path = root / f"{phase}.validation_timeline.jsonl"
        if not status_path.is_file() or not resources_path.is_file() or len(metrics_paths) != 1 or len(generation_paths) != 1 or not timeline_path.is_file():
            raise SystemExit(f"missing bootstrap artifacts for {phase} rep {rep}")
        status = json.loads(status_path.read_text())
        resources = json.loads(resources_path.read_text())
        if status.get("returncode") != 0 or status.get("timed_out") is True:
            raise SystemExit(f"bootstrap failed for {phase} rep {rep}")
        metric_rows = [json.loads(line) for line in metrics_paths[0].read_text().splitlines() if line.strip()]
        if len(metric_rows) != 1 or metric_rows[0].get("step") != 0:
            raise SystemExit(f"invalid bootstrap metrics for {phase} rep {rep}")
        metric_data = metric_rows[0].get("data", {})
        trainer_elapsed = metric_data.get("timing_s/testing")
        try:
            timing = load_validation_timing(timeline_path, trainer_elapsed)
        except ValueError as exc:
            raise SystemExit(f"invalid bootstrap timing for {phase} rep {rep}: {exc}") from exc
        elapsed = timing["validation_elapsed_seconds"]
        workload = manifest["calibration_workloads"][phase]
        try:
            outcomes = load_generation_outcomes(generation_paths[0], workload)
        except ValueError as exc:
            raise SystemExit(f"invalid bootstrap outcomes for {phase} rep {rep}: {exc}") from exc
        peak_rss = resources.get("peak_rss_gib")
        idle = resources.get("all_gpu_idle_fraction_during_validation", resources.get("gpu_wait_fraction"))
        if elapsed is None or peak_rss is None or idle is None:
            raise SystemExit(f"incomplete bootstrap metrics for {phase} rep {rep}")
        completed = datetime.fromtimestamp(status_path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        runs.append({
            "run_id": f"bootstrap-{phase}-{rep}",
            "phase": phase,
            "evidence_role": "bootstrap_history",
            "artifact_root": str(root),
            "completed_at": completed,
            "release_gate_passed": True,
            "content_addressed": True,
            "artifacts_readable": True,
            "resource_profile_hash": profile_hash,
            "validation_dataset_hashes": dataset_hashes,
            "sampled_decoding_semantic_hash": sampled_hash,
            "phase_topology_hash": topology_hashes[phase],
            "scorer_hash": scorer_hash,
            "timeout_policy_hash": timeout_hash,
            "max_response_length": max_response_length,
            "workload_descriptor_sha256": canonical_json_sha256(workload),
            "outcome_schema_version": workload["outcome_schema_version"],
            "artifact_bindings": {
                name: {"path": str(path), "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()}
                for name, path in {
                    "status": status_path,
                    "resources": resources_path,
                    "metrics": metrics_paths[0],
                    "generation": generation_paths[0],
                    "timeline": timeline_path,
                }.items()
            },
            "metrics": {
                "validation_elapsed_seconds": elapsed,
                "trainer_validation_elapsed_seconds": timing["trainer_validation_elapsed_seconds"],
                "pre_readiness_elapsed_seconds": timing["pre_readiness_elapsed_seconds"],
                "peak_rss_gib": peak_rss,
                "all_gpu_idle_fraction_during_validation": idle,
                **outcomes,
            },
        })
history = {
    "schema_version": 1,
    "algorithm": "stage123_history_conformal_v1",
    "cutoff_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "phase_scope": phases,
    "source": "six bootstrap repetitions per authorized phase; infrastructure calibration only",
    "runs": runs,
}
history_path.parent.mkdir(parents=True, exist_ok=True)
tmp = history_path.with_name(history_path.name + f".tmp.{os.getpid()}")
tmp.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
os.replace(tmp, history_path)
PY
  chmod a-w "$HISTORY_INDEX" 2>/dev/null || true
}

freeze_prediction_contract() {
  if [ ! -f "$PREDICTION_CONTRACT" ]; then
    python3 "$CONTRACT_TOOL" \
      --contract "$PREDICTION_CONTRACT" \
      --manifest "$MANIFEST" \
      --history-index "$HISTORY_INDEX" \
      --authorization-scope "$CALIBRATION_AUTHORIZATION_SCOPE" \
      --write
    chmod a-w "$PREDICTION_CONTRACT" 2>/dev/null || true
  fi
  python3 "$CONTRACT_TOOL" \
    --contract "$PREDICTION_CONTRACT" \
    --manifest "$MANIFEST" \
    --history-index "$HISTORY_INDEX" \
    --authorization-scope "$CALIBRATION_AUTHORIZATION_SCOPE"
}

for phase in "${PHASES[@]}"; do
  for rep in $(seq 0 $((BOOTSTRAP_REPS - 1))); do
    run_missing_rep bootstrap "$phase" "$rep"
  done
done

generate_history_snapshot
freeze_prediction_contract

history_sha=$(sha256sum "$HISTORY_INDEX" | awk '{print $1}')
contract_sha=$(sha256sum "$PREDICTION_CONTRACT" | awk '{print $1}')
for phase in "${PHASES[@]}"; do
  for rep in $(seq 0 $((ACCEPTANCE_REPS - 1))); do
    CALIBRATION_HISTORY_SHA256="$history_sha" \
    CALIBRATION_PREDICTION_CONTRACT_SHA256="$contract_sha" \
    run_missing_rep acceptance "$phase" "$rep"
  done
done

echo "calibration queue complete report_root=$REPORT_ROOT history=$HISTORY_INDEX contract=$PREDICTION_CONTRACT"
