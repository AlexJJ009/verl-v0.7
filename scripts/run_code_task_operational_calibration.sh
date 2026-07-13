#!/usr/bin/env bash
set -euo pipefail
PHASE=${1:?phase required}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case "$PHASE" in stage1|stage2|stage3) ;; *) echo "ERROR: phase must be stage1, stage2, or stage3" >&2; exit 1 ;; esac
: "${CALIBRATION_ROLE:?CALIBRATION_ROLE must be bootstrap or acceptance}"
case "$CALIBRATION_ROLE" in bootstrap|acceptance) ;; *) echo "ERROR: CALIBRATION_ROLE must be bootstrap or acceptance" >&2; exit 1 ;; esac
: "${REP_INDEX:?REP_INDEX required}"
case "$REP_INDEX" in ''|*[!0-9]*) echo "ERROR: REP_INDEX must be a non-negative integer" >&2; exit 1 ;; esac
BASE_ROOT=${CALIBRATION_ROOT:-/data-1/tmp/verl_agent_scratch/experiment_workflow/calibration}
ROOT="$BASE_ROOT/$CALIBRATION_ROLE/$PHASE/rep_${REP_INDEX}"
SESSION=${CALIBRATION_TMUX_NAME:-code_calibration_${CALIBRATION_ROLE}_${PHASE}_rep${REP_INDEX}}
CONTAINER=${CALIBRATION_CONTAINER_NAME:-code-calibration-${CALIBRATION_ROLE}-${PHASE}-rep${REP_INDEX}}
MANIFEST=${CALIBRATION_MANIFEST:-$REPO/recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml}
LOG="$ROOT/${PHASE}.host.log"
STATUS="$ROOT/${PHASE}.status.json"
RESOURCES="$ROOT/${PHASE}.resources.json"
READY="$ROOT/${PHASE}.validation_ready.json"
GPU_RESOURCES="$ROOT/${PHASE}.gpu_resources.json"
TIMELINE="$ROOT/${PHASE}.validation_timeline.jsonl"
[ "${ALLOW_CODE_OPERATIONAL_CALIBRATION:-0}" = 1 ] || { echo 'ERROR: calibration requires ALLOW_CODE_OPERATIONAL_CALIBRATION=1' >&2; exit 1; }
verify_preflight_admission() {
  : "${CALIBRATION_NORMALIZED_MANIFEST:?CALIBRATION_NORMALIZED_MANIFEST required}"
  : "${CALIBRATION_PREFLIGHT_REPORT:?CALIBRATION_PREFLIGHT_REPORT required}"
  : "${CALIBRATION_PREFLIGHT_RECEIPT:?CALIBRATION_PREFLIGHT_RECEIPT required}"
  : "${CALIBRATION_PREFLIGHT_POLICY:?CALIBRATION_PREFLIGHT_POLICY required}"
  : "${CALIBRATION_EXPECTED_PROFILE_HASH:?CALIBRATION_EXPECTED_PROFILE_HASH required}"
  : "${CALIBRATION_PREFLIGHT_RECEIPT_MAX_AGE_SECONDS:?CALIBRATION_PREFLIGHT_RECEIPT_MAX_AGE_SECONDS required}"
  python3 "$REPO/scripts/stage123_preflight_receipt.py" verify \
    --receipt "$CALIBRATION_PREFLIGHT_RECEIPT" \
    --normalized-manifest "$CALIBRATION_NORMALIZED_MANIFEST" \
    --report "$CALIBRATION_PREFLIGHT_REPORT" \
    --policy "$CALIBRATION_PREFLIGHT_POLICY" \
    --calibration-phase "$PHASE" \
    --profile-hash "$CALIBRATION_EXPECTED_PROFILE_HASH" \
    --max-age-seconds "$CALIBRATION_PREFLIGHT_RECEIPT_MAX_AGE_SECONDS" >/dev/null
}
verify_preflight_admission
if [ "$PHASE" = stage1 ]; then
  readarray -t stage1_identity < <(python3 - "$CALIBRATION_NORMALIZED_MANIFEST" "$REPO/recipe/on_policy_wdl_sft/code_task/calibration_workload_descriptor.py" <<'PY'
import importlib.util,json,sys
from pathlib import Path
manifest_path=Path(sys.argv[1]); helper_path=Path(sys.argv[2])
spec=importlib.util.spec_from_file_location('calibration_workload_descriptor', helper_path)
module=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(module)
manifest=json.loads(manifest_path.read_text()); workload=manifest['calibration_workloads']['stage1']
if workload.get('model_provenance_class') != 'sft_checkpoint': raise SystemExit('ERROR: Stage1 provenance class mismatch')
sources=workload.get('model_sources', [])
if len(sources) != 1 or sources[0].get('role') != 'rollout' or sources[0].get('state') != 'materialized':
    raise SystemExit('ERROR: Stage1 model source binding mismatch')
source=sources[0]; model=Path(source['path']).resolve(); provenance_doc=source.get('provenance', {}); provenance=Path(provenance_doc.get('path','')).resolve()
if Path(manifest['paths']['stage1_init_model']).resolve() != model: raise SystemExit('ERROR: Stage1 manifest model path mismatch')
if Path(manifest['paths']['stage1_init_provenance']).resolve() != provenance: raise SystemExit('ERROR: Stage1 manifest provenance path mismatch')
if module.artifact_sha256(model) != source.get('artifact_sha256'): raise SystemExit('ERROR: Stage1 model artifact hash mismatch')
if module.file_sha256(provenance) != provenance_doc.get('sha256'): raise SystemExit('ERROR: Stage1 provenance hash mismatch')
data=json.loads(provenance.read_text())
if Path(data.get('target_dir','')).resolve() != model: raise SystemExit('ERROR: Stage1 init provenance target mismatch')
print(model); print(provenance)
PY
  )
  [ "${#stage1_identity[@]}" = 2 ] || { echo 'ERROR: Stage1 identity resolution failed' >&2; exit 1; }
  STAGE1_INIT_MODEL_PATH=${stage1_identity[0]}
  STAGE1_INIT_PROVENANCE_PATH=${stage1_identity[1]}
fi
verify_acceptance_inputs() {
  local history=${CALIBRATION_HISTORY_INDEX:?CALIBRATION_HISTORY_INDEX required for acceptance}
  local contract=${CALIBRATION_PREDICTION_CONTRACT:?CALIBRATION_PREDICTION_CONTRACT required for acceptance}
  [ -f "$history" ] || { echo "ERROR: immutable history snapshot missing: $history" >&2; exit 1; }
  [ -f "$contract" ] || { echo "ERROR: prediction contract missing: $contract" >&2; exit 1; }
  python3 "$REPO/scripts/check_calibration_prediction_contract.py" \
    --contract "$contract" \
    --manifest "$MANIFEST" \
    --history-index "$history" >/dev/null
  python3 - "$MANIFEST" "$history" "$contract" "${CALIBRATION_HISTORY_SHA256:-}" "${CALIBRATION_PREDICTION_CONTRACT_SHA256:-}" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest = Path(sys.argv[1])
history = Path(sys.argv[2])
contract = Path(sys.argv[3])
expected_history = sys.argv[4]
expected_contract = sys.argv[5]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
history_sha = sha(history); contract_sha = sha(contract); manifest_sha = sha(manifest)
data = json.loads(contract.read_text())
hashes = data.get("hashes", {})
failures = []
if hashes.get("history_index_sha256") != history_sha:
    failures.append("prediction contract history_index_sha256 mismatch")
if hashes.get("manifest_sha256") != manifest_sha:
    failures.append("prediction contract manifest_sha256 mismatch")
if expected_history and expected_history != history_sha:
    failures.append("CALIBRATION_HISTORY_SHA256 mismatch")
if expected_contract and expected_contract != contract_sha:
    failures.append("CALIBRATION_PREDICTION_CONTRACT_SHA256 mismatch")
if failures:
    raise SystemExit("ERROR: " + "; ".join(failures))
PY
}
if [ "$CALIBRATION_ROLE" = acceptance ]; then
  verify_acceptance_inputs
fi
mkdir -p "$ROOT"
tmux has-session -t "$SESSION" 2>/dev/null && { echo "ERROR: tmux exists: $SESSION" >&2; exit 1; }
cat > "$ROOT/${PHASE}.worker.sh" <<EOF
#!/usr/bin/env bash
set -uo pipefail
start=\$(date +%s)
cleanup() { docker rm -f '$CONTAINER' >/dev/null 2>&1 || true; }
trap cleanup EXIT
sample_resources() {
  local command_pid=\$1 peak_bytes=0 container_pid='' cgroup='' current=0
  local readiness_wait_seconds=0 measurement_started=false
  while kill -0 "\$command_pid" 2>/dev/null; do
    if [ "\$measurement_started" = false ]; then
      if [ -s '$READY' ]; then
        measurement_started=true
      else
        readiness_wait_seconds=\$((readiness_wait_seconds + 1))
        sleep 1
        continue
      fi
    fi
    if [ -z "\$container_pid" ]; then
      container_pid=\$(docker inspect -f '{{.State.Pid}}' '$CONTAINER' 2>/dev/null || true)
      if [ -n "\$container_pid" ] && [ "\$container_pid" != 0 ]; then
        cgroup=\$(awk -F: '\$1 == "0" {print \$3}' "/proc/\$container_pid/cgroup" 2>/dev/null || true)
      fi
    fi
    if [ -n "\$cgroup" ] && [ -r "/sys/fs/cgroup\$cgroup/memory.current" ]; then
      current=\$(cat "/sys/fs/cgroup\$cgroup/memory.current")
      [ "\$current" -gt "\$peak_bytes" ] && peak_bytes=\$current
    fi
    sleep 1
  done
  local gpu_wait_attempts=0
  while [ ! -s '$GPU_RESOURCES' ] && [ "\$gpu_wait_attempts" -lt 100 ]; do
    gpu_wait_attempts=\$((gpu_wait_attempts + 1))
    sleep 0.1
  done
  [ -s '$GPU_RESOURCES' ] || { echo 'ERROR: GPU sampler artifact missing' >&2; return 1; }
  python3 - '$PHASE' "\$peak_bytes" '$GPU_RESOURCES' '$RESOURCES' <<'PY'
import json,sys
phase,peak,gpu_path,out=sys.argv[1:]
peak=int(peak); gpu=json.load(open(gpu_path))
data={
    'schema_version':1,
    'phase':phase,
    **gpu,
    'memory_sample_interval_seconds':1,
    'peak_rss_gib': peak / (1024 ** 3) if peak else None,
    'measurement_window':'validation_rollout_readiness_to_completion',
    'memory_source':'docker_cgroup_v2_memory.current_sampled_peak',
}
open(out,'w').write(json.dumps(data,indent=2,sort_keys=True)+'\n')
PY
}
validation_complete() {
  python3 - '$ROOT/$PHASE/logs/metrics/OnPolicyWDLSFT-CodeTask' <<'PY'
import glob,json,sys
required={'timing_s/testing','val-core/HumanEval+/acc/pass@1','val-core/MBPP+/acc/pass@1','val-core/LiveCodeBench/acc/pass@1'}
for path in glob.glob(sys.argv[1]+'/*.jsonl'):
    for line in open(path):
        try: data=json.loads(line).get('data',{})
        except json.JSONDecodeError: continue
        if required <= data.keys(): raise SystemExit(0)
raise SystemExit(1)
PY
}
fatal_runtime_error() {
  grep -Eq 'Fatal Python error|CUDA out of memory|OutOfMemoryError|Worker unexpectedly exits.*SYSTEM_ERROR|"failure_class"[[:space:]]*:[[:space:]]*"dependency_failure"|EvalPlus official evaluator is unavailable|LiveCodeBench official evaluator is unavailable|ModuleNotFoundError: No module named .(evalplus|lcb_runner).' '$LOG' 2>/dev/null
}
set +e
timeout --signal=TERM --kill-after=30s 1800s env DOCKER_CONTAINER_NAME='$CONTAINER' /data-1/verl07/run_train.sh env \
  CALIBRATION_ROLE='$CALIBRATION_ROLE' REP_INDEX='$REP_INDEX' \
  CALIBRATION_EVIDENCE_ROLE='$CALIBRATION_ROLE' \
  CALIBRATION_ALLOW_DEPLOYABLE='$([ "$CALIBRATION_ROLE" = acceptance ] && echo 1 || echo 0)' \
  WANDB_MODE='$([ "$CALIBRATION_ROLE" = bootstrap ] && echo disabled || echo "${WANDB_MODE:-offline}")' \
  WANDB_DISABLED='$([ "$CALIBRATION_ROLE" = bootstrap ] && echo true || echo "${WANDB_DISABLED:-false}")' \
  CODE_TASK_SKIP_DB_IMPORT='$([ "$CALIBRATION_ROLE" = bootstrap ] && echo 1 || echo "${CODE_TASK_SKIP_DB_IMPORT:-0}")' \
  QWEN3_1P7B_MODEL_PATH='${QWEN3_1P7B_MODEL_PATH:?}' \
  STAGE1_INIT_MODEL_PATH='${STAGE1_INIT_MODEL_PATH:-}' \
  STAGE1_INIT_PROVENANCE_PATH='${STAGE1_INIT_PROVENANCE_PATH:-}' \
  CALIBRATION_HUMANEVAL_PLUS_FILE='${CALIBRATION_HUMANEVAL_PLUS_FILE:?}' \
  CALIBRATION_MBPP_PLUS_FILE='${CALIBRATION_MBPP_PLUS_FILE:?}' \
  CALIBRATION_LIVE_CODE_BENCH_FILE='${CALIBRATION_LIVE_CODE_BENCH_FILE:?}' CALIBRATION_OUTPUT_ROOT='$ROOT/${PHASE}' \
  CALIBRATION_VALIDATION_READY_FILE='$READY' \
  CALIBRATION_VALIDATION_TIMELINE_FILE='$TIMELINE' \
  CALIBRATION_STAGE1_CKPT_DIR='${CALIBRATION_STAGE1_CKPT_DIR:-}' CALIBRATION_STAGE1_MODEL2='${CALIBRATION_STAGE1_MODEL2:-}' \
  CALIBRATION_STAGE1_RUN_PREFIX='${CALIBRATION_STAGE1_RUN_PREFIX:-}' CALIBRATION_STAGE1_HANDOFF_STEP='${CALIBRATION_STAGE1_HANDOFF_STEP:-}' \
  CALIBRATION_TRAIN_FILE='${CALIBRATION_TRAIN_FILE:-}' CALIBRATION_STAGE3_MODEL_PATH='${CALIBRATION_STAGE3_MODEL_PATH:-}' \
  bash /workspace/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_operational_calibration_phase.sh '$PHASE' >>'$LOG' 2>&1 &
command_pid=\$!
python3 '$REPO/scripts/sample_nvml_gpu_wait.py' --pid "\$command_pid" --ready-file '$READY' --output '$GPU_RESOURCES' --interval 0.2 & gpu_sampler_pid=\$!
sample_resources "\$command_pid" & sampler_pid=\$!
controlled_termination=false
fatal_termination=false
while kill -0 "\$command_pid" 2>/dev/null; do
  if fatal_runtime_error; then
    fatal_termination=true
    docker kill '$CONTAINER' >/dev/null 2>&1 || true
    kill -TERM "\$command_pid" 2>/dev/null || true
    break
  fi
  if validation_complete; then
    controlled_termination=true
    docker kill '$CONTAINER' >/dev/null 2>&1 || true
    kill -TERM "\$command_pid" 2>/dev/null || true
    break
  fi
  sleep 2
done
wait "\$command_pid"; rc=\$?
if [ "\$controlled_termination" = true ] && validation_complete && ! fatal_runtime_error; then rc=0; fi
if fatal_runtime_error; then rc=1; fi
wait "\$gpu_sampler_pid" || true
wait "\$sampler_pid" || true
end=\$(date +%s)
elapsed=\$((end-start))
timed_out=false
if [ "\$controlled_termination" = false ] && { [ "\$rc" -eq 124 ] || [ "\$rc" -eq 137 ]; } && [ "\$elapsed" -ge 1800 ]; then timed_out=true; fi
printf '{"phase":"%s","role":"%s","rep_index":%s,"decision":"%s","db_wandb_side_effects_allowed":%s,"returncode":%s,"elapsed_seconds":%s,"timed_out":%s,"controlled_termination_after_complete_validation":%s,"fatal_termination":%s}\n' '$PHASE' '$CALIBRATION_ROLE' '$REP_INDEX' '$([ "$CALIBRATION_ROLE" = bootstrap ] && echo infrastructure_only || echo acceptance_candidate)' '$([ "$CALIBRATION_ROLE" = bootstrap ] && echo false || echo true)' "\$rc" "\$elapsed" "\$timed_out" "\$controlled_termination" "\$fatal_termination" >'$STATUS'
exit \$rc
EOF
chmod +x "$ROOT/${PHASE}.worker.sh"
tmux new-session -d -s "$SESSION" "bash '$ROOT/${PHASE}.worker.sh'"
echo "started phase=$PHASE tmux=$SESSION container=$CONTAINER log=$LOG status=$STATUS"
