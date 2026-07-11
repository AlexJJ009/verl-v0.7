#!/usr/bin/env bash
set -euo pipefail
PHASE=${1:?phase required}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${CALIBRATION_ROOT:-/data-1/tmp/verl_agent_scratch/experiment_workflow/calibration}
SESSION=${CALIBRATION_TMUX_NAME:-code_calibration_${PHASE}}
CONTAINER=${CALIBRATION_CONTAINER_NAME:-code-calibration-${PHASE}}
LOG="$ROOT/${PHASE}.host.log"
STATUS="$ROOT/${PHASE}.status.json"
RESOURCES="$ROOT/${PHASE}.resources.json"
mkdir -p "$ROOT"
[ "${ALLOW_CODE_OPERATIONAL_CALIBRATION:-0}" = 1 ] || { echo 'ERROR: calibration requires ALLOW_CODE_OPERATIONAL_CALIBRATION=1' >&2; exit 1; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "ERROR: tmux exists: $SESSION" >&2; exit 1; }
cat > "$ROOT/${PHASE}.worker.sh" <<EOF
#!/usr/bin/env bash
set -uo pipefail
start=\$(date +%s)
cleanup() { docker rm -f '$CONTAINER' >/dev/null 2>&1 || true; }
trap cleanup EXIT
sample_resources() {
  local command_pid=\$1 samples=0 idle_samples=0 peak_bytes=0 container_pid='' cgroup='' current=0
  while kill -0 "\$command_pid" 2>/dev/null; do
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
    mapfile -t utils < <(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)
    if [ "\${#utils[@]}" -eq 8 ]; then
      samples=\$((samples + 1)); local all_idle=1 value
      for value in "\${utils[@]}"; do [ "\$value" -le 2 ] || all_idle=0; done
      [ "\$all_idle" -eq 1 ] && idle_samples=\$((idle_samples + 1))
    fi
    sleep 1
  done
  python3 - '$PHASE' "\$samples" "\$idle_samples" "\$peak_bytes" '$RESOURCES' <<'PY'
import json,sys
phase,samples,idle,peak,out=sys.argv[1:]
samples=int(samples); idle=int(idle); peak=int(peak)
data={
    'schema_version':1,
    'phase':phase,
    'sample_interval_seconds':1,
    'gpu_idle_threshold_pct':2,
    'gpu_sample_count':samples,
    'gpu_idle_sample_count':idle,
    'gpu_wait_fraction': (idle / samples) if samples else None,
    'peak_rss_gib': peak / (1024 ** 3) if peak else None,
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
set +e
timeout --signal=TERM --kill-after=30s 1800s env DOCKER_CONTAINER_NAME='$CONTAINER' /data-1/verl07/run_train.sh env \
  QWEN3_1P7B_MODEL_PATH='${QWEN3_1P7B_MODEL_PATH:?}' \
  CALIBRATION_VAL_FILE='${CALIBRATION_VAL_FILE:?}' CALIBRATION_OUTPUT_ROOT='$ROOT/${PHASE}' \
  CALIBRATION_STAGE1_CKPT_DIR='${CALIBRATION_STAGE1_CKPT_DIR:-}' CALIBRATION_STAGE1_MODEL2='${CALIBRATION_STAGE1_MODEL2:-}' \
  CALIBRATION_STAGE1_RUN_PREFIX='${CALIBRATION_STAGE1_RUN_PREFIX:-}' CALIBRATION_STAGE1_HANDOFF_STEP='${CALIBRATION_STAGE1_HANDOFF_STEP:-}' \
  CALIBRATION_TRAIN_FILE='${CALIBRATION_TRAIN_FILE:-}' CALIBRATION_STAGE3_MODEL_PATH='${CALIBRATION_STAGE3_MODEL_PATH:-}' \
  bash /workspace/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_operational_calibration_phase.sh '$PHASE' >>'$LOG' 2>&1 &
command_pid=\$!
sample_resources "\$command_pid" & sampler_pid=\$!
controlled_termination=false
while kill -0 "\$command_pid" 2>/dev/null; do
  if validation_complete; then
    controlled_termination=true
    docker kill '$CONTAINER' >/dev/null 2>&1 || true
    kill -TERM "\$command_pid" 2>/dev/null || true
    break
  fi
  sleep 2
done
wait "\$command_pid"; rc=\$?
if [ "\$controlled_termination" = true ] && validation_complete; then rc=0; fi
wait "\$sampler_pid" || true
end=\$(date +%s)
printf '{"phase":"%s","returncode":%s,"elapsed_seconds":%s,"timed_out":%s,"controlled_termination_after_complete_validation":%s}\n' '$PHASE' "\$rc" "\$((end-start))" "\$([ \$rc -eq 124 ] && echo true || echo false)" "\$controlled_termination" >'$STATUS'
exit \$rc
EOF
chmod +x "$ROOT/${PHASE}.worker.sh"
tmux new-session -d -s "$SESSION" "bash '$ROOT/${PHASE}.worker.sh'"
echo "started phase=$PHASE tmux=$SESSION container=$CONTAINER log=$LOG status=$STATUS"
