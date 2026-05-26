#!/usr/bin/env bash
# Generic tmux queue monitor for sequential local training runs.
#
# Usage pattern:
#   RUN_PREFIXES=(...)
#   RUN_SCRIPTS=(...)
#   TMUX_NAMES=(...)
#   source scripts/training_queue_monitor.sh
#   training_queue_monitor_main
#
# Or execute directly with newline-separated env vars:
#   QUEUE_RUN_PREFIXES=$'prefix_a\nprefix_b' \
#   QUEUE_RUN_SCRIPTS=$'/workspace/verl/run_a.sh\n/workspace/verl/run_b.sh' \
#   QUEUE_TMUX_NAMES=$'run_a\nrun_b' \
#   bash scripts/training_queue_monitor.sh

set -euo pipefail

training_queue_monitor_log() {
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

training_queue_monitor_get_df_target() {
    local path="$1"
    while [ ! -e "$path" ] && [ "$path" != "/" ]; do
        path=$(dirname "$path")
    done
    printf '%s\n' "$path"
}

training_queue_monitor_get_free_gb() {
    local target
    target=$(training_queue_monitor_get_df_target "$CKPT_ROOT")
    df -BG "$target" | awk 'NR==2 {sub("G","",$4); print $4}'
}

training_queue_monitor_get_gpu_util_total() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | awk '{s+=$1} END {print s+0}'
}

training_queue_monitor_host_script_path() {
    local script="$1"
    printf '%s\n' "${script/#$REPO_CONTAINER/$REPO_HOST}"
}

training_queue_monitor_latest_ckpt_dir() {
    local prefix="$1"
    find "$CKPT_ROOT" -maxdepth 1 -type d -name "${prefix}_*" 2>/dev/null | sort | tail -1
}

training_queue_monitor_latest_step() {
    local ckpt_dir="$1"
    if [ -f "$ckpt_dir/latest_checkpointed_iteration.txt" ]; then
        tr -dc '0-9' < "$ckpt_dir/latest_checkpointed_iteration.txt"
        return
    fi
    find "$ckpt_dir" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null \
        | sed 's/.*global_step_//' | sort -n | tail -1
}

training_queue_monitor_is_complete() {
    local prefix="$1"
    local ckpt_dir
    ckpt_dir=$(training_queue_monitor_latest_ckpt_dir "$prefix")
    [ -n "$ckpt_dir" ] || return 1
    [ -d "$ckpt_dir/global_step_${FINAL_STEP}" ] && return 0
    local step
    step=$(training_queue_monitor_latest_step "$ckpt_dir" || true)
    [ -n "$step" ] && [ "$step" -ge "$FINAL_STEP" ]
}

training_queue_monitor_wait_for_resources() {
    while true; do
        local free_gb gpu_util
        free_gb=$(training_queue_monitor_get_free_gb)
        gpu_util=$(training_queue_monitor_get_gpu_util_total)
        if [ "$free_gb" -ge "$MIN_FREE_GB" ] && [ "$gpu_util" -lt "$MAX_GPU_UTIL" ]; then
            training_queue_monitor_log "resources ok: ckpt_fs free=${free_gb}G, gpu_util_total=${gpu_util}"
            return
        fi
        training_queue_monitor_log "waiting resources: ckpt_fs free=${free_gb}G need>=${MIN_FREE_GB}G; gpu_util_total=${gpu_util} need<${MAX_GPU_UTIL}; sleep ${POLL_SEC}s"
        sleep "$POLL_SEC"
    done
}

training_queue_monitor_launch_run() {
    local script="$1"
    local tmux_name="$2"
    local host_script
    host_script=$(training_queue_monitor_host_script_path "$script")

    [ -f "$host_script" ] || { training_queue_monitor_log "ERROR: missing host script $host_script"; exit 1; }
    if tmux has-session -t "$tmux_name" 2>/dev/null; then
        training_queue_monitor_log "tmux session already exists: $tmux_name; monitoring existing session"
        return
    fi

    local launch_log="${LOG_FILE%.log}_${tmux_name}.log"
    training_queue_monitor_log "launching $script in tmux $tmux_name"
    if [ -x "$LAUNCHER" ]; then
        tmux new-session -d -s "$tmux_name" \
            "bash $LAUNCHER $script 2>&1 | tee -a $launch_log"
    else
        training_queue_monitor_log "launcher not found/executable ($LAUNCHER); falling back to docker image $DOCKER_IMAGE"
        tmux new-session -d -s "$tmux_name" \
            "docker run --rm --gpus all --ipc=host --shm-size=64g -v /data-1:/data-1 -v $REPO_HOST:$REPO_CONTAINER -w $REPO_CONTAINER $DOCKER_IMAGE bash $script 2>&1 | tee -a $launch_log"
    fi
    sleep 5
    tmux has-session -t "$tmux_name" 2>/dev/null || {
        training_queue_monitor_log "ERROR: tmux session failed to start: $tmux_name; see $launch_log"
        exit 1
    }
}

training_queue_monitor_wait_for_completion() {
    local prefix="$1"
    local tmux_name="$2"

    while true; do
        if training_queue_monitor_is_complete "$prefix"; then
            local ckpt_dir step
            ckpt_dir=$(training_queue_monitor_latest_ckpt_dir "$prefix")
            step=$(training_queue_monitor_latest_step "$ckpt_dir" || echo "$FINAL_STEP")
            training_queue_monitor_log "complete: prefix=$prefix ckpt_dir=$ckpt_dir latest_step=$step"
            return
        fi

        local ckpt_dir step tmux_state
        ckpt_dir=$(training_queue_monitor_latest_ckpt_dir "$prefix" || true)
        if [ -n "$ckpt_dir" ]; then
            step=$(training_queue_monitor_latest_step "$ckpt_dir" || echo "none")
        else
            step="none"
        fi
        if tmux has-session -t "$tmux_name" 2>/dev/null; then
            tmux_state="alive"
        else
            tmux_state="missing"
        fi

        if [ "$tmux_state" = "missing" ]; then
            training_queue_monitor_log "ERROR: tmux $tmux_name exited before final step; latest_step=$step, need=$FINAL_STEP"
            exit 1
        fi
        training_queue_monitor_log "waiting completion: prefix=$prefix tmux=$tmux_state latest_step=$step sleep=${POLL_SEC}s"
        sleep "$POLL_SEC"
    done
}

training_queue_monitor_parse_env_lists() {
    [ -n "${QUEUE_RUN_PREFIXES:-}" ] || return 0
    [ -n "${QUEUE_RUN_SCRIPTS:-}" ] || { echo "ERROR: QUEUE_RUN_SCRIPTS is required with QUEUE_RUN_PREFIXES" >&2; exit 1; }
    [ -n "${QUEUE_TMUX_NAMES:-}" ] || { echo "ERROR: QUEUE_TMUX_NAMES is required with QUEUE_RUN_PREFIXES" >&2; exit 1; }
    mapfile -t RUN_PREFIXES <<< "$QUEUE_RUN_PREFIXES"
    mapfile -t RUN_SCRIPTS <<< "$QUEUE_RUN_SCRIPTS"
    mapfile -t TMUX_NAMES <<< "$QUEUE_TMUX_NAMES"
}

training_queue_monitor_validate_queue() {
    [ "${#RUN_PREFIXES[@]}" -gt 0 ] || { echo "ERROR: RUN_PREFIXES is empty" >&2; exit 1; }
    [ "${#RUN_PREFIXES[@]}" -eq "${#RUN_SCRIPTS[@]}" ] || { echo "ERROR: RUN_PREFIXES and RUN_SCRIPTS length mismatch" >&2; exit 1; }
    [ "${#RUN_PREFIXES[@]}" -eq "${#TMUX_NAMES[@]}" ] || { echo "ERROR: RUN_PREFIXES and TMUX_NAMES length mismatch" >&2; exit 1; }
}

training_queue_monitor_main() {
    REPO_HOST=${REPO_HOST:-/data-1/verl07/verl}
    REPO_CONTAINER=${REPO_CONTAINER:-/workspace/verl}
    LAUNCHER=${LAUNCHER:-/data-1/verl07/run_train.sh}
    DOCKER_IMAGE=${DOCKER_IMAGE:-verl-harness}
    CKPT_ROOT=${CKPT_ROOT:-/data-1/checkpoints}
    MIN_FREE_GB=${MIN_FREE_GB:-160}
    MAX_GPU_UTIL=${MAX_GPU_UTIL:-50}
    FINAL_STEP=${FINAL_STEP:-115}
    POLL_SEC=${POLL_SEC:-300}
    LOG_FILE=${LOG_FILE:-"${REPO_HOST}/logs/training_queue_monitor.log"}

    training_queue_monitor_parse_env_lists
    if ! declare -p RUN_PREFIXES >/dev/null 2>&1; then RUN_PREFIXES=(); fi
    if ! declare -p RUN_SCRIPTS >/dev/null 2>&1; then RUN_SCRIPTS=(); fi
    if ! declare -p TMUX_NAMES >/dev/null 2>&1; then TMUX_NAMES=(); fi
    training_queue_monitor_validate_queue

    training_queue_monitor_log "=========================================================="
    training_queue_monitor_log "training queue monitor started (PID $$)"
    training_queue_monitor_log "REPO_HOST=$REPO_HOST"
    training_queue_monitor_log "REPO_CONTAINER=$REPO_CONTAINER"
    training_queue_monitor_log "CKPT_ROOT=$CKPT_ROOT"
    training_queue_monitor_log "FINAL_STEP=$FINAL_STEP"
    training_queue_monitor_log "MIN_FREE_GB=$MIN_FREE_GB"
    training_queue_monitor_log "MAX_GPU_UTIL=$MAX_GPU_UTIL"
    training_queue_monitor_log "LAUNCHER=$LAUNCHER"
    training_queue_monitor_log "DOCKER_IMAGE=$DOCKER_IMAGE"
    training_queue_monitor_log "queue_length=${#RUN_PREFIXES[@]}"
    training_queue_monitor_log "=========================================================="

    local idx
    for idx in "${!RUN_PREFIXES[@]}"; do
        local prefix script tmux_name
        prefix="${RUN_PREFIXES[$idx]}"
        script="${RUN_SCRIPTS[$idx]}"
        tmux_name="${TMUX_NAMES[$idx]}"

        if training_queue_monitor_is_complete "$prefix"; then
            training_queue_monitor_log "skip already complete: $prefix"
            continue
        fi

        training_queue_monitor_wait_for_resources
        training_queue_monitor_launch_run "$script" "$tmux_name"
        training_queue_monitor_wait_for_completion "$prefix" "$tmux_name"
    done

    training_queue_monitor_log "training queue complete."
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    training_queue_monitor_main "$@"
fi
