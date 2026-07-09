#!/usr/bin/env bash
# Shared shell helpers for recording training release-gate events.

training_release_gate_record_event() {
    local monitor_name="$1" prefix="$2" status="$3" step="$4" final_step="$5" ckpt="$6" metrics="$7" notes="$8" log_file="$9"

    : "${TRAINING_RELEASE_GATE:=1}"
    : "${TRAINING_RELEASE_GATE_SCRIPT:=scripts/training_result_release_gate.py}"
    : "${TRAINING_RELEASE_GATE_STATE:=/data-1/experiment_registry/training_release_gate.jsonl}"

    if [ "$TRAINING_RELEASE_GATE" != "1" ]; then
        return 0
    fi
    if [ ! -f "$TRAINING_RELEASE_GATE_SCRIPT" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: release gate script not found: $TRAINING_RELEASE_GATE_SCRIPT" | tee -a "$log_file"
        return 1
    fi

    local run_name="$prefix"
    if [ -n "$ckpt" ]; then
        run_name="$(basename "$ckpt")"
    fi
    local cmd=(
        python3 "$TRAINING_RELEASE_GATE_SCRIPT"
        --state "$TRAINING_RELEASE_GATE_STATE"
        record
        --run-name "$run_name"
        --family "$prefix"
        --status "$status"
        --source "monitor:${monitor_name}"
        --notes "$notes"
    )
    if [ -n "$ckpt" ]; then
        cmd+=(--checkpoint "$ckpt")
    fi
    if [ -n "$metrics" ] && [ -f "$metrics" ]; then
        cmd+=(--metrics "$metrics")
    fi
    if [ -n "$final_step" ] && [ "$final_step" != "none" ]; then
        cmd+=(--final-step "$final_step")
    fi
    if [ -n "$step" ] && [ "$step" != "none" ]; then
        cmd+=(--observed-step "$step")
    fi

    if ! "${cmd[@]}" >>"$log_file" 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: release gate record failed for ${run_name} status=${status}" | tee -a "$log_file"
        return 1
    fi
    if [ "$status" = "success_complete" ]; then
        if ! python3 "$TRAINING_RELEASE_GATE_SCRIPT" --state "$TRAINING_RELEASE_GATE_STATE" check --run-name "$run_name" --family "$prefix" >>"$log_file" 2>&1; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: release gate check failed after success record for ${run_name}" | tee -a "$log_file"
            return 1
        fi
    fi
    return 0
}
