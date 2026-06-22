#!/usr/bin/env bash
# Generic sequential training queue monitor helpers.
set -euo pipefail

training_queue_monitor_main() {
    : "${MONITOR_NAME:?MONITOR_NAME required}"
    : "${QUEUE_TMUX:?QUEUE_TMUX required}"
    : "${CKPT_ROOT:?CKPT_ROOT required}"
    : "${METRICS_ROOT:?METRICS_ROOT required}"
    : "${EXTRA_METRICS_ROOTS:=}"
    : "${WANDB_PROJECT:?WANDB_PROJECT required}"
    : "${POLL_SEC:=300}"
    : "${LOG_FILE:?LOG_FILE required}"
    : "${WXPUSHER_NOTIFY:=0}"
    : "${WXPUSHER_SCRIPT:=/root/agent-core/skills/wxpusher-notify/scripts/wxpusher_notify.py}"

    notify_monitor() {
        local title="$1" body="$2"
        if [ "$WXPUSHER_NOTIFY" != "1" ]; then
            return
        fi
        if [ ! -f "$WXPUSHER_SCRIPT" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: WxPusher script not found: $WXPUSHER_SCRIPT" | tee -a "$LOG_FILE"
            return
        fi
        if ! python3 "$WXPUSHER_SCRIPT" --title "$title" --body "$body" >>"$LOG_FILE" 2>&1; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: WxPusher notification failed: $title" | tee -a "$LOG_FILE"
        fi
    }

    mkdir -p "$(dirname "$LOG_FILE")"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${MONITOR_NAME} monitor start; queue=${QUEUE_TMUX}" | tee -a "$LOG_FILE"
    notify_monitor "${MONITOR_NAME} monitor started" "Status: started
What happened: ${MONITOR_NAME} monitor started.
Evidence: queue=${QUEUE_TMUX}; prefixes=${RUN_PREFIXES[*]}
Next action: Monitor will notify on launch, completion, failure, or queue exit."
    declare -A notified_launched
    declare -A notified_complete
    for prefix in "${RUN_PREFIXES[@]}"; do
        notified_launched["$prefix"]=0
        notified_complete["$prefix"]=0
    done

    while true; do
        local any_active=0
        if tmux has-session -t "$QUEUE_TMUX" 2>/dev/null; then
            any_active=1
        fi
        for idx in "${!RUN_PREFIXES[@]}"; do
            local prefix="${RUN_PREFIXES[$idx]}"
            local tmux_name="${TMUX_NAMES[$idx]}"
            local final_step="${FINAL_STEPS[$idx]}"
            local ckpt step metrics status
            ckpt=$(find "$CKPT_ROOT" -maxdepth 1 -type d -name "${prefix}_*" 2>/dev/null | sort | tail -1)
            step="none"
            status="missing"
            if [ -n "$ckpt" ]; then
                if [ -f "$ckpt/latest_checkpointed_iteration.txt" ]; then
                    step=$(tr -dc '0-9' < "$ckpt/latest_checkpointed_iteration.txt")
                else
                    step=$(find "$ckpt" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
                    step=${step:-none}
                fi
                metrics="${METRICS_ROOT}/${WANDB_PROJECT}/$(basename "$ckpt").jsonl"
                if [ ! -f "$metrics" ] && [ -n "$EXTRA_METRICS_ROOTS" ]; then
                    local root candidate
                    for root in $EXTRA_METRICS_ROOTS; do
                        candidate="${root}/${WANDB_PROJECT}/$(basename "$ckpt").jsonl"
                        if [ -f "$candidate" ]; then
                            metrics="$candidate"
                            break
                        fi
                    done
                fi
                if [ -f "$metrics" ]; then
                    status="metrics-present"
                    if [ "$step" != "none" ] && [ "$step" -ge "$final_step" ]; then
                        status="checkpoint-final"
                    fi
                else
                    status="checkpoint-only"
                fi
            fi
            if tmux has-session -t "$tmux_name" 2>/dev/null; then
                any_active=1
                if [ "${notified_launched[$prefix]}" = "0" ]; then
                    notified_launched["$prefix"]=1
                    notify_monitor "${MONITOR_NAME} run launched" "Status: started
What happened: ${prefix} has an active tmux session.
Evidence: tmux=${tmux_name}; step=${step}; checkpoint=${ckpt:-none}
Next action: Monitor will notify on completion or early stop."
                fi
            fi
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${prefix}: step=${step} final=${final_step} status=${status} ckpt=${ckpt:-none}" | tee -a "$LOG_FILE"
            if [ "$status" = "checkpoint-final" ] && [ "${notified_complete[$prefix]}" = "0" ]; then
                notified_complete["$prefix"]=1
                notify_monitor "${MONITOR_NAME} run complete" "Status: completed
What happened: ${prefix} reached final_step=${final_step}.
Evidence: step=${step}; checkpoint=${ckpt:-none}
Next action: Queue/monitor will continue or finish."
            fi
            if [ "${notified_launched[$prefix]}" = "1" ] \
                && [ "${notified_complete[$prefix]}" = "0" ] \
                && [ "$status" != "missing" ] \
                && ! tmux has-session -t "$tmux_name" 2>/dev/null \
                && ! tmux has-session -t "$QUEUE_TMUX" 2>/dev/null; then
                notify_monitor "${MONITOR_NAME} run stopped" "Status: failed
What happened: ${prefix} is no longer active and did not reach final_step=${final_step}.
Evidence: step=${step}; status=${status}; checkpoint=${ckpt:-none}
Next action: Monitor Agent should classify logs using the runbook before relaunching or changing parameters."
            fi
        done
        if [ "$any_active" = "0" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${MONITOR_NAME} monitor exit: no queue/training tmux active" | tee -a "$LOG_FILE"
            local incomplete_launched=()
            for prefix in "${RUN_PREFIXES[@]}"; do
                if [ "${notified_launched[$prefix]}" = "1" ] && [ "${notified_complete[$prefix]}" = "0" ]; then
                    incomplete_launched+=("$prefix")
                fi
            done
            if [ "${#incomplete_launched[@]}" -gt 0 ]; then
                notify_monitor "${MONITOR_NAME} monitor exited with incomplete runs" "Status: needs review
What happened: ${MONITOR_NAME} monitor exited while at least one launched run did not reach its final checkpoint.
Evidence: incomplete=${incomplete_launched[*]}; queue=${QUEUE_TMUX}; log=${LOG_FILE}
Next action: Monitor Agent should inspect logs and checkpoints before relaunching or changing parameters."
            else
                notify_monitor "${MONITOR_NAME} monitor exited" "Status: completed
What happened: ${MONITOR_NAME} monitor exited because no queue or training tmux session is active.
Evidence: queue=${QUEUE_TMUX}; log=${LOG_FILE}
Next action: Review whether every intended prefix reached final step."
            fi
            return 0
        fi
        sleep "$POLL_SEC"
    done
}
