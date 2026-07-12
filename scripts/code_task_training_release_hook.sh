#!/usr/bin/env bash
# Release hook for successful code-task training runs.
set -euo pipefail

REPO=${REPO:-${VERL_REPO_ROOT:-/data-1/code/verl}}
REPO=$(readlink -f "$REPO")
RUN_NAME=${RUN_NAME:?RUN_NAME required}
RUN_PREFIX=${RUN_PREFIX:-${RUN_NAME%_*}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:?CHECKPOINT_DIR required}
METRICS_PATH=${METRICS_PATH:?METRICS_PATH required}
WANDB_PROJECT=${WANDB_PROJECT:-OnPolicyWDLSFT-CodeTask}
TRAINING_RELEASE_GATE_SCRIPT=${TRAINING_RELEASE_GATE_SCRIPT:-${REPO}/scripts/training_result_release_gate.py}
REGISTRY_IMPORT_SCRIPT=${REGISTRY_IMPORT_SCRIPT:-${REPO}/scripts/import_code_task_training_registry.py}
WANDB_SYNC_SCRIPT=${WANDB_SYNC_SCRIPT:-${REPO}/recipe/on_policy_wdl_sft/staged_v1/sync_wandb_offline.sh}
WANDB_RECOVERY_SYNC_SCRIPT=${WANDB_RECOVERY_SYNC_SCRIPT:-${REPO}/scripts/sync_wandb_skip_artifacts.py}
WANDB_SYNC_LAUNCHER=${WANDB_SYNC_LAUNCHER:-/data-1/verl07/run_train.sh}
WANDB_ROOT=${WANDB_ROOT:-/data-1/wandb_runs}
WANDB_ENTITY=${WANDB_ENTITY:-gongxunli-beihang-universally}
RELEASE_LOG_DIR=${RELEASE_LOG_DIR:-${REPO}/recipe/on_policy_wdl_sft/code_task/release_logs}
SKIP_DB_IMPORT=${SKIP_DB_IMPORT:-0}
SKIP_WANDB_SYNC=${SKIP_WANDB_SYNC:-0}
RELEASE_HOOK_STRICT=${RELEASE_HOOK_STRICT:-1}
REGISTRY_DB=${REGISTRY_DB:-${EXPERIMENT_REGISTRY_DB:-/data-1/experiment_registry/experiment_registry.sqlite}}

mkdir -p "$RELEASE_LOG_DIR"
LOG_FILE=${RELEASE_LOG_FILE:-${RELEASE_LOG_DIR}/${RUN_NAME}.release.log}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

run_step() {
    local name="$1"
    shift
    log "${name}: start"
    if "$@" >>"$LOG_FILE" 2>&1; then
        log "${name}: complete"
        return 0
    fi
    local rc=$?
    log "WARNING ${name}: failed rc=${rc}"
    if [ "$RELEASE_HOOK_STRICT" = "1" ]; then
        return "$rc"
    fi
    return 0
}

verify_registry_import() {
    python3 - "$RUN_NAME" "$REGISTRY_DB" <<'PY'
import re
import sqlite3
import sys

run_name, db = sys.argv[1], sys.argv[2]
key = "verl.code_task.training." + re.sub(r"[^a-zA-Z0-9]+", "_", run_name).strip("_").lower()
conn = sqlite3.connect(db)
row = conn.execute(
    "select id, experiment_id, total_steps from training_runs where training_run_key=?",
    (key,),
).fetchone()
if row is None:
    raise SystemExit(f"registry row missing for {key}")
print(f"registry-import verified: training_run_id={row[0]} experiment_id={row[1]} total_steps={row[2]}")
PY
}

find_wandb_file() {
    find "$1" -maxdepth 1 -type f -name 'run-*.wandb' 2>/dev/null | sort | tail -1
}

find_wandb_synced_marker() {
    find "$1" -maxdepth 1 -type f \( -name '*.wandb.synced' -o -name '.synced' \) 2>/dev/null | sort | tail -1
}

verify_wandb_sync() {
    local sync_dir="$1"
    local marker
    marker=$(find_wandb_synced_marker "$sync_dir")
    if [ -z "$marker" ]; then
        echo "W&B sync marker missing under ${sync_dir}" >&2
        return 1
    fi
    echo "wandb-sync verified: marker=${marker}"
}

recover_wandb_sync_without_artifacts() {
    local sync_dir="$1"
    if [ ! -f "$WANDB_RECOVERY_SYNC_SCRIPT" ]; then
        echo "W&B recovery sync script not found: ${WANDB_RECOVERY_SYNC_SCRIPT}" >&2
        return 1
    fi
    local wandb_file run_id
    wandb_file=$(find_wandb_file "$sync_dir")
    if [ -z "$wandb_file" ]; then
        echo "No run-*.wandb file found under ${sync_dir}" >&2
        return 1
    fi
    run_id=$(basename "$wandb_file")
    run_id=${run_id#run-}
    run_id=${run_id%.wandb}
    "$WANDB_SYNC_LAUNCHER" /opt/venv/bin/python "$WANDB_RECOVERY_SYNC_SCRIPT" "$sync_dir" \
        --entity "$WANDB_ENTITY" \
        --project "$WANDB_PROJECT" \
        --run-id "$run_id" \
        --release-gate-run-name "$RUN_NAME" \
        --release-gate-script "$TRAINING_RELEASE_GATE_SCRIPT" \
        --mark-synced
}

find_wandb_dir() {
    local root="${WANDB_ROOT}/${RUN_PREFIX}/wandb"
    if [ -d "$root" ]; then
        find "$root" -maxdepth 1 -type d -name 'offline-run-*' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
        return
    fi
    return 0
}

log "release hook start run=${RUN_NAME} prefix=${RUN_PREFIX}"
python3 "$TRAINING_RELEASE_GATE_SCRIPT" check --run-name "$RUN_NAME" >>"$LOG_FILE" 2>&1

if [ "$SKIP_DB_IMPORT" != "1" ]; then
    import_args=(
        env VERL_REPO_ROOT="$REPO" EXPERIMENT_REGISTRY_DB="$REGISTRY_DB" WANDB_ROOT="$WANDB_ROOT" \
        python3 "$REGISTRY_IMPORT_SCRIPT"
        --run-name "$RUN_NAME" \
        --run-prefix "$RUN_PREFIX" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --metrics-path "$METRICS_PATH"
    )
    if [ -n "${FINAL_STEP:-}" ]; then
        import_args+=(--final-step "$FINAL_STEP")
    fi
    if [ -n "${TRAIN_FILE:-}" ]; then
        import_args+=(--train-file "$TRAIN_FILE")
    fi
    run_step "registry-import" "${import_args[@]}"
    run_step "registry-verify" verify_registry_import
else
    log "registry-import: skipped by SKIP_DB_IMPORT=1"
fi

if [ "$SKIP_WANDB_SYNC" != "1" ]; then
    WANDB_SYNC_DIR=${WANDB_SYNC_DIR:-$(find_wandb_dir || true)}
    if [ -z "${WANDB_SYNC_DIR:-}" ] || [ ! -d "$WANDB_SYNC_DIR" ]; then
        log "WARNING wandb-sync: offline run dir not found for RUN_PREFIX=${RUN_PREFIX}"
        if [ "$RELEASE_HOOK_STRICT" = "1" ]; then
            exit 1
        fi
    else
        log "wandb-sync: start"
        if env WANDB_PROJECT="$WANDB_PROJECT" WANDB_ENTITY="$WANDB_ENTITY" \
            RELEASE_GATE_RUN_NAME="$RUN_NAME" TRAINING_RELEASE_GATE_SCRIPT="$TRAINING_RELEASE_GATE_SCRIPT" \
            MARK_SYNCED=true "$WANDB_SYNC_LAUNCHER" bash "$WANDB_SYNC_SCRIPT" "$WANDB_SYNC_DIR" >>"$LOG_FILE" 2>&1; then
            log "wandb-sync: complete"
        else
            rc=$?
            log "WARNING wandb-sync: normal sync failed rc=${rc}; trying artifact-free recovery"
        fi
        if ! verify_wandb_sync "$WANDB_SYNC_DIR" >>"$LOG_FILE" 2>&1; then
            log "wandb-sync: synced marker missing after normal sync; trying artifact-free recovery"
            run_step "wandb-sync-recovery" recover_wandb_sync_without_artifacts "$WANDB_SYNC_DIR"
            run_step "wandb-sync-verify" verify_wandb_sync "$WANDB_SYNC_DIR"
        else
            log "wandb-sync-verify: complete"
        fi
    fi
else
    log "wandb-sync: skipped by SKIP_WANDB_SYNC=1"
fi

log "release hook complete run=${RUN_NAME}"
