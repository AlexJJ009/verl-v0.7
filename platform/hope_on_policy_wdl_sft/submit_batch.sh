#!/usr/bin/env bash
# ==============================================================================
# Batch submitter for On-Policy WDL-SFT (LABELFIX rerun) on Meituan AFO via hope.
#
# Each iteration generates a per-experiment temp copy of run.hope + jupyter.sh,
# substitutes EXPERIMENT / app name / image / LGX / REPO_SUBPATH, and submits
# via `hope run run.hope` from the temp dir. The source template is never
# modified. Per-experiment stdout+stderr is captured to a log file.
#
# Default experiments: 1a 1b 1c. Override by positional args.
#
# Required env (one-time per shell):
#   HOPE_IMAGE        Docker image URL for [docker] afo.docker.image.name
#                     (skip with ALLOW_TEMPLATE_IMAGE=1 if the template image
#                     placeholder is already filled).
#
# Common optional env:
#   LGX               Dolphinfs anchor; default is template's value
#                     (yangfengkai02). Set to 陆晓东's lgx root.
#   REPO_SUBPATH      Repo path under LGX (default verl08/verl-v0.7-...)
#   USERGROUP         hadoop-ai-search override
#   SMOKE             1 → 10-step smoke; default unset (full run)
#   EXTRA_HOPE_ARGS   extra args passed verbatim to `hope run`
#   LOG_DIR_LOCAL     where submission logs land
#                     (default: <here>/submission_logs)
#   KEEP_TEMP         1 → keep temp submission dirs for inspection
#
# Flags:
#   --dry-run                Print substituted run.hope and exit (no hope call).
#   --continue-on-error      Keep submitting after a failure.
#   --all                    Use 1a 1b 1c 2a-base 2a-sft 2b-base 2b-sft 2c-base 2c-sft.
#   --4abc                   Use 4a 4b-math-base 4c-math-sft.
#   -h | --help              Show this help.
#
# Examples:
#   # Submit just the joint trio:
#   HOPE_IMAGE=<url> LGX=/mnt/.../lgx ./submit_batch.sh
#
#   # Submit every LABELFIX rerun:
#   HOPE_IMAGE=<url> LGX=/mnt/.../lgx ./submit_batch.sh --all
#
#   # Just 1A as a 10-step smoke:
#   SMOKE=1 HOPE_IMAGE=<url> LGX=/mnt/.../lgx ./submit_batch.sh 1a
#
#   # Dry-run to inspect the rendered run.hope:
#   HOPE_IMAGE=<url> ./submit_batch.sh --dry-run 1a
# ==============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_DIR="$HERE"

# --- Defaults ----------------------------------------------------------------
DEFAULT_EXPERIMENTS=(1a 1b 1c)
ALL_EXPERIMENTS=(1a 1b 1c 2a-base 2a-sft 2b-base 2b-sft 2c-base 2c-sft)
FOUR_ABC_EXPERIMENTS=(4a 4b-math-base 4c-math-sft)
KNOWN_EXPERIMENTS=(
    1a 1b 1c
    2a-base 2a-sft 2b-base 2b-sft 2c-base 2c-sft
    2z-base 2z-sft 2g-base 2g-sft
    4a 4a-dual 4a-model2-group-adv-is 4b-math-base 4c-math-sft
)

DRY_RUN=0
CONTINUE_ON_ERROR=0
EXPERIMENTS=()

# --- Arg parsing -------------------------------------------------------------
print_help() {
    sed -n '2,/^# ===*$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
        --all) EXPERIMENTS=("${ALL_EXPERIMENTS[@]}"); shift ;;
        --4abc) EXPERIMENTS=("${FOUR_ABC_EXPERIMENTS[@]}"); shift ;;
        -h|--help) print_help; exit 0 ;;
        --) shift; break ;;
        -*)
            echo "ERROR: unknown flag '$1'" >&2
            exit 2
            ;;
        *)
            EXPERIMENTS+=("$1"); shift
            ;;
    esac
done

if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
    EXPERIMENTS=("${DEFAULT_EXPERIMENTS[@]}")
fi

# --- Validate experiment names against dispatcher ---------------------------
is_known() {
    local needle="${1,,}"
    local exp
    for exp in "${KNOWN_EXPERIMENTS[@]}"; do
        if [ "$needle" = "$exp" ]; then return 0; fi
    done
    return 1
}
for exp in "${EXPERIMENTS[@]}"; do
    if ! is_known "$exp"; then
        echo "ERROR: experiment '$exp' is not recognized by jupyter.sh." >&2
        echo "       Allowed: ${KNOWN_EXPERIMENTS[*]}" >&2
        exit 2
    fi
done

# --- Required env ------------------------------------------------------------
if [ "$DRY_RUN" -ne 1 ] && [ "${ALLOW_TEMPLATE_IMAGE:-0}" != "1" ] && [ -z "${HOPE_IMAGE:-}" ]; then
    cat >&2 <<EOF
ERROR: HOPE_IMAGE is not set.
Set it to the verl-v2 docker image URL, e.g.
  export HOPE_IMAGE='registry-offlinebiz.sankuai.com/.../verl-...:tag'
or pass ALLOW_TEMPLATE_IMAGE=1 if the template's image placeholder is already
filled in for your account.
EOF
    exit 2
fi

# --- Tooling -----------------------------------------------------------------
if [ "$DRY_RUN" -ne 1 ] && ! command -v hope >/dev/null 2>&1; then
    echo "ERROR: 'hope' CLI not found in PATH. Run on a host with the hope client installed," >&2
    echo "       or use --dry-run to inspect the generated run.hope." >&2
    exit 2
fi

LOG_DIR_LOCAL="${LOG_DIR_LOCAL:-$HERE/submission_logs}"
mkdir -p "$LOG_DIR_LOCAL"
RUN_TS="$(date +%Y%m%d-%H%M%S)"

# --- Helpers -----------------------------------------------------------------
# In-place key replacement in INI-style run.hope. Matches '<key> = <oldval>'
# anywhere on a line and rewrites the value, preserving comments above.
ini_set() {
    local file="$1" key="$2" value="$3"
    # Use python for robust key=value rewriting (handles dots in key, no regex
    # quoting headaches, idempotent).
    python3 - "$file" "$key" "$value" <<'PY'
import re, sys
fpath, key, value = sys.argv[1], sys.argv[2], sys.argv[3]
with open(fpath) as f:
    lines = f.readlines()
pat = re.compile(r'^(\s*)' + re.escape(key) + r'(\s*=\s*).*$')
hit = False
for i, line in enumerate(lines):
    m = pat.match(line)
    if m:
        lines[i] = f"{m.group(1)}{key}{m.group(2)}{value}\n"
        hit = True
        break
if not hit:
    # Append under [others] if section exists, else at EOF.
    out = []
    inserted = False
    for line in lines:
        out.append(line)
        if not inserted and line.strip() == '[others]':
            out.append(f"{key} = {value}\n")
            inserted = True
    if not inserted:
        out.append(f"\n{key} = {value}\n")
    lines = out
with open(fpath, 'w') as f:
    f.writelines(lines)
PY
}

extract_job_id() {
    # Best-effort extraction. AFO/hope job ids tend to look like application_*
    # or are reported as 'job_id: ...' / 'application id: ...'. Print all
    # candidates so the user can pick.
    local logfile="$1"
    local ids
    ids=$(grep -Eoi 'application_[0-9]+_[0-9]+|job[_ ]?id[: ]+\S+|app[_ ]?id[: ]+\S+' "$logfile" 2>/dev/null | sort -u || true)
    if [ -n "$ids" ]; then
        printf '%s\n' "$ids"
    fi
}

submit_one() {
    local exp="$1"
    local exp_lc="${exp,,}"
    local app_name="verl-wdl-${exp_lc}"
    if [ "${SMOKE:-0}" = "1" ]; then
        app_name="${app_name}-smoke"
    fi

    local stage_dir
    stage_dir="$(mktemp -d -t "hope_${exp_lc}_XXXXXX")"

    # Stage template files. Keep filenames identical so worker.script resolves.
    cp "$TEMPLATE_DIR/run.hope"   "$stage_dir/run.hope"
    cp "$TEMPLATE_DIR/jupyter.sh" "$stage_dir/jupyter.sh"

    # --- Substitute fields in run.hope --------------------------------------
    ini_set "$stage_dir/run.hope" "afo.app.name"           "$app_name"
    ini_set "$stage_dir/run.hope" "afo.app.env.EXPERIMENT" "$exp_lc"
    if [ -n "${HOPE_IMAGE:-}" ]; then
        ini_set "$stage_dir/run.hope" "afo.docker.image.name" "$HOPE_IMAGE"
    fi
    if [ -n "${LGX:-}" ]; then
        ini_set "$stage_dir/run.hope" "afo.app.env.LGX" "$LGX"
    fi
    if [ -n "${REPO_SUBPATH:-}" ]; then
        ini_set "$stage_dir/run.hope" "afo.app.env.REPO_SUBPATH" "$REPO_SUBPATH"
    fi
    if [ -n "${USERGROUP:-}" ]; then
        ini_set "$stage_dir/run.hope" "usergroup" "$USERGROUP"
    fi
    if [ "${SMOKE:-0}" = "1" ]; then
        ini_set "$stage_dir/run.hope" "afo.app.env.SMOKE" "1"
    fi

    # --- Resolved values (for the print summary) ----------------------------
    local img lgx_val repo_sub usergroup
    img=$(grep -E '^afo.docker.image.name' "$stage_dir/run.hope" | head -1 | sed 's/^[^=]*=\s*//')
    lgx_val=$(grep -E '^afo.app.env.LGX' "$stage_dir/run.hope" | head -1 | sed 's/^[^=]*=\s*//')
    repo_sub=$(grep -E '^afo.app.env.REPO_SUBPATH' "$stage_dir/run.hope" | head -1 | sed 's/^[^=]*=\s*//')
    usergroup=$(grep -E '^usergroup' "$stage_dir/run.hope" | head -1 | sed 's/^[^=]*=\s*//')

    cat <<EOF
=================================================================
[batch] Experiment      : $exp_lc
[batch] App name        : $app_name
[batch] Image           : $img
[batch] Usergroup       : $usergroup
[batch] LGX             : $lgx_val
[batch] REPO_SUBPATH    : $repo_sub
[batch] SMOKE           : ${SMOKE:-0}
[batch] Stage dir       : $stage_dir
EOF

    # --- Sanity: bail if any REPLACE_ME / placeholder remains ---------------
    local has_placeholder=0
    if grep -qE 'REPLACE_ME|REPLACE_WITH_YOUR_V2_IMAGE_URL' "$stage_dir/run.hope"; then
        has_placeholder=1
        if [ "$DRY_RUN" = 1 ]; then
            echo "[batch] WARN (dry-run): run.hope still contains placeholders:" >&2
            grep -nE 'REPLACE_ME|REPLACE_WITH_YOUR_V2_IMAGE_URL' "$stage_dir/run.hope" >&2 || true
        else
            echo "[batch] ERROR: run.hope still contains a placeholder. Refusing to submit:" >&2
            grep -nE 'REPLACE_ME|REPLACE_WITH_YOUR_V2_IMAGE_URL' "$stage_dir/run.hope" >&2 || true
            return 1
        fi
    fi

    if [ "$DRY_RUN" = 1 ]; then
        echo "[batch] --dry-run: rendered run.hope:"
        echo "-----------------------------------------------------------------"
        cat "$stage_dir/run.hope"
        echo "-----------------------------------------------------------------"
        if [ "${KEEP_TEMP:-0}" != 1 ]; then
            rm -rf "$stage_dir"
        else
            echo "[batch] kept stage dir: $stage_dir"
        fi
        return 0
    fi

    # --- Submit -------------------------------------------------------------
    local log
    log="${LOG_DIR_LOCAL}/${RUN_TS}_${exp_lc}.log"
    echo "[batch] Log file        : $log"
    echo "[batch] Submitting via 'hope run run.hope'..."

    # Pipe both stdout and stderr to log; preserve hope's exit code.
    local rc=0
    (
        cd "$stage_dir"
        # shellcheck disable=SC2086
        hope run run.hope ${EXTRA_HOPE_ARGS:-}
    ) >"$log" 2>&1 || rc=$?

    local job_ids
    job_ids="$(extract_job_id "$log" || true)"
    if [ -n "$job_ids" ]; then
        echo "[batch] Detected ids:"
        printf '         %s\n' $job_ids
    else
        echo "[batch] No job id auto-detected. Inspect $log to grab it."
    fi

    if [ "${KEEP_TEMP:-0}" != 1 ]; then
        rm -rf "$stage_dir"
    else
        echo "[batch] kept stage dir: $stage_dir"
    fi

    if [ "$rc" -ne 0 ]; then
        echo "[batch] hope run exited with rc=$rc for $exp_lc. Tail of log:" >&2
        tail -n 40 "$log" >&2 || true
        return "$rc"
    fi
    return 0
}

# --- Main loop ---------------------------------------------------------------
echo "[batch] Experiments to submit: ${EXPERIMENTS[*]}"
echo "[batch] DRY_RUN=$DRY_RUN  CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR  SMOKE=${SMOKE:-0}"

declare -a OK=() FAIL=()
for exp in "${EXPERIMENTS[@]}"; do
    if submit_one "$exp"; then
        OK+=("$exp")
    else
        FAIL+=("$exp")
        if [ "$CONTINUE_ON_ERROR" -ne 1 ]; then
            echo "[batch] Aborting batch due to failure on $exp (use --continue-on-error to skip)." >&2
            break
        fi
    fi
done

echo "================== Submission Summary =================="
echo "[batch] OK   : ${OK[*]:-<none>}"
echo "[batch] FAIL : ${FAIL[*]:-<none>}"
if [ ${#FAIL[@]} -gt 0 ]; then
    exit 1
fi
