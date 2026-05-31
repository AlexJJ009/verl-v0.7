#!/usr/bin/env bash
# Render and submit staged v1 Meituan jobs. Default: Stage 1 beta search.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_EXPERIMENTS=(s1-beta-0 s1-beta-01 s1-beta-02 s1-beta-03 s1-beta-04 s1-beta-05 s1-beta-06 s1-beta-07 s1-beta-08 s1-beta-09 s1-beta-10)
EXPERIMENTS=()
DRY_RUN=0
CONTINUE_ON_ERROR=0

while [ $# -gt 0 ]; do
    case "$1" in
        --stage1|--stage1-grid) EXPERIMENTS=("${DEFAULT_EXPERIMENTS[@]}"); shift ;;
        --stage1-base) EXPERIMENTS=(s1-base-sft); shift ;;
        --all) EXPERIMENTS=("${DEFAULT_EXPERIMENTS[@]}"); shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
        -h|--help)
            sed -n '1,80p' "$0"
            exit 0
            ;;
        *) EXPERIMENTS+=("$1"); shift ;;
    esac
done

if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
    EXPERIMENTS=("${DEFAULT_EXPERIMENTS[@]}")
fi

if [ "$DRY_RUN" -ne 1 ] && [ "${ALLOW_TEMPLATE_IMAGE:-0}" != "1" ] && [ -z "${HOPE_IMAGE:-}" ]; then
    echo "ERROR: HOPE_IMAGE is not set. Export HOPE_IMAGE or set ALLOW_TEMPLATE_IMAGE=1." >&2
    exit 2
fi
if [ "$DRY_RUN" -ne 1 ] && ! command -v hope >/dev/null 2>&1; then
    echo "ERROR: hope CLI not found; use --dry-run to inspect rendered configs." >&2
    exit 2
fi

ini_set() {
    local file="$1" key="$2" value="$3"
    python3 - "$file" "$key" "$value" <<'PY'
import re, sys
fpath, key, value = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(fpath).readlines()
pat = re.compile(r'^(\s*)' + re.escape(key) + r'(\s*=\s*).*$')
for i, line in enumerate(lines):
    m = pat.match(line)
    if m:
        lines[i] = f"{m.group(1)}{key}{m.group(2)}{value}\n"
        break
else:
    lines.append(f"{key} = {value}\n")
open(fpath, "w").writelines(lines)
PY
}

submit_one() {
    local exp="$1"
    local exp_lc="${exp,,}"
    local stage_dir log_file
    stage_dir="$(mktemp -d -t "hope_staged_v1_${exp_lc}_XXXXXX")"
    log_file="${LOG_DIR_LOCAL:-$HERE/submission_logs}/${exp_lc}_$(date +%Y%m%d-%H%M%S).log"
    mkdir -p "$(dirname "$log_file")"

    cp "$HERE/run.hope" "$stage_dir/run.hope"
    cp "$HERE/jupyter.sh" "$stage_dir/jupyter.sh"
    ini_set "$stage_dir/run.hope" "afo.app.name" "verl-staged-v1-${exp_lc}"
    ini_set "$stage_dir/run.hope" "afo.app.env.EXPERIMENT" "$exp_lc"
    [ -z "${HOPE_IMAGE:-}" ] || ini_set "$stage_dir/run.hope" "afo.docker.image.name" "$HOPE_IMAGE"
    [ -z "${LGX:-}" ] || ini_set "$stage_dir/run.hope" "afo.app.env.LGX" "$LGX"
    [ -z "${REPO_SUBPATH:-}" ] || ini_set "$stage_dir/run.hope" "afo.app.env.REPO_SUBPATH" "$REPO_SUBPATH"
    [ -z "${SMOKE:-}" ] || ini_set "$stage_dir/run.hope" "afo.app.env.SMOKE" "$SMOKE"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "===== $exp_lc: $stage_dir/run.hope ====="
        cat "$stage_dir/run.hope"
        return 0
    fi

    echo "submitting $exp_lc; log=$log_file"
    (cd "$stage_dir" && hope run run.hope) 2>&1 | tee "$log_file"
}

for exp in "${EXPERIMENTS[@]}"; do
    if ! submit_one "$exp"; then
        [ "$CONTINUE_ON_ERROR" -eq 1 ] || exit 1
    fi
done
