#!/usr/bin/env bash
set -euo pipefail
event=${1:?event required}
body=${2:?body required}
case "$event" in
    run_started) title="Stage123 training started" ;;
    run_failed) title="Stage123 training failed" ;;
    user_decision_required) title="Stage123 needs your decision" ;;
    *) echo "unsupported notification event: $event" >&2; exit 2 ;;
esac
exec python3 "${WXPUSHER_SCRIPT:-/root/agent-core/skills/wxpusher-notify/scripts/wxpusher_notify.py}" \
    --title "$title" --body "$body"
