#!/usr/bin/env bash
set -euo pipefail

repo_host=/data-1/code/verl
image=verl-harness:latest
output=
while (($#)); do
    case "$1" in
        --repo-host) repo_host=$2; shift 2 ;;
        --image) image=$2; shift 2 ;;
        --output) output=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "$output" ] || { echo "--output is required" >&2; exit 2; }

mkdir -p "$(dirname "$output")"
tmp="${output}.tmp.$$"
trap 'rm -f "$tmp"' EXIT

REPO_HOST="$repo_host" IMAGE="$image" OUTPUT="$tmp" python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess


def run(*args: str) -> tuple[int, str, str]:
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


repo_host = os.environ["REPO_HOST"]
image = os.environ["IMAGE"]
checks = []
generated_at = now()

docker_available = shutil.which("docker") is not None
image_id = ""
docker_detail = "docker executable unavailable"
if docker_available:
    code, stdout, stderr = run("docker", "image", "inspect", image, "--format", "{{.Id}}")
    image_id = stdout if code == 0 else ""
    docker_detail = {"returncode": code, "stdout": stdout, "stderr": stderr}
checks.append({"name": "docker_image", "ok": bool(image_id), "detail": docker_detail})

tmux_code, tmux_stdout, tmux_stderr = run("tmux", "list-sessions", "-F", "#{session_name}")
sessions = [line for line in tmux_stdout.splitlines() if line]
conflicts = [name for name in sessions if name.startswith("stage123_") or name == "code_task_qwen3_1p7b_stage123_queue"]
checks.append({"name": "tmux_snapshot", "ok": tmux_code in (0, 1), "detail": {"sessions": sessions, "stderr": tmux_stderr}})

mount_target = str(Path("/data-1/checkpoints").resolve())
df_code, df_stdout, df_stderr = run("df", "-P", "/data-1", "/data-2")
checks.append({"name": "checkpoint_mount", "ok": mount_target == "/data-2/checkpoints", "detail": mount_target})
checks.append({"name": "storage", "ok": df_code == 0, "detail": {"stdout": df_stdout, "stderr": df_stderr}})

payload = {
    "schema_version": 1,
    "artifact_type": "stage123_host_facts",
    "generated_at": generated_at,
    "completed_at": now(),
    "repo_host": repo_host,
    "docker_image": {"reference": image, "immutable_id": image_id},
    "tmux": {"sessions": sessions, "stage123_conflicts": conflicts},
    "mounts": {"checkpoint_mount": mount_target},
    "storage": {"df": df_stdout},
    "checks": checks,
    "producer": {"command": "scripts/stage123_host_facts.sh", "host_owned": True},
    "ok": all(item["ok"] for item in checks),
}
Path(os.environ["OUTPUT"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
mv -f "$tmp" "$output"
