#!/usr/bin/env bash
set -euo pipefail

PUEUE_VERSION="4.0.4"
PUEUE_SHA256="c1b10d7e4e62211075ddd0e1dc3e8cbfc5a43d662cb3be7402a28504e23fcb51"
PUEUED_SHA256="5afeff6adbafb909e8d54e2caff158e6966c2adffa2c09e60fd631cc51b60390"
OPERATION_ROOT="${VERL_A800_OPERATION_ROOT:-/data_storage/yl_test/lgx}"
BIN_DIR="${VERL_A800_BIN_DIR:-${OPERATION_ROOT}/home/.local/bin}"
PUEUE_ROOT="${VERL_A800_PUEUE_ROOT:-${OPERATION_ROOT}/runtime/verl/pueue}"
CONFIG_PATH="${PUEUE_CONFIG_PATH:-${PUEUE_ROOT}/config/pueue.yml}"
DOWNLOAD_ROOT="${PUEUE_ROOT}/downloads"

case "${OPERATION_ROOT}" in
  /data_storage/yl_test/lgx|/data_storage/yl_test/lgx/*) ;;
  *)
    echo "error: operation root must stay below /data_storage/yl_test/lgx: ${OPERATION_ROOT}" >&2
    exit 2
    ;;
esac

for path in "${BIN_DIR}" "${PUEUE_ROOT}" "${CONFIG_PATH}"; do
  case "${path}" in
    /data_storage/yl_test/lgx/*) ;;
    *)
      echo "error: mutable Pueue path escapes the A800 operation root: ${path}" >&2
      exit 2
      ;;
  esac
done

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "error: this bootstrap pins the Linux x86_64 Pueue release" >&2
  exit 2
fi

install_release_binary() {
  local name=$1
  local expected_sha=$2
  local target="${BIN_DIR}/${name}"
  local asset="${name}-x86_64-unknown-linux-musl"
  local url="https://github.com/Nukesor/pueue/releases/download/v${PUEUE_VERSION}/${asset}"
  local tmp

  if [[ -x "${target}" ]] && printf '%s  %s\n' "${expected_sha}" "${target}" | sha256sum --check --status; then
    return
  fi

  tmp=$(mktemp "${DOWNLOAD_ROOT}/${asset}.XXXXXX")
  if ! curl --fail --location --retry 5 --retry-all-errors --retry-delay 2 \
    --output "${tmp}" "${url}"; then
    rm -f -- "${tmp}"
    echo "error: failed to download ${asset}" >&2
    return 4
  fi
  if ! printf '%s  %s\n' "${expected_sha}" "${tmp}" | sha256sum --check --status; then
    rm -f -- "${tmp}"
    echo "error: checksum mismatch for ${asset}" >&2
    return 3
  fi
  install -m 0755 "${tmp}" "${target}"
  rm -f -- "${tmp}"
}

mkdir -p -- "${BIN_DIR}" "${DOWNLOAD_ROOT}" "${PUEUE_ROOT}/config" \
  "${PUEUE_ROOT}/state" "${PUEUE_ROOT}/runtime"

install_release_binary pueue "${PUEUE_SHA256}"
install_release_binary pueued "${PUEUED_SHA256}"

config_tmp=$(mktemp "${PUEUE_ROOT}/config/pueue.yml.XXXXXX")
trap 'rm -f -- "${config_tmp:-}"' EXIT
cat >"${config_tmp}" <<EOF
shared:
  pueue_directory: "${PUEUE_ROOT}/state"
  runtime_directory: "${PUEUE_ROOT}/runtime"
  use_unix_socket: true
  unix_socket_path: "${PUEUE_ROOT}/runtime/pueue.sock"
  pid_path: "${PUEUE_ROOT}/runtime/pueue.pid"
daemon:
  pause_group_on_failure: false
  pause_all_on_failure: false
client:
  read_local_logs: true
  show_confirmation_questions: false
EOF
chmod 0600 "${config_tmp}"
mv -f -- "${config_tmp}" "${CONFIG_PATH}"
trap - EXIT

export PUEUE_CONFIG_PATH="${CONFIG_PATH}"
if ! "${BIN_DIR}/pueue" status >/dev/null 2>&1; then
  daemon_session="verl-a800-pueued"
  if tmux has-session -t "=${daemon_session}" 2>/dev/null; then
    tmux kill-session -t "=${daemon_session}"
  fi
  printf -v daemon_command 'PUEUE_CONFIG_PATH=%q exec %q >>%q 2>&1' \
    "${CONFIG_PATH}" "${BIN_DIR}/pueued" "${PUEUE_ROOT}/pueued.log"
  tmux new-session -d -s "${daemon_session}" "${daemon_command}"
fi

for _ in $(seq 1 50); do
  if "${BIN_DIR}/pueue" status >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
"${BIN_DIR}/pueue" status >/dev/null

if ! "${BIN_DIR}/pueue" group --json | python3 -c \
  'import json,sys; raise SystemExit(0 if "gpu8" in json.load(sys.stdin) else 1)'; then
  "${BIN_DIR}/pueue" group add gpu8 --parallel 1
fi
"${BIN_DIR}/pueue" parallel --group gpu8 1 >/dev/null

env_tmp=$(mktemp "${PUEUE_ROOT}/pueue.env.XXXXXX")
trap 'rm -f -- "${env_tmp:-}"' EXIT
printf 'export PATH=%q:$PATH\nexport PUEUE_CONFIG_PATH=%q\n' \
  "${BIN_DIR}" "${CONFIG_PATH}" >"${env_tmp}"
chmod 0600 "${env_tmp}"
mv -f -- "${env_tmp}" "${PUEUE_ROOT}/pueue.env"
trap - EXIT

printf 'Pueue v%s ready\n' "${PUEUE_VERSION}"
printf 'Environment: %s\n' "${PUEUE_ROOT}/pueue.env"
printf 'Config: %s\n' "${CONFIG_PATH}"
"${BIN_DIR}/pueue" group --json
