#!/usr/bin/env bash
set -euo pipefail

REPO_HOST=${REPO_HOST:-/data-1/verl07/verl}
REPO_CONTAINER=${REPO_CONTAINER:-/workspace/verl}
DOCKER_IMAGE=${DOCKER_IMAGE:-verl-harness:latest}
DATA1_HOST=${DATA1_HOST:-/data-1}
DATA2_HOST=${DATA2_HOST:-/data-2}
REPO_MOUNT_MODE=${REPO_MOUNT_MODE:-rw}

for data_root in "$DATA1_HOST" "$DATA2_HOST"; do
    test "${data_root#/}" != "$data_root"
    test -d "$data_root"
done
case "$REPO_MOUNT_MODE" in ro|rw) ;; *) exit 64 ;; esac

env_args=()
name_args=()
if [ -n "${DOCKER_CONTAINER_NAME:-}" ]; then
    name_args=(--name "$DOCKER_CONTAINER_NAME")
fi
while IFS='=' read -r name _; do
    case "$name" in
        HOME|HOSTNAME|PATH|PWD|PYTHONPATH|SHELL|SHLVL|USER|VIRTUAL_ENV|_) continue ;;
    esac
    env_args+=(--env "$name")
done < <(env)

exec docker run --rm --gpus all --ipc=host --network=host --shm-size=64g \
    "${name_args[@]}" \
    "${env_args[@]}" \
    -v "${DATA1_HOST}:/data-1" \
    -v "${DATA2_HOST}:/data-2" \
    -v "${REPO_HOST}:${REPO_CONTAINER}:${REPO_MOUNT_MODE}" \
    -w "${REPO_CONTAINER}" \
    "${DOCKER_IMAGE}" \
    "$@"
