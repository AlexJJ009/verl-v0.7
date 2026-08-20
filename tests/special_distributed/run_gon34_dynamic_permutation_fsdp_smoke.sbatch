#!/usr/bin/env bash
# Version-controlled GON-34 engineering smoke. Submit only after staging the
# exact candidate read-only on both worker nodes.
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40S:8
#SBATCH --exclusive
#SBATCH --exclude=controller-dev
#SBATCH --time=00:30:00
#SBATCH --job-name=GON34-DYNPERM-FSDP-SMOKE
#SBATCH --nice=10000
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${GON34_CANDIDATE_SHA:?required}"
: "${GON34_STAGE_REL:?required}"
: "${GON34_OUTPUT_REL:?required}"
: "${GON34_NODE_ROOT_MAP:?required semicolon-separated node=/absolute/root entries}"

test "${SLURMD_NODENAME:?}" != controller-dev
NODE_ROOT=""
IFS=';' read -r -a node_roots <<<"${GON34_NODE_ROOT_MAP}"
for node_root in "${node_roots[@]}"; do
  if test "${node_root%%=*}" = "${SLURMD_NODENAME}"; then
    NODE_ROOT="${node_root#*=}"
    break
  fi
done
test -n "${NODE_ROOT}" && test "${NODE_ROOT#/}" != "${NODE_ROOT}"

WORKSPACE="${NODE_ROOT}/${GON34_STAGE_REL}"
OUTPUT="${NODE_ROOT}/${GON34_OUTPUT_REL}/${SLURM_JOB_ID}"
SCRATCH="${NODE_ROOT}/scratch/jobs/${SLURM_JOB_ID}-gon34-dynperm"
LOGS="${NODE_ROOT}/logs/jobs/${SLURM_JOB_ID}-gon34-dynperm"
CONTAINER_NAME="l40s-job-${SLURM_JOB_ID}-gon34-dynperm-smoke"

test -f "${WORKSPACE}/.candidate-sha"
test "$(tr -d '\n' < "${WORKSPACE}/.candidate-sha")" = "${GON34_CANDIDATE_SHA}"
test -z "$(find "${WORKSPACE}" \( -type f -o -type d \) -perm /222 -print -quit)"
mkdir -p "${OUTPUT}" "${SCRATCH}" "${LOGS}"
test -z "$(find "${OUTPUT}" -mindepth 1 -print -quit)"

IMAGE_ID="$(docker image inspect verl-harness:latest --format '{{.Id}}')"
printf '{"candidate_sha":"%s","job_id":"%s","node":"%s","workspace":"%s","image_id":"%s","formal_experiment":false}\n' \
  "${GON34_CANDIDATE_SHA}" "${SLURM_JOB_ID}" "${SLURMD_NODENAME}" "${WORKSPACE}" "${IMAGE_ID}" \
  >"${LOGS}/job-admission.json"

cleanup() {
  rc=$?
  trap - EXIT TERM INT
  docker rm --force "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  printf '{"job_id":"%s","node":"%s","exit_code":%s,"candidate_sha":"%s"}\n' \
    "${SLURM_JOB_ID}" "${SLURMD_NODENAME}" "${rc}" "${GON34_CANDIDATE_SHA}" >"${LOGS}/job-terminal.json"
  exit "${rc}"
}
trap cleanup EXIT TERM INT

docker run --rm --name "${CONTAINER_NAME}" \
  --label "l40s.slurm.job_id=${SLURM_JOB_ID}" \
  --label "l40s.delivery.batch=GON-34" \
  --network=none --ipc=host --security-opt=no-new-privileges --cap-drop=ALL \
  --gpus "\"device=${CUDA_VISIBLE_DEVICES:?}\"" \
  --user "$(id -u):$(id -g)" --workdir /workspace \
  --mount "type=bind,src=${WORKSPACE},dst=/workspace,readonly" \
  --mount "type=bind,src=${OUTPUT},dst=/output" \
  --mount "type=bind,src=${SCRATCH},dst=/scratch" \
  -e PYTHONPATH=/workspace -e PYTHONDONTWRITEBYTECODE=1 \
  -e WANDB_MODE=offline -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HOME=/scratch/hf -e XDG_CACHE_HOME=/scratch/xdg -e TMPDIR=/scratch/tmp \
  -e FLASH_ATTENTION_DETERMINISTIC=1 -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "${IMAGE_ID}" bash -lc '
    set -euo pipefail
    mkdir -p /scratch/hf /scratch/xdg /scratch/tmp /output
    torchrun --standalone --nproc-per-node=8 \
      tests/special_distributed/test_dynamic_permutation_fsdp_smoke.py \
      --candidate-sha '"${GON34_CANDIDATE_SHA}"' \
      --workspace /workspace \
      --output /output
  ' >"${LOGS}/stdout.log" 2>"${LOGS}/stderr.log"
