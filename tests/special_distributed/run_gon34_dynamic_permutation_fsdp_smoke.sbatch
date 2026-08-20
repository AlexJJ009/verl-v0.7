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
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${GON34_CANDIDATE_SHA:?required}"
: "${GON34_STAGE_REL:?required}"
: "${GON34_OUTPUT_REL:?required}"
: "${GON34_NODE_ROOT_MAP:?required semicolon-separated node=/absolute/root entries}"

die() {
  printf 'GON34 admission failed: %s\n' "$*" >&2
  exit 64
}

validate_rel_path() {
  local name="$1"
  local rel="$2"
  local part
  local -a parts

  test -n "${rel}" || die "${name} is empty"
  test "${rel#/}" = "${rel}" || die "${name} must be relative"
  [[ "${rel}" != *$'\n'* && "${rel}" != *$'\r'* ]] || die "${name} contains control characters"
  IFS='/' read -r -a parts <<<"${rel}"
  for part in "${parts[@]}"; do
    test -n "${part}" && test "${part}" != . && test "${part}" != .. \
      || die "${name} contains an unsafe path component"
    [[ "${part}" =~ ^[A-Za-z0-9._=-]+$ ]] || die "${name} contains unsupported characters"
  done
}

canonical_existing_child_dir() {
  local root="$1"
  local rel="$2"
  local name="$3"
  local path

  validate_rel_path "${name}" "${rel}"
  path="$(realpath -e "${root}/${rel}")" || die "${name} does not exist"
  test -d "${path}" || die "${name} must identify a directory"
  case "${path}" in
    "${root}"/*) printf '%s\n' "${path}" ;;
    *) die "${name} escapes node root after canonicalization" ;;
  esac
}

[[ "${GON34_CANDIDATE_SHA}" =~ ^[0-9a-f]{40}$ ]] \
  || die "GON34_CANDIDATE_SHA must be exact lowercase 40-hex"
test "${SLURMD_NODENAME:?}" != controller-dev
[[ "${SLURM_JOB_ID:?}" =~ ^[0-9]+$ ]] || die "SLURM_JOB_ID must be numeric"
NODE_ROOT=""
IFS=';' read -r -a node_roots <<<"${GON34_NODE_ROOT_MAP}"
for node_root in "${node_roots[@]}"; do
  if test "${node_root%%=*}" = "${SLURMD_NODENAME}"; then
    NODE_ROOT="${node_root#*=}"
    break
  fi
done
test -n "${NODE_ROOT}" && test "${NODE_ROOT#/}" != "${NODE_ROOT}"
NODE_ROOT="$(realpath -e "${NODE_ROOT}")" || die "NODE_ROOT does not exist"
test -d "${NODE_ROOT}" || die "NODE_ROOT must identify a directory"
test "${NODE_ROOT}" != / || die "NODE_ROOT must not be the filesystem root"

STAGE_PREFIX="$(canonical_existing_child_dir "${NODE_ROOT}" workspace/jobs STAGE_PREFIX)"
OUTPUT_PREFIX="$(canonical_existing_child_dir "${NODE_ROOT}" checkpoints/jobs OUTPUT_PREFIX)"
WORKSPACE="$(canonical_existing_child_dir "${NODE_ROOT}" "${GON34_STAGE_REL}" GON34_STAGE_REL)"
OUTPUT_BASE="$(canonical_existing_child_dir "${NODE_ROOT}" "${GON34_OUTPUT_REL}" GON34_OUTPUT_REL)"
SCRATCH_BASE="$(canonical_existing_child_dir "${NODE_ROOT}" scratch/jobs SCRATCH_BASE)"
LOGS_BASE="$(canonical_existing_child_dir "${NODE_ROOT}" logs/jobs LOGS_BASE)"
case "${WORKSPACE}" in
  "${STAGE_PREFIX}"/*) ;;
  *) die "GON34_STAGE_REL escapes intended workspace/jobs prefix" ;;
esac
case "${OUTPUT_BASE}" in
  "${OUTPUT_PREFIX}"/*) ;;
  *) die "GON34_OUTPUT_REL escapes intended checkpoints/jobs prefix" ;;
esac
OUTPUT="${OUTPUT_BASE}/${SLURM_JOB_ID}"
SCRATCH="${SCRATCH_BASE}/${SLURM_JOB_ID}-gon34-dynperm"
LOGS="${LOGS_BASE}/${SLURM_JOB_ID}-gon34-dynperm"
CONTAINER_NAME="l40s-job-${SLURM_JOB_ID}-gon34-dynperm-smoke"

test -f "${WORKSPACE}/.candidate-sha"
test "$(tr -d '\n' < "${WORKSPACE}/.candidate-sha")" = "${GON34_CANDIDATE_SHA}"
test -z "$(find "${WORKSPACE}" \( -type f -o -type d \) -perm /222 -print -quit)"
for job_path in "${OUTPUT}" "${SCRATCH}" "${LOGS}"; do
  test ! -e "${job_path}" && test ! -L "${job_path}" || die "job-owned path already exists: ${job_path}"
  mkdir "${job_path}"
done
test "$(realpath -e "${OUTPUT}")" = "${OUTPUT}" || die "output path canonicalization mismatch"
test -z "$(find "${OUTPUT}" -mindepth 1 -print -quit)"

ADMISSION_DIR="${LOGS}/node-local-admission"
mkdir "${ADMISSION_DIR}"
printf '{"candidate_sha":"%s","job_id":"%s","node":"%s","cuda_visible_devices":"%s","receipt_dir":"%s","admitted":false}\n' \
  "${GON34_CANDIDATE_SHA}" "${SLURM_JOB_ID}" "${SLURMD_NODENAME}" "${CUDA_VISIBLE_DEVICES:-}" \
  "${ADMISSION_DIR}" >"${ADMISSION_DIR}/receipt.json"
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name --format=csv,noheader,nounits \
  >"${ADMISSION_DIR}/gpu-compute-apps.csv"
docker ps --format '{{.ID}}|{{.Names}}|{{.Image}}|{{.Labels}}' >"${ADMISSION_DIR}/docker-ps.txt"
squeue -h -w "${SLURMD_NODENAME}" -t RUNNING,COMPLETING -o '%A|%t|%N|%j' \
  >"${ADMISSION_DIR}/slurm-node-jobs.txt"
awk -F'|' -v self="${SLURM_JOB_ID}" '$1 != self {print}' \
  "${ADMISSION_DIR}/slurm-node-jobs.txt" >"${ADMISSION_DIR}/foreign-slurm-jobs.txt"
test ! -s "${ADMISSION_DIR}/gpu-compute-apps.csv" || die "foreign GPU compute process present"
test ! -s "${ADMISSION_DIR}/docker-ps.txt" || die "foreign container present"
test ! -s "${ADMISSION_DIR}/foreign-slurm-jobs.txt" || die "foreign Slurm allocation present"
printf '{"candidate_sha":"%s","job_id":"%s","node":"%s","cuda_visible_devices":"%s","receipt_dir":"%s","admitted":true}\n' \
  "${GON34_CANDIDATE_SHA}" "${SLURM_JOB_ID}" "${SLURMD_NODENAME}" "${CUDA_VISIBLE_DEVICES:?}" \
  "${ADMISSION_DIR}" >"${ADMISSION_DIR}/receipt.json"

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
  -e NCCL_SOCKET_IFNAME=lo -e GLOO_SOCKET_IFNAME=lo \
  -e FLASH_ATTENTION_DETERMINISTIC=1 -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -e GON34_CANDIDATE_SHA="${GON34_CANDIDATE_SHA}" \
  "${IMAGE_ID}" bash -c '
    set -euo pipefail
    mkdir -p /scratch/hf /scratch/xdg /scratch/tmp /output
    exec torchrun --nnodes=1 --node-rank=0 \
      --master-addr=127.0.0.1 --master-port=29500 \
      --nproc-per-node=8 \
      tests/special_distributed/test_dynamic_permutation_fsdp_smoke.py \
      --candidate-sha "$GON34_CANDIDATE_SHA" \
      --workspace /workspace \
      --output /output
  ' >"${LOGS}/stdout.log" 2>"${LOGS}/stderr.log"
