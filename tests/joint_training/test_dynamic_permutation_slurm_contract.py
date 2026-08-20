# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import stat
import subprocess
from pathlib import Path

import pytest

SBATCH = Path("tests/special_distributed/run_gon34_dynamic_permutation_fsdp_smoke.sbatch")


def test_gpu_smoke_is_bounded_exclusive_and_controller_excluded():
    text = SBATCH.read_text()
    assert "#SBATCH --gres=gpu:L40S:8" in text
    assert "#SBATCH --exclusive" in text
    assert "#SBATCH --exclude=controller-dev" in text
    assert "#SBATCH --time=00:30:00" in text
    assert "#SBATCH --nice=10000" in text
    assert "#SBATCH --no-requeue" in text
    assert 'formal_experiment":false' in text


def test_gpu_smoke_is_candidate_bound_read_only_and_offline():
    text = SBATCH.read_text()
    assert "GON34_CANDIDATE_SHA" in text
    assert "GON34_NODE_ROOT_MAP" in text
    assert ".candidate-sha" in text
    assert "\\( -type f -o -type d \\) -perm /222" in text
    assert "dst=/workspace,readonly" in text
    assert "--network=none" in text
    assert "WANDB_MODE=offline" in text


def test_gpu_smoke_uses_fsdp_runner_and_unique_job_paths():
    text = SBATCH.read_text()
    assert "torchrun --standalone --nproc-per-node=8" in text
    assert "test_dynamic_permutation_fsdp_smoke.py" in text
    assert "${SLURM_JOB_ID}" in text


def test_gpu_smoke_rejects_non_exact_lowercase_sha_without_shell_splice():
    text = SBATCH.read_text()
    assert "^[0-9a-f]{40}$" in text
    assert "bash -lc" not in text
    assert "--candidate-sha '\"${GON34_CANDIDATE_SHA}\"'" not in text
    assert '-e GON34_CANDIDATE_SHA="${GON34_CANDIDATE_SHA}"' in text
    assert '--candidate-sha "$GON34_CANDIDATE_SHA"' in text


def test_gpu_smoke_canonicalizes_stage_and_output_paths_fail_closed():
    text = SBATCH.read_text()
    assert "validate_rel_path" in text
    assert "realpath -e" in text
    assert "must be relative" in text
    assert "contains an unsafe path component" in text
    assert "escapes node root after canonicalization" in text
    assert "OUTPUT_BASE=" in text
    assert 'OUTPUT="${OUTPUT_BASE}/${SLURM_JOB_ID}"' in text
    assert "workspace/jobs prefix" in text
    assert "checkpoints/jobs prefix" in text


def test_gpu_smoke_records_node_local_admission_and_blocks_foreign_workloads():
    text = SBATCH.read_text()
    assert "node-local-admission" in text
    assert "gpu-compute-apps.csv" in text
    assert "docker-ps.txt" in text
    assert "foreign-slurm-jobs.txt" in text
    assert "nvidia-smi --query-compute-apps" in text
    assert "docker ps --format" in text
    assert "squeue -h -w" in text
    assert '"admitted":false' in text
    assert '"admitted":true' in text
    assert "foreign GPU compute process present" in text
    assert "foreign container present" in text


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _wrapper_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, str], Path]:
    node_root = tmp_path / "node-root"
    workspace = node_root / "workspace" / "jobs" / "candidate"
    for path in (
        workspace,
        node_root / "checkpoints" / "jobs" / "candidate-output",
        node_root / "scratch" / "jobs",
        node_root / "logs" / "jobs",
    ):
        path.mkdir(parents=True, exist_ok=True)
    candidate_sha = "a" * 40
    (workspace / ".candidate-sha").write_text(candidate_sha + "\n")
    for path in [workspace / ".candidate-sha", workspace]:
        path.chmod(path.stat().st_mode & ~0o222)

    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{mock_bin}:{env['PATH']}",
            "GON34_CANDIDATE_SHA": candidate_sha,
            "GON34_STAGE_REL": "workspace/jobs/candidate",
            "GON34_OUTPUT_REL": "checkpoints/jobs/candidate-output",
            "GON34_NODE_ROOT_MAP": f"gpu-worker-b={node_root}",
            "SLURMD_NODENAME": "gpu-worker-b",
            "SLURM_JOB_ID": "999999",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        }
    )
    monkeypatch.setenv("GON34_TEST_NODE_ROOT", str(node_root))
    return env, mock_bin


@pytest.mark.parametrize(
    ("candidate_sha", "stage_rel", "output_rel", "message"),
    [
        ("A" * 40, "workspace/jobs/candidate", "checkpoints/jobs/candidate-output", "exact lowercase 40-hex"),
        ("a" * 39, "workspace/jobs/candidate", "checkpoints/jobs/candidate-output", "exact lowercase 40-hex"),
        ("a" * 40, "/workspace/jobs/candidate", "checkpoints/jobs/candidate-output", "must be relative"),
        ("a" * 40, "workspace/jobs/../candidate", "checkpoints/jobs/candidate-output", "unsafe path component"),
        ("a" * 40, "workspace/jobs/candidate", "../outputs", "unsafe path component"),
    ],
)
def test_gpu_smoke_rejects_invalid_sha_and_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_sha: str,
    stage_rel: str,
    output_rel: str,
    message: str,
):
    env, _ = _wrapper_env(tmp_path, monkeypatch)
    env.update(
        {
            "GON34_CANDIDATE_SHA": candidate_sha,
            "GON34_STAGE_REL": stage_rel,
            "GON34_OUTPUT_REL": output_rel,
        }
    )
    result = subprocess.run(["bash", str(SBATCH)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 64
    assert message in result.stderr


def test_gpu_smoke_rejects_stage_symlink_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    env, _ = _wrapper_env(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    node_root = Path(os.environ["GON34_TEST_NODE_ROOT"])
    (node_root / "workspace" / "jobs" / "escape").symlink_to(outside, target_is_directory=True)
    env["GON34_STAGE_REL"] = "workspace/jobs/escape"
    result = subprocess.run(["bash", str(SBATCH)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 64
    assert "escapes node root after canonicalization" in result.stderr


@pytest.mark.parametrize(
    ("stage_rel", "output_rel", "message"),
    [
        ("other-stage", "checkpoints/jobs/candidate-output", "workspace/jobs prefix"),
        ("workspace/jobs/candidate", "other-output", "checkpoints/jobs prefix"),
    ],
)
def test_gpu_smoke_rejects_paths_outside_intended_prefixes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_rel: str,
    output_rel: str,
    message: str,
):
    env, _ = _wrapper_env(tmp_path, monkeypatch)
    node_root = Path(os.environ["GON34_TEST_NODE_ROOT"])
    (node_root / "other-stage").mkdir()
    (node_root / "other-output").mkdir()
    env.update({"GON34_STAGE_REL": stage_rel, "GON34_OUTPUT_REL": output_rel})
    result = subprocess.run(["bash", str(SBATCH)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 64
    assert message in result.stderr


@pytest.mark.parametrize(
    ("gpu_output", "docker_output", "squeue_output", "message"),
    [
        ("123|GPU-0|python\n", "", "999999|R|gpu-worker-b|self\n", "foreign GPU compute process present"),
        ("", "deadbeef|foreign|image|labels\n", "999999|R|gpu-worker-b|self\n", "foreign container present"),
        ("", "", "999999|R|gpu-worker-b|self\n42|R|gpu-worker-b|foreign\n", "foreign Slurm allocation present"),
    ],
)
def test_gpu_smoke_foreign_workload_admission_fails_with_durable_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gpu_output: str,
    docker_output: str,
    squeue_output: str,
    message: str,
):
    env, mock_bin = _wrapper_env(tmp_path, monkeypatch)
    env.update(
        {
            "MOCK_GPU_OUTPUT": gpu_output,
            "MOCK_DOCKER_OUTPUT": docker_output,
            "MOCK_SQUEUE_OUTPUT": squeue_output,
        }
    )
    _write_executable(mock_bin / "nvidia-smi", "printf '%s' \"${MOCK_GPU_OUTPUT}\"\n")
    _write_executable(
        mock_bin / "docker",
        'test "${1:-}" = ps || exit 99\nprintf \'%s\' "${MOCK_DOCKER_OUTPUT}"\n',
    )
    _write_executable(mock_bin / "squeue", "printf '%s' \"${MOCK_SQUEUE_OUTPUT}\"\n")
    result = subprocess.run(["bash", str(SBATCH)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 64
    assert message in result.stderr
    node_root = Path(os.environ["GON34_TEST_NODE_ROOT"])
    receipt = node_root / "logs" / "jobs" / "999999-gon34-dynperm" / "node-local-admission" / "receipt.json"
    assert receipt.is_file()
    assert '"admitted":false' in receipt.read_text()
