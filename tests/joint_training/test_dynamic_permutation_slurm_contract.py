# Copyright 2026 The verl authors.
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

from pathlib import Path

SBATCH = Path("tests/special_distributed/run_gon34_dynamic_permutation_fsdp_smoke.sbatch")


def test_gpu_smoke_is_bounded_exclusive_and_controller_excluded():
    text = SBATCH.read_text()
    assert "#SBATCH --gres=gpu:L40S:8" in text
    assert "#SBATCH --exclusive" in text
    assert "#SBATCH --exclude=controller-dev" in text
    assert "#SBATCH --time=00:30:00" in text
    assert "#SBATCH --nice=10000" in text
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
