#!/usr/bin/env bash
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TEST_PATH=tests/utils/dataset/test_multiturn_sft_dataset_qwen_thinking_on_cpu.py

run_gate() {
    python3 scripts/check_sft_loss_mask_policy.py
    pytest -q "$TEST_PATH"
}

if python3 -c 'import transformers, torch, pandas, pyarrow' >/dev/null 2>&1; then
    cd "$REPO_ROOT"
    run_gate
elif command -v docker >/dev/null 2>&1 && docker image inspect verl-harness:latest >/dev/null 2>&1; then
    docker run --rm -i \
        -e PYTHONPATH=/workspace/verl \
        -v /data-1:/data-1 \
        -v /data-2:/data-2 \
        -v "$REPO_ROOT:/workspace/verl" \
        -w /workspace/verl \
        verl-harness:latest \
        bash -lc "python3 scripts/check_sft_loss_mask_policy.py && pytest -q $TEST_PATH"
else
    echo "SFT loss-mask gate requires the project Python environment or verl-harness:latest." >&2
    exit 1
fi
