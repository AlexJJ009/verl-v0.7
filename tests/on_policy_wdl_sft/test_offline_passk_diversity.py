import json
import os
import subprocess
from pathlib import Path

import pytest

from recipe.joint_training.merge_passk_shards import _validate_contracts, _validate_coverage, merge_code
from recipe.joint_training.offline_eval import render_chat_prompt, stable_prompt_id


class FakeQwenTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking=None):
        assert tokenize is False
        assert add_generation_prompt is True
        suffix = "<assistant>" if enable_thinking is not False else "<assistant><think>\n\n</think>\n\n"
        return json.dumps(messages, sort_keys=True) + suffix


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _contract(path: Path, *, temperature: float = 0.6) -> None:
    path.write_text(
        json.dumps(
            {
                "model": "/models/qwen3",
                "validation_parquet": "/data/eval.parquet",
                "generation_params": {
                    "n": 2,
                    "temperature": temperature,
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p": 0.0,
                    "max_tokens": 4096,
                    "seed": 42,
                    "sample_offset": 0,
                    "chat_template_kwargs": {"enable_thinking": True},
                    "thinking_canary": {"template_effect": True},
                },
            }
        ),
        encoding="utf-8",
    )


def test_thinking_is_explicit_in_prompt_render_and_prompt_id_is_stable():
    tokenizer = FakeQwenTokenizer()
    messages = [{"role": "user", "content": "1+1?"}]
    enabled = render_chat_prompt(tokenizer, messages, True)
    disabled = render_chat_prompt(tokenizer, messages, False)
    assert enabled != disabled
    assert stable_prompt_id("math", messages, "2") == stable_prompt_id("math", messages, "2")
    assert stable_prompt_id("math", messages, "2") != stable_prompt_id("math", messages, "3")


def test_code_shards_merge_only_with_exact_sample_coverage(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    rows = []
    for task_index in range(2):
        for sample_index in range(4):
            rows.append(
                {
                    "task_index": task_index,
                    "task_id": f"task-{task_index}",
                    "prompt_id": f"prompt-{task_index}",
                    "sample_index": sample_index,
                    "data_source": "HumanEval+",
                    "solution_str": f"solution-{task_index}-{sample_index}",
                }
            )
    _write_jsonl(first, [row for row in rows if row["sample_index"] < 2])
    _write_jsonl(second, [row for row in rows if row["sample_index"] >= 2])
    output = tmp_path / "merged.jsonl"
    result = merge_code([first, second], output, expected_n=4)
    assert result["prompt_count"] == 2
    assert result["response_count"] == 8
    merged = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["sample_index"] for row in merged[:4]] == [0, 1, 2, 3]


def test_code_shards_reject_duplicate_or_missing_sample_index(tmp_path):
    path = tmp_path / "bad.jsonl"
    _write_jsonl(
        path,
        [
            {"prompt_id": "p", "sample_index": 0, "data_source": "x", "solution_str": "a"},
            {"prompt_id": "p", "sample_index": 0, "data_source": "x", "solution_str": "b"},
        ],
    )
    with pytest.raises(ValueError, match="coverage failed"):
        merge_code([path], tmp_path / "out.jsonl", expected_n=2)


def test_math_coverage_keeps_duplicate_prompt_text_as_distinct_dataset_rows():
    rows = []
    for dataset_row_index in (293, 294):
        for sample_index in range(4):
            rows.append(
                {
                    "dataset_path": "/datasets/mawps.parquet",
                    "dataset_row_index": dataset_row_index,
                    "prompt_id": "same-content-hash",
                    "sample_index": sample_index,
                    "data_source": "mwpt5/MAWPS",
                }
            )
    coverage = _validate_coverage(rows, expected_n=4)
    assert coverage == {"prompt_count": 2, "response_count": 8, "expected_n": 4}


def test_contract_requires_thinking_and_matching_decode_params(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _contract(first)
    _contract(second)
    assert _validate_contracts([first, second])["params"]["temperature"] == 0.6
    _contract(second, temperature=1.0)
    with pytest.raises(ValueError, match="contract mismatch"):
        _validate_contracts([first, second])


@pytest.mark.parametrize(("task", "expected_max_num_seqs"), [("math", "64"), ("code", "32")])
def test_8gpu_entry_uses_eight_tp1_sample_shards(task, expected_max_num_seqs):
    script = Path("recipe/on_policy_wdl_sft/offline_eval/run_qwen3_1p7b_passk_8gpu.sh")
    result = subprocess.run(
        ["bash", str(script)],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "TASK": task,
            "MODEL_PATH": "/models/qwen3",
            "OUTPUT_ROOT": "/outputs/eval",
            "EVAL_CONFIG_ONLY": "1",
        },
    )
    config = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert config["gpu_ids"] == "0,1,2,3,4,5,6,7"
    assert config["tensor_parallel"] == "1"
    assert config["total_n"] == "256"
    assert config["n_per_shard"] == "32"
    assert config["num_shards"] == "8"
    assert config["temperature"] == "0.6"
    assert config["gpu_memory_utilization"] == "0.90"
    assert config["max_num_seqs"] == expected_max_num_seqs
    assert config["max_num_batched_tokens"] == "8192"
    assert config["enforce_eager"] == "false"


def test_8gpu_entry_accepts_only_tmux_or_explicit_scheduler_management():
    script = Path("recipe/on_policy_wdl_sft/offline_eval/run_qwen3_1p7b_passk_8gpu.sh").read_text()
    assert '"${EVAL_SCHEDULER_MANAGED:-0}" != 1' in script
    assert "inside tmux or an admitted scheduler-managed worker" in script


def test_8gpu_entry_uses_spawn_for_vllm_engine_processes():
    script = Path("recipe/on_policy_wdl_sft/offline_eval/run_qwen3_1p7b_passk_8gpu.sh").read_text()
    assert "VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}" in script
