import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_code_training_queues_use_cot_v3_namespaces_and_invalidate_legacy_stage123():
    code_root = ROOT / "recipe/on_policy_wdl_sft/code_task"
    stage1_queue = (code_root / "run_code_task_kodcode_qwen3_1p7b_coldstart_fraction_stage1_queue.sh").read_text()
    stage2_queue = (code_root / "run_code_task_kodcode_qwen3_1p7b_coldstart_fraction_stage2_p40_m2kl_vs_nokl_queue.sh").read_text()
    stage123_queue = (code_root / "run_code_task_qwen3_1p7b_stage123_queue_impl.sh").read_text()
    split_queue = (code_root / "run_code_task_qwen3_1p7b_stage123_model2_kl_split_stage3_queue.sh").read_text()
    cold_monitor = (code_root / "monitor_code_task_qwen3_1p7b_coldstart_sft_fraction_notify.sh").read_text()
    legacy = yaml.safe_load((ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml").read_text())
    split = yaml.safe_load(
        (ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/stage123_model2_kl_split_stage3.yaml").read_text()
    )
    assert "format_cold_start_fraction_cot_v3" in stage1_queue
    assert "COLDSTART-FRAC25-COT-V3" in stage1_queue
    assert "COLDSTART-FRAC25-COT-V3" in stage2_queue
    assert "format_cold_start_fraction_cot_v3" in cold_monitor
    assert "launch_allowed" in stage123_queue
    assert "launch_allowed" in split_queue
    assert legacy["launch_allowed"] is False and "invalidated" in legacy["status"]
    assert split["launch_allowed"] is False and "invalidated" in split["status"]


def load_code_prepare_module():
    path = ROOT / "recipe/on_policy_wdl_sft/format_cold_start/prepare_code_kodcode_sft_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_code_kodcode_sft_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_kodcode_rl_prepare_module():
    path = ROOT / "recipe/on_policy_wdl_sft/code_task/prepare_kodcode_light_rl_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_kodcode_light_rl_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_kodcode_rl_prompt_restores_author_provided_function_declaration():
    module = load_kodcode_rl_prepare_module()
    row = {
        "question": "Return one more than the input.",
        "test_info": [
            {
                "docstring": "Return x plus one.",
                "function_declaration": "def add_one(x):",
                "function_name": "add_one",
                "parameter_list": "(x)",
            }
        ],
    }

    contract = module.extract_function_contract(row)
    prompt = module.build_prompt(row["question"], contract["function_declaration"])
    user_content = prompt[-1]["content"]

    assert module.AUTHOR_FUNCTION_DECLARATION_TEMPLATE.format(
        function_declaration="def add_one(x):"
    ) in user_content
    assert module.AUTHOR_PREPROCESSING_COMMIT == "c348f894a803d0eff3c4d529dbf82af6e1262ae1"
    assert "<think>your concise reasoning</think>" in user_content
    assert "<answer>\n```python" in user_content
    assert module.PROMPT_TEMPLATE_VERSION == "code-think-answer-python-v2-kodcode-author-signature"


def test_kodcode_rl_prompt_rejects_missing_or_inconsistent_function_contract():
    module = load_kodcode_rl_prepare_module()

    for test_info in (
        [],
        [{"function_name": "add_one", "function_declaration": ""}],
        [{"function_name": "add_one", "function_declaration": "def different(x):"}],
    ):
        try:
            module.extract_function_contract({"test_info": test_info})
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid test_info was accepted: {test_info!r}")


def test_kodcode_rl_contract_accepts_valid_unicode_python_identifier():
    module = load_kodcode_rl_prepare_module()
    contract = module.extract_function_contract(
        {
            "test_info": [
                {
                    "function_name": "无不有",
                    "function_declaration": "def 无不有(longestWord, shortestWord):",
                    "parameter_list": "(longestWord, shortestWord)",
                }
            ]
        }
    )
    assert contract["function_name"] == "无不有"


def test_locally_adapted_code_validation_preserves_problem_and_adds_shared_contract():
    common = load_kodcode_rl_prepare_module()
    validation_path = ROOT / "recipe/on_policy_wdl_sft/code_task/prepare_official_only_validation.py"
    spec = importlib.util.spec_from_file_location("prepare_official_only_validation", validation_path)
    validation = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(validation)

    canonical = "def identity(x):\n    return x"
    record = validation._record("HumanEval+", "sample", canonical, {}, {"benchmark": "HumanEval+"})
    user_content = record["prompt"][-1]["content"]

    assert user_content.startswith(canonical)
    assert common.CONTRACT_SUFFIX in user_content
    assert "<think>your concise reasoning</think>" in user_content
    assert "<answer>\n```python" in user_content


def test_livecodebench_local_adapter_requires_python_stdio_without_changing_question():
    validation_path = ROOT / "recipe/on_policy_wdl_sft/code_task/prepare_official_only_validation.py"
    spec = importlib.util.spec_from_file_location("prepare_official_only_validation_lcb", validation_path)
    validation = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(validation)

    question = "Read one integer and print it."
    prompt = validation.build_livecodebench_prompt(question, "")

    assert question in prompt
    assert "complete Python program" in prompt
    assert "stdin" in prompt and "stdout" in prompt


def test_real_kodcode_author_signature_dataset_has_full_model_visible_contract():
    dataset_path = Path(
        "/data-1/dataset/code/verl_rl/kodcode_light_rl_10k_train_rl_format_author_signature_v2.parquet"
    )
    if not dataset_path.is_file():
        return

    frame = pd.read_parquet(dataset_path, columns=["prompt", "extra_info"])
    assert len(frame) == 10000
    for prompt, extra in zip(frame["prompt"], frame["extra_info"], strict=True):
        messages = prompt.tolist() if hasattr(prompt, "tolist") else prompt
        user_content = "\n".join(message["content"] for message in messages if message["role"] == "user")
        assert extra["function_declaration"] in user_content
        assert extra["entry_point"] in user_content
        assert "<think>your concise reasoning</think>" in user_content
        assert "<answer>\n```python" in user_content


def test_code_cold_start_uses_raw_problem_specific_reasoning():
    module = load_code_prepare_module()
    source = pd.DataFrame(
        [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "Return x + 1.\n\nNote that the required function declaration is "
                        "`def add_one(x):`. Preserve this exact function name and parameter signature.",
                    }
                ],
                "data_source": "kodcode_light_rl_10k",
                "split": "train",
                "extra_info": {
                    "original_solution": "def add_one(x):\n    return x + 1",
                    "entry_point": "add_one",
                    "function_declaration": "def add_one(x):",
                    "prompt_template_version": "code-think-answer-python-v2-kodcode-author-signature",
                    "raw_index": 0,
                    "uid": "row-0",
                },
            }
        ]
    )
    raw = pd.DataFrame(
        [
            {
                "conversations": [
                    {"from": "human", "value": "Return x + 1."},
                    {
                        "from": "gpt",
                        "value": "<think>Adding one directly satisfies the function contract.</think>\n"
                        "```python\ndef add_one(x):\n    return x + 1\n```",
                    },
                ]
            }
        ]
    )

    converted = module.convert(source, raw, seed=1, max_samples=-1, tokenizer=None)
    assistant = converted.iloc[0]["messages"][-1]["content"]

    assert "Adding one directly satisfies" in assistant
    assert "We need provide executable Python code" not in assistant
    assert "<answer>\n```python" in assistant
    assert converted.iloc[0]["extra_info"]["format_cold_start"] == "code-cot-python-answer-v3"
    assert converted.iloc[0]["extra_info"]["function_declaration"] == "def add_one(x):"


def test_code_cold_start_rejects_degenerate_repetition():
    module = load_code_prepare_module()
    quality = module.reasoning_quality("wait no " * 500)
    assert quality["accepted"] is False
    assert quality["repeat_8gram_fraction"] > module.MAX_REPEAT_8GRAM_FRACTION


def test_code_cold_start_preserves_raw_source_index_after_seeded_selection():
    module = load_code_prepare_module()
    raw = pd.DataFrame(
        [
            {
                "question": f"Task {raw_index}.",
                "conversations": [
                    {"from": "gpt", "value": f"<think>Reasoning for raw row {raw_index}.</think>"}
                ],
            }
            for raw_index in range(21)
        ]
    )
    source = pd.DataFrame(
        [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": f"Task {raw_index}. Required function `def f{raw_index}(x):`.",
                    }
                ],
                "extra_info": {
                    "original_solution": f"def f{raw_index}(x):\n    return x",
                    "entry_point": f"f{raw_index}",
                    "function_declaration": f"def f{raw_index}(x):",
                    "raw_index": raw_index,
                    "uid": module.expected_uid(raw_index, raw.iloc[raw_index]),
                },
            }
            for raw_index in (10, 20)
        ]
    )
    converted = module.convert(source, raw, seed=7, max_samples=-1, tokenizer=None)

    assert set(converted["extra_info"].map(lambda extra: extra["source_index"])) == {10, 20}
    for messages, extra in zip(converted["messages"], converted["extra_info"], strict=True):
        joined = "\n".join(message["content"] for message in messages)
        assert extra["rl_source_index"] in {0, 1}
        assert extra["entry_point"] == f"f{extra['source_index']}"
        assert extra["function_declaration"] in joined


def test_code_cold_start_rejects_raw_index_uid_mismatch():
    module = load_code_prepare_module()
    raw = pd.DataFrame(
        [{"question": "Return x.", "conversations": [{"from": "gpt", "value": "<think>Reasoning.</think>"}]}]
    )
    source = pd.DataFrame(
        [{
            "prompt": [{"role": "user", "content": "Return x. Required function `def solve(x):`."}],
            "extra_info": {
                "raw_index": 0,
                "uid": "0" * 24,
                "entry_point": "solve",
                "function_declaration": "def solve(x):",
                "original_solution": "def solve(x):\n    return x",
            },
        }]
    )
    with pytest.raises(ValueError, match="raw_index does not match extra_info.uid"):
        module.convert(source, raw, seed=1, max_samples=-1)


def test_code_and_math_cold_start_launchers_require_whole_message_preflight():
    for relative_path in (
        "recipe/on_policy_wdl_sft/format_cold_start/run_sft_code_qwen3_1p7b_kodcode_format.sh",
        "recipe/on_policy_wdl_sft/format_cold_start/run_sft_math_qwen3_1p7b_format.sh",
    ):
        text = (ROOT / relative_path).read_text()
        assert '"data.tokenize_whole_message=True"' in text
        assert '"data.ignore_input_ids_mismatch=False"' in text
        assert "scripts/validate_sft_loss_mask.py" in text
    code_text = (
        ROOT / "recipe/on_policy_wdl_sft/format_cold_start/run_sft_code_qwen3_1p7b_kodcode_format.sh"
    ).read_text()
    assert "MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}" in code_text
    assert "MAX_LENGTH=${MAX_LENGTH:-9216}" in code_text
    assert "LR=${LR:-5e-6}" in code_text
    assert "LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-5}" in code_text
    assert "DATA_SHUFFLE=${DATA_SHUFFLE:-False}" in code_text
    assert "TRAIN_SEED=${TRAIN_SEED:-20260706}" in code_text
    assert '"+data.shuffle=${DATA_SHUFFLE}"' in code_text
    assert '"trainer.seed=${TRAIN_SEED}"' in code_text
    assert '"optim.lr_warmup_steps=${LR_WARMUP_STEPS}"' in code_text


def test_code_cold_start_queue_matches_math_lr_and_full_n3_eval_contract():
    queue = (
        ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_task_qwen3_1p7b_coldstart_sft_fraction_queue.sh"
    ).read_text()
    evaluator = (ROOT / "recipe/on_policy_wdl_sft/code_task/eval_code_vllm.py").read_text()

    assert "LR=${LR:-5e-6}" in queue
    assert "LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-5}" in queue
    assert "TRAIN_SEED=${TRAIN_SEED:-$DATA_SEED}" in queue
    assert "DATA_SHUFFLE=${DATA_SHUFFLE:-False}" in queue
    assert 'TRAIN_SEED="$TRAIN_SEED" DATA_SHUFFLE="$DATA_SHUFFLE"' in queue
    assert 'LR="$LR" LR_WARMUP_STEPS="$LR_WARMUP_STEPS"' in queue
    assert "EVAL_N_SAMPLES=${EVAL_N_SAMPLES:-3}" in queue
    assert "EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-8192}" in queue
    assert "EVAL_TOP_K=${EVAL_TOP_K:--1}" in queue
    assert "--top-k '${EVAL_TOP_K}'" in queue
    assert 'parser.add_argument("--top-k", type=int, default=-1)' in evaluator
    assert "top_k=args.top_k" in evaluator


def test_code_cold_start_step_queue_matches_math_selection_contract():
    manifest = yaml.safe_load(
        (ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/code_qwen3_1p7b_cold_start_cotmask_v3.yaml").read_text()
    )
    queue = (ROOT / "scripts/code_cold_start_queue.py").read_text()
    monitor = (
        ROOT / "recipe/on_policy_wdl_sft/code_task/monitor_code_qwen3_1p7b_cold_start_cotmask_v3.sh"
    ).read_text()

    assert manifest["training"]["max_steps"] == 30
    assert manifest["training"]["step_interval"] == 5
    assert manifest["training"]["data_shuffle"] is False
    assert manifest["experiment_id"].endswith("author_signature_v2")
    assert "author_signature_v2" in manifest["paths"]["train_file"]
    assert "author_signature_v2" in manifest["paths"]["rl_train_file"]
    assert "AUTHOR-SIGNATURE-V2" in manifest["training"]["run_name"]
    assert manifest["seed"] == 20260706
    assert manifest["validation"]["n"] == 3
    assert manifest["validation"]["enable_thinking"] is True
    assert manifest["admission_thresholds"]["format_contract_success_rate"] == 0.85
    assert set(manifest["validation"]["benchmarks"]) == {"humaneval", "mbpp", "livecodebench"}
    assert 'for step in range(0, training["max_steps"] + 1, training["step_interval"])' in queue
    assert '"TOTAL_TRAINING_STEPS": str(step)' in queue
    assert '"DATA_SHUFFLE": "False"' in queue
    assert '"--enable-thinking"' in queue
    assert '"--verify-only"' in queue
    assert "prepare_code_kodcode_sft_dataset.py" in queue
    assert 'selection_policy = "earliest checkpoint with micro format_contract_success_rate >= threshold"' in queue
    assert 'selection_policy = "highest micro format_contract_success_rate; earliest step breaks ties"' in queue
    assert "PAUSE_FOR_STAGE123_ADMISSION" in manifest["paths"]["pause_marker"]
    assert "candidate_evaluated" in monitor
    assert "author_signature_v2_steps/events.jsonl" in monitor
    assert "author_signature_v2_steps/model1_selection.json" in monitor
    wrapper = (
        ROOT / "recipe/on_policy_wdl_sft/code_task/run_code_qwen3_1p7b_cold_start_cotmask_v3.sh"
    ).read_text()
    assert 'elif [ -z "${TMUX:-}" ]; then' in wrapper
    assert 'must be launched from a real host tmux session' in wrapper
    assert '-e TMUX="${TMUX:-}"' in wrapper


def test_code_format_contract_requires_all_tags_fence_extraction_and_eos(tmp_path):
    module_path = ROOT / "recipe/on_policy_wdl_sft/code_task/summarize_code_format_contract.py"
    spec = importlib.util.spec_from_file_location("summarize_code_format_contract", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    good = {
        "data_source": "HumanEval+",
        "solution_str": "<think>reason</think>\n<answer>\n```python\ndef solve():\n    return 1\n```\n</answer>",
        "finish_reason": "stop",
    }
    missing_think = {
        **good,
        "solution_str": "<answer>\n```python\ndef solve():\n    return 1\n```\n</answer>",
    }
    truncated = {**good, "finish_reason": "length"}

    assert module.evaluate_row(good)["format_contract_success"] is True
    assert module.evaluate_row(missing_think)["format_contract_success"] is False
    assert module.evaluate_row(truncated)["format_contract_success"] is False


def test_code_stage123_profile_is_ctx8k_n3_lr_aligned_and_entropy_free():
    code_root = ROOT / "recipe/on_policy_wdl_sft/code_task"
    profile = (code_root / "qwen3_1p7b_stage123_resource_profile.sh").read_text()
    stage1 = (code_root / "run_s1_code_kodcode_qwen3_1p7b_instruct_ctx8k_beta_0.sh").read_text()
    stage2 = (code_root / "run_s2_code_kodcode_qwen3_1p7b_instruct_ctx8k_p40_common.sh").read_text()
    stage3 = (code_root / "run_s3_code_qwen3_1p7b_stage123_common.sh").read_text()

    for text in (profile, stage1, stage2):
        assert "MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}" in text
        assert "ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-9216}" in text
    assert "VAL_N=${VAL_N:-3}" in profile
    assert "VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.2}" in profile
    assert "VAL_TOP_P=${VAL_TOP_P:-0.95}" in profile
    assert "TOP_K=${TOP_K:--1}" in profile
    assert "VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}" in profile
    assert "ACTOR_CALCULATE_ENTROPY=${ACTOR_CALCULATE_ENTROPY:-False}" in profile
    assert "CALCULATE_ENTROPY=${CALCULATE_ENTROPY:-False}" in profile
    for text in (stage1, stage2, stage3):
        assert "LR=${LR:-1e-6}" in text
        assert "LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-0}" in text
    assert "val-core/code3_macro/acc/mean@3" in stage1
    assert "val-core/model2/code3_macro/acc/mean@3" in stage2
    assert "code_stage123_macro_overrides" in stage3


def test_stage123_wrappers_use_strict_cot_reward_functions():
    code_stage1 = (ROOT / "recipe/on_policy_wdl_sft/code_task/run_s1_code_base.sh").read_text()
    code_stage2 = (ROOT / "recipe/on_policy_wdl_sft/code_task/run_s2_code_model2_rollout_common.sh").read_text()
    code_stage3 = (ROOT / "recipe/on_policy_wdl_sft/code_task/run_s3_code_qwen3_1p7b_stage123_common.sh").read_text()
    math_common = (ROOT / "recipe/on_policy_wdl_sft/ablation_single_model/_common_ablation.sh").read_text()

    assert "official_aligned_reward.py" in code_stage1
    assert "official_aligned_reward.py" in code_stage2
    assert "run_s1_code_base.sh" in code_stage3
    assert "custom_reward_function_latex_verify.py" in math_common
