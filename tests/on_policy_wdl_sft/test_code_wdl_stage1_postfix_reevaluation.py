import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


reeval = load_script(
    "code_wdl_stage1_postfix_reevaluation",
    ROOT / "scripts/code_wdl_stage1_postfix_reevaluation.py",
)


def test_contract_binds_frozen_model_code3_and_postfix_runtime_hashes():
    contract = reeval.FROZEN_CONTRACT
    assert contract["model"]["path"] == (
        "/data-2/model_weights/code_task/"
        "qwen3_1p7b_stage123_cotmask_v3_author_signature_v2_step20/"
        "b0-stage1/final_model"
    )
    assert contract["model"]["config_sha256"] == "a4a451865e8d45a519133031f19cda7d347813159fde1756d63e2beaf67f2288"
    assert contract["model"]["weights_sha256"] == "a6c69262975ada9e1bc5054128d9f6f79b14167653ba817809bc771799d43c74"
    assert contract["validation"] == {
        "HumanEval+": {
            "path": "/data-1/dataset/code/verl_rl/online_full_humaneval_plus/official_humaneval_plus_val.parquet",
            "parquet_rows": 164,
            "expected_eval_rows": 164,
            "sha256": "e317c71511c7b6b3df98ef88bf409644bc000e11a0621a57cdc944ccb82a9fab",
        },
        "MBPP+": {
            "path": "/data-1/dataset/code/verl_rl/online_full_mbpp_plus/official_mbpp_plus_val.parquet",
            "parquet_rows": 378,
            "expected_eval_rows": 378,
            "sha256": "3221e7f53c88bfbd91d788fb7bcb37168fb088fa504fddf12b9126c2147312d2",
        },
        "LiveCodeBench": {
            "path": "/data-1/dataset/code/verl_rl/online_full_livecodebench_v5/official_livecodebench_val.parquet",
            "parquet_rows": 880,
            "expected_eval_rows": 837,
            "sha256": "fe7d2bfe2779bcf106492347ca173e30b9220c15c1b8783949d35edcd93a43d1",
        },
    }
    assert contract["runtime_hashes"] == {
        "recipe/on_policy_wdl_sft/code_task/official_aligned_reward.py": "2854639c4bd3e34b89b3b4d53d553406b46a800fb44ff0c3657670f2792c59a2",
        "verl/workers/reward_manager/dapo.py": "4d05aaf514a199bca81d393d9d057eeb5f38b7067303e010f16a4c2b17c4829b",
        "verl/experimental/reward_loop/reward_manager/dapo.py": "54c24d5df68c0c6afc86b534e2ecd0fef3842de3b6295d71dbfbac852dc701ba",
    }


def test_contract_freezes_generation_and_expected_response_count():
    assert reeval.FROZEN_CONTRACT["generation"] == {
        "n": 3,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_response_length": 8192,
    }
    assert reeval.FROZEN_CONTRACT["expected_response_rows"] == 4137


def test_launcher_materializes_frozen_generation_contract(tmp_path: Path):
    environment = reeval._evaluation_environment(tmp_path / "output")
    assert environment["VAL_N"] == "3"
    assert environment["STAGE123_EXPECTED_VAL_N"] == "3"
    assert environment["VAL_TEMPERATURE"] == "0.2"
    assert environment["VAL_TOP_P"] == "0.95"
    assert environment["VAL_DO_SAMPLE"] == "True"
    assert environment["MAX_RESPONSE_LENGTH"] == "8192"


def test_real_launch_requires_tmux(tmp_path: Path):
    with pytest.raises(RuntimeError, match="tmux"):
        reeval.validate_launch_guard(tmp_path / "new-output", real_run=True, environ={})


def test_real_launch_refuses_output_collision(tmp_path: Path):
    output_root = tmp_path / "existing-output"
    output_root.mkdir()
    with pytest.raises(RuntimeError, match="collision"):
        reeval.validate_launch_guard(output_root, real_run=True, environ={"TMUX": "/tmp/tmux"})


def test_candidate_receipt_must_not_overwrite_admission_receipt(tmp_path: Path):
    candidate = tmp_path / "stage1_reuse_receipt.candidate.json"
    reeval.validate_candidate_receipt_path(candidate)
    with pytest.raises(RuntimeError, match="admission receipt"):
        reeval.validate_candidate_receipt_path(Path(reeval.ADMISSION_RECEIPT))


def _write_rows(path: Path):
    rows = []
    for source in ("HumanEval+", "MBPP+", "LiveCodeBench"):
        rows.append(
            {
                "data_source": source,
                "format_contract_success": True,
                "acc": 1.0,
                "score": 1.0,
                "response_finish_reason": "stop",
                "response_eos_present": True,
                "code_reward_dependency_error": 0,
                "code_reward_timeout": 0,
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_postfix_gate_receipt_binds_provenance_sha256(tmp_path: Path):
    current = tmp_path / "fresh-step40.jsonl"
    baseline = tmp_path / "historical-step0.jsonl"
    provenance = tmp_path / "post_fix_reevaluation_provenance.json"
    output = tmp_path / "stage1_reuse_receipt.candidate.json"
    _write_rows(current)
    _write_rows(baseline)
    provenance.write_text(
        json.dumps({"schema_version": 1, "evaluation_kind": "post_fix_reevaluation"}) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/code_wdl_stage1_reuse_gate.py"),
            "--current-file",
            str(current),
            "--baseline-file",
            str(baseline),
            "--step",
            "40",
            "--baseline-step",
            "0",
            "--evaluation-kind",
            "post_fix_reevaluation",
            "--provenance",
            str(provenance),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "pass"
    assert receipt["decision"] == "reuse_allowed"
    assert receipt["provenance_file"] == str(provenance.resolve())
    assert receipt["provenance_sha256"] == hashlib.sha256(provenance.read_bytes()).hexdigest()
