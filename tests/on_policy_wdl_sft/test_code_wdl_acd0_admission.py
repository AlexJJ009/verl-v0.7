import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


admission = load_script(
    "code_wdl_acd0_admission",
    ROOT / "scripts/code_wdl_acd0_admission.py",
)
probe = load_script(
    "code_wdl_acd0_gpu_probe",
    ROOT / "scripts/code_wdl_acd0_gpu_probe.py",
)


def test_evaluator_result_requires_pass_fail_and_fail_closed_cases():
    results = {
        source: {
            "known_pass": {"score": 1.0, "code_reward_dependency_error": 0},
            "known_fail": {"score": -1.0, "code_reward_dependency_error": 0},
        }
        for source in admission.OFFICIAL_SOURCES
    }
    results["fail_closed"] = {
        "malformed": {"score": -1.0, "code_reward_status": "format_error"},
        "missing_eos": {"score": -1.0, "truncated": True, "has_eos": False},
    }

    checks = admission.validate_evaluator_results(results)

    assert checks and all(checks.values())


def test_evaluator_result_rejects_false_positive():
    results = {
        source: {
            "known_pass": {"score": 1.0, "code_reward_dependency_error": 0},
            "known_fail": {"score": -1.0, "code_reward_dependency_error": 0},
        }
        for source in admission.OFFICIAL_SOURCES
    }
    results["LiveCodeBench"]["known_fail"]["score"] = 1.0
    results["fail_closed"] = {
        "malformed": {"score": -1.0, "code_reward_status": "format_error"},
        "missing_eos": {"score": -1.0, "truncated": True, "has_eos": False},
    }

    checks = admission.validate_evaluator_results(results)

    assert checks["LiveCodeBench_known_fail"] is False


def test_gpu_probe_arm_contracts_are_distinct():
    common = {
        "optimizer_steps": 1,
        "positive_loss": 0.5,
        "n_correct": 7,
        "actor_grad_norm": 1.0,
        "formal_checkpoint_files": [],
    }
    arms = {
        "arm-a-stage1-continuation": dict(common),
        "arm-d0-matched-scale-no-weak": {
            **common,
            "model1_grad_norm": 0.0,
            "model2_grad_norm": 2.0,
        },
        "arm-c-mixture": {
            **common,
            "model1_grad_norm": 0.5,
            "model2_grad_norm": 2.0,
        },
    }

    checks = probe.validate_arm_results(arms)

    assert checks and all(checks.values())


def test_gpu_probe_rejects_d0_model1_gradient():
    common = {
        "optimizer_steps": 1,
        "positive_loss": 0.5,
        "n_correct": 7,
        "actor_grad_norm": 1.0,
        "formal_checkpoint_files": [],
    }
    arms = {
        "arm-a-stage1-continuation": dict(common),
        "arm-d0-matched-scale-no-weak": {
            **common,
            "model1_grad_norm": 0.1,
            "model2_grad_norm": 2.0,
        },
        "arm-c-mixture": {
            **common,
            "model1_grad_norm": 0.5,
            "model2_grad_norm": 2.0,
        },
    }

    checks = probe.validate_arm_results(arms)

    assert checks["d0_model1_gradient_zero"] is False


def test_gpu_probe_receipt_is_bound_to_manifest(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("experiment_id: test\n", encoding="utf-8")
    payload = probe.build_receipt(
        manifest,
        {"arm-a-stage1-continuation": {}},
        {"all": True},
        {"gpus": []},
    )

    assert payload["manifest_sha256"] == probe.sha256(manifest)
    assert payload["status"] == "pass"
    assert json.loads(json.dumps(payload))["checks"]["all"] is True
