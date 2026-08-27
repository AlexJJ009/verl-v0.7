import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
RENDERER = ROOT / "scripts/a800/render_gon35_grpo_admission.py"
SHIM = ROOT / "scripts/a800/gon35-bin/verl-dev-run"
PYTHON_STARTUP = ROOT / "scripts/a800/gon35-python-startup"


def load_renderer():
    spec = importlib.util.spec_from_file_location("gon35_admission", RENDERER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_renderer_pins_recipe_image_and_p0_identities() -> None:
    module = load_renderer()
    assert module.RECIPE_CANDIDATE == "3a93787095dc9d722d9116998bfa9d7a9c517815"
    assert module.IMAGE.endswith("@sha256:d380888dc8a10796c7f841e341bd775c2d6500ede539f4ea16bb7bf0de92665d")
    assert module.MODEL_SHA256 == "ff8ff12d311bcc862247bd1d13f4380ec53f8af87095b183cf393147222d94b0"
    assert module.DATA_SHA256 == "88d3accf25f54933b5776bfb0a4c07f5719a25199abc0ed800ccfc68eae15d66"
    assert module.SCORER_SHA256 == "6fc2364da021bc5d14e1e3e8788d52cd49a3036088cacbb96d4eb5535e4473e5"


def test_renderer_requires_all_candidate_bound_gate_evidence() -> None:
    text = RENDERER.read_text()
    for gate in ("p0-evidence", "p1-evidence", "full-ci-evidence", "review-evidence"):
        assert gate in text
    assert '"findings": []' in text
    assert '"full_gpu_submission_allowed": True' in text
    assert '"TOTAL_TRAINING_STEPS": "160"' in text
    assert '"TOTAL_EPOCHS": "3"' in text


def full_ci_payload(module, root_candidate: str) -> dict:
    return {
        "evidence_kind": "root_full_ci_pass",
        "root_candidate_sha": root_candidate,
        "recipe_candidate_sha": module.RECIPE_CANDIDATE,
        "status": "passed",
    }


def parity_payload(module, root_candidate: str) -> dict:
    digest = "a" * 64
    result = {
        "tests": 10,
        "passed": 8,
        "failed": 1,
        "skipped": 1,
        "log_sha256": digest,
        "junit_sha256": digest,
    }
    profile = {
        "base": result,
        "candidate": result,
        "failure_set_sha256": digest,
        "candidate_only_tests": 0,
        "candidate_new_failures": 0,
        "base_only_tests": 0,
        "base_failures_resolved": 0,
        "shared_failure_detail_changes": 0,
    }
    return {
        "schema_version": 1,
        "evidence_kind": "root_full_ci_base_candidate_parity",
        "repository_full_name": "AlexJJ009/verl-v0.7",
        "base": {"root_sha": module.ROOT_BASE, "recipe_sha": module.RECIPE_BASE},
        "candidate": {"root_sha": root_candidate, "recipe_sha": module.RECIPE_CANDIDATE},
        "runtime": {
            "image": module.IMAGE,
            "image_id": module.PARITY_IMAGE_ID,
            "launcher": module.PARITY_LAUNCHER,
            "launcher_sha256": module.PARITY_LAUNCHER_SHA256,
            "payload": module.PARITY_PAYLOAD,
            "repository_mount": module.PARITY_REPOSITORY_MOUNT,
        },
        "default_profile": profile,
        "a800_dev_profile": profile,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_ci_admission_accepts_full_ci_pass(tmp_path: Path) -> None:
    module = load_renderer()
    root_candidate = "1" * 40
    evidence = tmp_path / "full-ci.json"
    write_json(evidence, full_ci_payload(module, root_candidate))
    _, _, mode = module.load_ci_admission_evidence(
        evidence,
        root_candidate=root_candidate,
        recipe_candidate=module.RECIPE_CANDIDATE,
    )
    assert mode == "full_ci_pass"


def test_ci_admission_accepts_exact_base_relative_parity(tmp_path: Path) -> None:
    module = load_renderer()
    root_candidate = "2" * 40
    evidence = tmp_path / "parity.json"
    write_json(evidence, parity_payload(module, root_candidate))
    _, _, mode = module.load_ci_admission_evidence(
        evidence,
        root_candidate=root_candidate,
        recipe_candidate=module.RECIPE_CANDIDATE,
    )
    assert mode == "base_relative_parity"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("candidate", "root_sha"), "3" * 40),
        (("runtime", "payload"), "python -m pytest -q tests"),
        (("default_profile", "candidate_new_failures"), 1),
        (("a800_dev_profile", "shared_failure_detail_changes"), 1),
    ],
)
def test_ci_admission_rejects_parity_identity_or_regression_drift(
    tmp_path: Path,
    path: tuple[str, str],
    value: object,
) -> None:
    module = load_renderer()
    root_candidate = "2" * 40
    payload = parity_payload(module, root_candidate)
    payload[path[0]][path[1]] = value
    evidence = tmp_path / "parity.json"
    write_json(evidence, payload)
    with pytest.raises(SystemExit):
        module.load_ci_admission_evidence(
            evidence,
            root_candidate=root_candidate,
            recipe_candidate=module.RECIPE_CANDIDATE,
        )


def test_launcher_shim_only_translates_admitted_external_outputs() -> None:
    text = SHIM.read_text()
    assert "artifact_output_root=/data_storage/yl_test/lgx/artifacts/verl/outputs" in text
    assert "/data-1/outputs/" in text
    for variable in ("BASE_CKPT_DIR", "LOG_DIR", "WANDB_DIR", "GRPO_ADMISSION_RECEIPT"):
        assert variable in text
    assert 'cd -- "$1"' in text
    assert "exec bash /workspace/verl/recipe/on_policy_wdl_sft/standard_grpo/run_math_stage1_grpo.sh" in text
    assert '"${GON35_CONTAINER_OUTPUT_ROOT}"' in text
    assert "only admits the exact Math Stage1 GRPO entry" in text
    assert 'canonical_host_output=$(realpath -e -- "${GON35_HOST_OUTPUT_ROOT}")' in text
    assert 'expected_container_output="/data-1/outputs/${run_leaf}"' in text
    assert '"${GON35_CONTAINER_OUTPUT_ROOT}" == "${expected_container_output}"' in text
    assert 'mkdir -p -- "${GON35_HOST_OUTPUT_ROOT}/cache/${cache_dir}"' in text
    assert 'TRITON_CACHE_DIR="$1/cache/bootstrap/triton"' in text
    assert 'TORCHINDUCTOR_CACHE_DIR="$1/cache/bootstrap/torchinductor"' in text
    assert 'GON35_COMPILER_CACHE_ROOT="$1/cache/processes"' in text
    assert "/workspace/verl/scripts/a800/gon35-python-startup" in text
    assert 'MPLCONFIGDIR="$1/cache/matplotlib"' in text
    assert "GRPO_EXPECTED_LAUNCHER_SHA256" in text
    assert "sha256sum" in text
    assert "pueue " not in text.lower()
    assert "slurm" not in text.lower()


def test_compiler_cache_namespace_changes_after_fork(tmp_path: Path) -> None:
    code = """
import json
import os

read_fd, write_fd = os.pipe()
child_pid = os.fork()
if child_pid == 0:
    os.close(read_fd)
    payload = [os.environ["TRITON_CACHE_DIR"], os.environ["TORCHINDUCTOR_CACHE_DIR"]]
    os.write(write_fd, json.dumps(payload).encode())
    os.close(write_fd)
    os._exit(0)

os.close(write_fd)
child = json.loads(os.read(read_fd, 8192))
os.close(read_fd)
os.waitpid(child_pid, 0)
parent = [os.environ["TRITON_CACHE_DIR"], os.environ["TORCHINDUCTOR_CACHE_DIR"]]
print(json.dumps({"parent": parent, "child": child}))
"""
    env = os.environ.copy()
    env["GON35_COMPILER_CACHE_ROOT"] = str(tmp_path / "compiler")
    env["PYTHONPATH"] = str(PYTHON_STARTUP)
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(result.stdout)
    assert payload["parent"] != payload["child"]
    for paths in payload.values():
        assert paths[0].endswith("/triton")
        assert paths[1].endswith("/torchinductor")
        assert Path(paths[0]).is_dir()
        assert Path(paths[1]).is_dir()


def test_compiler_cache_namespace_fails_closed(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("blocked")
    env = os.environ.copy()
    env["GON35_COMPILER_CACHE_ROOT"] = str(blocker / "compiler")
    env["PYTHONPATH"] = str(PYTHON_STARTUP)
    result = subprocess.run(
        [sys.executable, "-c", "raise AssertionError('application code must not run')"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 70
    assert "compiler-cache isolation failed" in result.stderr
    assert "application code must not run" not in result.stderr
