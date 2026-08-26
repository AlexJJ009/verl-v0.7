import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RENDERER = ROOT / "scripts/a800/render_gon35_grpo_admission.py"
SHIM = ROOT / "scripts/a800/gon35-bin/verl-dev-run"


def load_renderer():
    spec = importlib.util.spec_from_file_location("gon35_admission", RENDERER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_renderer_pins_recipe_image_and_p0_identities() -> None:
    module = load_renderer()
    assert module.RECIPE_CANDIDATE == "cb677ebded6558875949d10d8a79af9356cb681d"
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


def test_launcher_shim_only_translates_admitted_external_outputs() -> None:
    text = SHIM.read_text()
    assert "/data_storage/yl_test/lgx/artifacts/verl/outputs/" in text
    assert "/data-1/outputs/" in text
    for variable in ("BASE_CKPT_DIR", "LOG_DIR", "WANDB_DIR", "GRPO_ADMISSION_RECEIPT"):
        assert variable in text
    assert 'exec "${launcher}" "$@"' in text
    assert "pueue " not in text.lower()
    assert "slurm" not in text.lower()
