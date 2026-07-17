from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_environment_distinguishes_no_kl_and_model2_kl():
    module = load_module("stage123_matrix_memory_probe", ROOT / "scripts/run_stage123_matrix_memory_probe.py")
    base = {
        "phase": "stage2",
        "train_file": "/train.parquet",
        "source": {"checkpoint_root": "/checkpoints/source", "model2_path": "/models/model2", "run_prefix": "SOURCE", "handoff_step": 40},
    }
    no_kl = module.run_environment({**base, "submodel_kl": {"enabled": False, "model1_enabled": False, "model1_coef": 0.0, "model2_enabled": False, "model2_coef": 0.0}})
    model2_kl = module.run_environment({**base, "submodel_kl": {"enabled": True, "model1_enabled": False, "model1_coef": 0.0, "model2_enabled": True, "model2_coef": 0.01, "model2_ref_path": "/models/ref"}})
    assert no_kl["SUBMODEL_KL_MODEL2_ENABLED"] == "false"
    assert model2_kl["SUBMODEL_KL_MODEL2_ENABLED"] == "true"
    assert model2_kl["SUBMODEL_KL_MODEL2_REF_PATH"] == "/models/ref"
    assert model2_kl["STAGE123_EXPECTED_VAL_N"] == "3"


def test_summary_requires_auditable_gpu_headroom():
    module = load_module("stage123_matrix_memory_probe_summary", ROOT / "scripts/run_stage123_matrix_memory_probe.py")
    run = {"id": "arm", "submodel_kl": {"enabled": True}}
    repetition = {"status": "passed", "resources": {"peak_gpu_memory_used_mib": 43000, "per_gpu_memory": [{"index": 0, "total_memory_mib": 46068}]}}
    assert module.summarize(run, [repetition], 4096)["status"] == "failed"
    assert module.summarize(run, [repetition], 2048)["status"] == "passed"
