import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_policy_module():
    path = ROOT / "scripts/check_sft_loss_mask_policy.py"
    spec = importlib.util.spec_from_file_location("check_sft_loss_mask_policy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_manifest(path: Path, *, status: str, launch_allowed: bool, v3: bool = False):
    payload = {
        "status": status,
        "execution": {"launch_allowed": launch_allowed},
        "paths": {},
    }
    if v3:
        payload["execution"]["requires_whole_message_loss_mask"] = True
        payload["paths"]["loss_mask_preflight_receipt"] = "/tmp/loss-mask.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload))


def build_policy_tree(tmp_path: Path):
    allowlist = tmp_path / "tests/special_sanity/sft_input_ids_mismatch_allowlist.json"
    allowlist.parent.mkdir(parents=True)
    allowlist.write_text(json.dumps({"schema_version": 1, "entries": []}))
    launcher = tmp_path / "recipe/on_policy_wdl_sft/format_cold_start/run_sft_math_qwen3_1p7b_format.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text('"data.tokenize_whole_message=True"\n"data.ignore_input_ids_mismatch=False"\n')
    manifest_root = tmp_path / "recipe/on_policy_wdl_sft/experiment_manifest"
    write_manifest(
        manifest_root / "math_qwen3_1p7b_cold_start_cotmask_v3.yaml",
        status="implementation_validated_launch_not_started",
        launch_allowed=False,
        v3=True,
    )
    for name in (
        "math_qwen3_1p7b_cold_start.yaml",
        "math_qwen3_1p7b_cold_start_lr5e6_v2.yaml",
        "math_qwen3_1p7b_stage123_lr5e6_v2.yaml",
    ):
        write_manifest(manifest_root / name, status="invalidated_test", launch_allowed=False)


def test_policy_gate_passes_current_repository():
    module = load_policy_module()
    assert module.check() == []


def test_policy_gate_canary_rejects_new_warning_suppression(tmp_path):
    module = load_policy_module()
    build_policy_tree(tmp_path)
    bad_launcher = tmp_path / "recipe/new_experiment/run_sft_bad.sh"
    bad_launcher.parent.mkdir(parents=True)
    bad_launcher.write_text("data.ignore_input_ids_mismatch=True\n")
    failures = module.check(tmp_path)
    assert any("inventory changed" in failure for failure in failures)


def test_policy_gate_canary_rejects_missing_whole_message_mode(tmp_path):
    module = load_policy_module()
    build_policy_tree(tmp_path)
    launcher = tmp_path / "recipe/on_policy_wdl_sft/format_cold_start/run_sft_math_qwen3_1p7b_format.sh"
    launcher.write_text('"data.ignore_input_ids_mismatch=False"\n')
    failures = module.check(tmp_path)
    assert any("data.tokenize_whole_message=True" in failure for failure in failures)
