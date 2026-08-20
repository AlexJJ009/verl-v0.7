from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "recipe/on_policy_wdl_sft/_common_wdl_sft_is_joint.sh"


def test_partial_joint_cache_is_removed_before_prepare() -> None:
    text = LAUNCHER.read_text()
    assert "joint_cache_complete()" in text
    assert '[ -f "$MODEL_PATH/config.json" ]' in text
    assert '[ -f "$MODEL_PATH/model.safetensors" ]' in text
    assert '[ -f "$MODEL_PATH/model.safetensors.index.json" ]' in text
    assert 'if [ -d "$MODEL_PATH" ] && ! joint_cache_complete; then' in text
    assert 'rm -rf -- "$MODEL_PATH"' in text
    assert "if ! joint_cache_complete; then" in text


def test_joint_cache_basename_has_a_dynamic_module_length_gate() -> None:
    text = LAUNCHER.read_text()
    assert "joint_module_name=${joint_module_name//-/_hyphen_}" in text
    assert 'if [ "${#joint_module_name}" -gt 180 ]; then' in text
    assert "Set MODEL_PATH to a shorter, run-unique cache path." in text


def test_validation_only_mode_is_explicit_and_defaults_off() -> None:
    text = LAUNCHER.read_text()
    assert "val_only=${VAL_ONLY:-False}" in text
    assert "trainer.val_only=${val_only}" in text
    assert "+trainer.val_only=${val_only}" not in text
