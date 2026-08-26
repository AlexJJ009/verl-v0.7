from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = ROOT / "scripts/a800/bootstrap_pueue.sh"


def test_bootstrap_pins_release_and_checksums() -> None:
    text = BOOTSTRAP.read_text()
    assert 'PUEUE_VERSION="4.0.4"' in text
    assert "c1b10d7e4e62211075ddd0e1dc3e8cbfc5a43d662cb3be7402a28504e23fcb51" in text
    assert "5afeff6adbafb909e8d54e2caff158e6966c2adffa2c09e60fd631cc51b60390" in text
    assert "sha256sum --check --status" in text


def test_bootstrap_keeps_mutable_state_external() -> None:
    text = BOOTSTRAP.read_text()
    assert "/data_storage/yl_test/lgx/runtime/verl/pueue" not in text
    assert 'PUEUE_ROOT="${VERL_A800_PUEUE_ROOT:-${OPERATION_ROOT}/runtime/verl/pueue}"' in text
    assert "mutable Pueue path escapes the A800 operation root" in text
    assert "pueue.env" in text


def test_bootstrap_uses_native_pueue_group_with_single_concurrency() -> None:
    text = BOOTSTRAP.read_text()
    assert 'group add gpu8 --parallel 1' in text
    assert 'parallel --group gpu8 1' in text
    assert 'daemon_session="verl-a800-pueued"' in text
    assert 'tmux new-session -d -s "${daemon_session}"' in text
    assert "docker" not in text.lower()
    assert "slurm" not in text.lower()
