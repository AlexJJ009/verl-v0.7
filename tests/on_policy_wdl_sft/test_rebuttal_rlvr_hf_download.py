from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def load_script(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DOWNLOAD = load_script(
    "download_rebuttal_rlvr_hf_dataset",
    "scripts/download_rebuttal_rlvr_hf_dataset.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_remote_tree(root: Path) -> dict[str, str]:
    (root / "metadata").mkdir(parents=True)
    (root / "data/math").mkdir(parents=True)
    (root / ".gitattributes").write_text("*.parquet filter=lfs\n")
    (root / "README.md").write_text("readme\n")
    (root / "metadata/publication_inventory.json").write_text('{"schema_version": 1}\n')
    (root / "data/math/train.parquet").write_bytes(b"parquet")
    checksums = {}
    for relative in (
        ".gitattributes",
        "README.md",
        "data/math/train.parquet",
        "metadata/publication_inventory.json",
    ):
        checksums[relative] = digest(root / relative)
    (root / "metadata/checksums.sha256").write_text(
        "".join(f"{value}  {relative}\n" for relative, value in sorted(checksums.items()))
    )
    checksums["metadata/checksums.sha256"] = digest(root / "metadata/checksums.sha256")
    return checksums


def pin_test_tree(monkeypatch: pytest.MonkeyPatch, source: Path) -> dict[str, str]:
    checksums = make_remote_tree(source)
    monkeypatch.setattr(DOWNLOAD, "REVIEWED_CHECKSUMS_SHA256", checksums["metadata/checksums.sha256"])
    monkeypatch.setattr(
        DOWNLOAD,
        "REVIEWED_PRIVATE_INVENTORY_SHA256",
        checksums["metadata/publication_inventory.json"],
    )
    return checksums


def safe_route() -> dict[str, object]:
    return {
        "schema_version": 2,
        "endpoint": "https://huggingface.co",
        "proxy_url": "http://127.0.0.1:7890",
        "route_group": "大流量",
        "route_group_type": "Selector",
        "runtime_group_type": "select",
        "selected_leaf_sha256": "a" * 64,
        "selected_namespace": "BW",
        "selected_region": "Hong Kong",
        "selected_residential": False,
        "selected_protocol": "anytls",
        "controller_protocol": "AnyTLS",
        "selector_projection_verified": True,
        "runtime_group_identity_verified": True,
        "runtime_proxy_identity_verified": True,
        "required_hosts_verified": list(DOWNLOAD.HF_ROUTE_REQUIRED_HOSTS),
        "connection_hosts_verified": [],
    }


def configure_private_hf_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir(mode=0o700)
    token = hf_home / "token"
    token.write_text("hf_test_token\n")
    token.chmod(0o600)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    for name in DOWNLOAD.AMBIENT_TOKEN_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    return hf_home


def test_verify_download_accepts_exact_pinned_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "download"
    checksums = pin_test_tree(monkeypatch, root)
    assert DOWNLOAD.verify_download(root) == checksums


def test_verify_download_rejects_unexpected_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "download"
    pin_test_tree(monkeypatch, root)
    (root / "stale.bin").write_bytes(b"legacy")
    with pytest.raises(DOWNLOAD.DownloadError, match="allowlist mismatch"):
        DOWNLOAD.verify_download(root)


def test_route_change_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    changed = safe_route()
    changed["selected_leaf_sha256"] = "b" * 64
    monkeypatch.setattr(DOWNLOAD, "admit_hf_network", lambda **kwargs: changed)
    with pytest.raises(DOWNLOAD.DownloadError, match="selector changed"):
        DOWNLOAD.admit_same_route("a" * 64)


def test_invalid_target_is_rejected_before_download(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    args = SimpleNamespace(
        repo_id=DOWNLOAD.DEFAULT_REPO_ID,
        revision=DOWNLOAD.VERIFIED_DATASET_COMMIT,
        local_dir=existing,
        receipt=tmp_path / "receipt.json",
    )
    with pytest.raises(DOWNLOAD.DownloadError, match="already exists"):
        DOWNLOAD.validate_args(args)


def test_endpoint_override_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.invalid")
    args = SimpleNamespace(
        repo_id=DOWNLOAD.DEFAULT_REPO_ID,
        revision=DOWNLOAD.VERIFIED_DATASET_COMMIT,
        local_dir=tmp_path / "download",
        receipt=tmp_path / "receipt.json",
    )
    with pytest.raises(DOWNLOAD.DownloadError, match="HF_ENDPOINT override"):
        DOWNLOAD.validate_args(args)


def test_downloader_rejects_drifted_endpoint_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DOWNLOAD, "HF_ENDPOINT", "https://mirror.invalid")
    with pytest.raises(DOWNLOAD.DownloadError, match="endpoint pin drifted"):
        DOWNLOAD.admit_same_route()


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("schema_version", 1),
        ("endpoint", "https://mirror.invalid"),
        ("proxy_url", "http://127.0.0.1:9999"),
        ("route_group", "AI-工具"),
        ("route_group_type", "Fallback"),
        ("runtime_group_type", "url-test"),
        ("selected_leaf_sha256", "short"),
        ("selected_namespace", "other"),
        ("selected_region", "Taiwan"),
        ("selected_residential", True),
        ("selected_protocol", ""),
        ("controller_protocol", "MysteryProtocol"),
        ("selector_projection_verified", False),
        ("runtime_group_identity_verified", False),
        ("runtime_proxy_identity_verified", False),
        ("required_hosts_verified", ["huggingface.co"]),
        ("connection_hosts_verified", ["HUGGINGFACE.CO"]),
    ],
)
def test_downloader_rejects_incomplete_route_admission(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    invalid: object,
) -> None:
    admission = safe_route()
    admission[field] = invalid
    monkeypatch.setattr(DOWNLOAD, "admit_hf_network", lambda **kwargs: admission)
    with pytest.raises(DOWNLOAD.RouteAdmissionError, match="complete verified route admission"):
        DOWNLOAD.admit_same_route()


def test_private_mode_rejects_ambient_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_private_hf_home(tmp_path, monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "must-not-be-used")
    args = SimpleNamespace(
        repo_id=DOWNLOAD.DEFAULT_REPO_ID,
        revision=DOWNLOAD.VERIFIED_DATASET_COMMIT,
        local_dir=tmp_path / "download",
        receipt=tmp_path / "receipt.json",
    )
    with pytest.raises(DOWNLOAD.DownloadError, match="ambient Hugging Face token"):
        DOWNLOAD.validate_args(args)


def test_download_gates_every_hub_call_and_writes_verified_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "remote"
    checksums = pin_test_tree(monkeypatch, source)
    configure_private_hf_home(tmp_path, monkeypatch)
    events: list[str] = []

    def route(**kwargs):
        events.append("admit")
        result = safe_route()
        if kwargs:
            assert kwargs == {"connection_hosts": ("huggingface.co",)}
            result["connection_hosts_verified"] = ["huggingface.co"]
        return result

    monkeypatch.setattr(DOWNLOAD, "admit_hf_network", route)
    monkeypatch.setattr(
        DOWNLOAD,
        "configure_hf_http_observer",
        lambda admission: SimpleNamespace(observed_hosts=["huggingface.co"]),
    )

    def fake_download(repo_id, relative, *, repo_type, revision, local_dir, endpoint, token):
        events.append(f"hub:download:{relative}")
        assert repo_id == DOWNLOAD.DEFAULT_REPO_ID
        assert repo_type == "dataset"
        assert revision == DOWNLOAD.VERIFIED_DATASET_COMMIT
        assert endpoint == "https://huggingface.co"
        assert token == "hf_test_token"
        destination = Path(local_dir) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
        cache = Path(local_dir) / ".cache/huggingface"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "generated").write_text("cache")
        return str(destination)

    class FakeApi:
        def __init__(self, *, endpoint, token):
            assert endpoint == "https://huggingface.co"
            assert token == "hf_test_token"

        def repo_info(self, repo_id, *, repo_type, revision, files_metadata, timeout):
            events.append("hub:repo_info")
            assert files_metadata is True
            assert timeout == 30
            return SimpleNamespace(
                sha=revision,
                private=True,
                gated=False,
                siblings=[SimpleNamespace(rfilename=relative) for relative in sorted(checksums)],
            )

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_download, HfApi=FakeApi),
    )
    local_dir = tmp_path / "final"
    receipt = tmp_path / "state/download.json"
    result = DOWNLOAD.download(
        SimpleNamespace(
            repo_id=DOWNLOAD.DEFAULT_REPO_ID,
            revision=DOWNLOAD.VERIFIED_DATASET_COMMIT,
            local_dir=local_dir,
            receipt=receipt,
        )
    )
    expected_events = [
        "admit",
        "admit",
        f"hub:download:{DOWNLOAD.MANIFEST_PATH}",
        "admit",
        "admit",
        "hub:repo_info",
        "admit",
    ]
    for relative in sorted(name for name in checksums if name != DOWNLOAD.MANIFEST_PATH):
        expected_events.extend(["admit", f"hub:download:{relative}", "admit"])
    expected_events.append("admit")
    assert events == expected_events
    assert result["schema_version"] == 2
    assert result["revision"] == DOWNLOAD.VERIFIED_DATASET_COMMIT
    assert result["remote_private"] is True
    assert result["route_admission"]["selected_residential"] is False
    assert result["route_admission"]["connection_hosts_verified"] == ["huggingface.co"]
    assert result["observed_connection_hosts"] == ["huggingface.co"]
    assert result["authentication_mode"] == "explicit_operator_hf_home_token_private_only"
    assert json.loads(receipt.read_text())["file_count"] == len(checksums)
    assert not (local_dir / ".cache").exists()
    assert DOWNLOAD.verify_download(local_dir) == checksums


def test_failed_initial_route_makes_no_hub_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_private_hf_home(tmp_path, monkeypatch)
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("Hub call should not occur")

    monkeypatch.setattr(
        DOWNLOAD,
        "admit_hf_network",
        lambda **kwargs: (_ for _ in ()).throw(DOWNLOAD.RouteAdmissionError("route")),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=forbidden, HfApi=forbidden),
    )
    with pytest.raises(DOWNLOAD.RouteAdmissionError, match="route"):
        DOWNLOAD.download(
            SimpleNamespace(
                repo_id=DOWNLOAD.DEFAULT_REPO_ID,
                revision=DOWNLOAD.VERIFIED_DATASET_COMMIT,
                local_dir=tmp_path / "final",
                receipt=tmp_path / "receipt.json",
            )
        )
    assert called is False


def test_private_bundle_visibility_gate_rejects_public_state() -> None:
    info = SimpleNamespace(private=False, gated=False)
    with pytest.raises(DOWNLOAD.DownloadError, match="unexpectedly public"):
        DOWNLOAD.validate_remote_visibility(info)


def test_failed_download_still_runs_post_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    def admit(expected_leaf=None):
        events.append("admit")
        return safe_route()

    def fail() -> None:
        events.append("hub")
        raise RuntimeError("download failed")

    monkeypatch.setattr(DOWNLOAD, "admit_same_route", admit)
    with pytest.raises(RuntimeError, match="download failed"):
        DOWNLOAD.guarded_hub_call("a" * 64, fail)
    assert events == ["admit", "hub", "admit"]


def test_operation_failure_and_route_drift_reports_both_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def admit(expected_leaf=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise DOWNLOAD.DownloadError("selector changed")
        return safe_route()

    monkeypatch.setattr(DOWNLOAD, "admit_same_route", admit)
    with pytest.raises(DOWNLOAD.DownloadError, match="operation=RuntimeError; route=DownloadError"):
        DOWNLOAD.guarded_hub_call(
            "a" * 64,
            lambda: (_ for _ in ()).throw(RuntimeError("request failed")),
        )


def test_cli_rejects_anonymous_option(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_rebuttal_rlvr_hf_dataset.py",
            "--local-dir",
            str(tmp_path / "data"),
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--anonymous",
        ],
    )
    with pytest.raises(SystemExit) as error:
        DOWNLOAD.parse_args()
    assert error.value.code == 2


def test_checked_in_private_receipt_pins_guarded_download_and_blocks_public() -> None:
    receipt = json.loads(
        (REPO_ROOT / "docs/joint_training/reports/data/rebuttal_rlvr_hf_private_receipt_20260729.json").read_text()
    )
    assert receipt["repository"]["head"] == DOWNLOAD.VERIFIED_DATASET_COMMIT
    assert receipt["repository"]["private"] is True
    assert receipt["network_route"]["endpoint"] == "https://huggingface.co"
    assert receipt["network_route"]["schema_version"] == 2
    assert receipt["network_route"]["route_group"] == "大流量"
    assert receipt["network_route"]["route_group_type"] == "Selector"
    assert receipt["network_route"]["runtime_group_type"] == "select"
    assert receipt["network_route"]["selected_namespace"] == "BW"
    assert receipt["network_route"]["selected_region"] == "Hong Kong"
    assert receipt["network_route"]["selected_residential"] is False
    assert receipt["network_route"]["selected_protocol"] == "anytls"
    assert receipt["network_route"]["controller_protocol"] == "AnyTLS"
    assert receipt["network_route"]["selector_projection_verified"] is True
    assert receipt["network_route"]["runtime_group_identity_verified"] is True
    assert receipt["network_route"]["runtime_proxy_identity_verified"] is True
    assert receipt["network_route"]["request_boundary_admission_verified"] is True
    assert receipt["network_route"]["redirect_hostname_admission_verified"] is True
    assert receipt["network_route"]["connection_hosts_verified"]
    assert receipt["network_route"]["shared_selector_mid_request_switch_risk_disclosed"] is True
    assert receipt["network_route"]["guarded_download_verified"] is True
    assert receipt["publication"]["public_transition_complete"] is False


def test_handoff_keeps_v3_guarded_downloader_separate_from_public_v4_cli() -> None:
    guide = (REPO_ROOT / "docs/joint_training/guides/rebuttal_rlvr_hf_dataset_handoff.md").read_text()
    assert "scripts/download_rebuttal_rlvr_hf_dataset.py" in guide
    assert f"--revision {DOWNLOAD.VERIFIED_DATASET_COMMIT}" in guide
    assert "An authorized consumer can use the guarded wrapper now" in guide
    assert "bare `hf download`" in guide
    assert 'HF_HOME="$HF_AUTH_HOME" hf auth login' not in guide
    assert "without making an unguarded\n`hf auth login` API call" in guide
    assert "--anonymous" not in guide
    assert "standard credential-free `hf download`" in guide
    assert "`validate_dataset.py`" in guide
    assert "remains private-v3-only" in guide
    assert "private_handoff_only" in guide


def test_meituan_checkout_requires_both_immutable_source_pins() -> None:
    guide = (REPO_ROOT / "docs/joint_training/guides/meituan_rlvr_image_build.md").read_text()
    assert "${EXPECTED_REPO_COMMIT:?" in guide
    assert "${EXPECTED_RECIPE_COMMIT:?" in guide
    assert 'checkout --detach "$EXPECTED_REPO_COMMIT"' in guide
    assert 'test "$REPO_COMMIT" = "$EXPECTED_REPO_COMMIT"' in guide
    assert 'test "$RECIPE_COMMIT" = "$EXPECTED_RECIPE_COMMIT"' in guide
