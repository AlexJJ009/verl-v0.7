from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = load_script("prepare_rebuttal_rlvr_public_dataset", "scripts/prepare_rebuttal_rlvr_public_dataset.py")
PUBLISH = load_script("publish_rebuttal_rlvr_public_dataset", "scripts/publish_rebuttal_rlvr_public_dataset.py")


def test_publish_defaults_to_owner_controlled_dataset_repo() -> None:
    assert PUBLISH.DEFAULT_REPO_ID == "AlexGeek/RLdataset"


def test_repo_snapshot_records_git_and_lfs_content_ids() -> None:
    info = SimpleNamespace(
        id="AlexGeek/RLdataset",
        sha="d" * 40,
        private=True,
        gated=False,
        last_modified=None,
        siblings=[
            SimpleNamespace(
                rfilename="data/example.parquet",
                size=123,
                blob_id="b" * 40,
                lfs=SimpleNamespace(sha256="a" * 64, size=123, pointer_size=130),
            ),
            SimpleNamespace(
                rfilename="README.md",
                size=12,
                blob_id="c" * 40,
                lfs=None,
            ),
        ],
    )
    snapshot = PUBLISH.repo_snapshot(info)
    assert snapshot["files"] == [
        {
            "path": "README.md",
            "size": 12,
            "blob_id": "c" * 40,
            "lfs": None,
        },
        {
            "path": "data/example.parquet",
            "size": 123,
            "blob_id": "b" * 40,
            "lfs": {"sha256": "a" * 64, "size": 123, "pointer_size": 130},
        },
    ]


def test_reviewed_bundle_gate_rejects_self_relabelled_inventory(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    PUBLISH.load_and_verify_bundle(bundle)
    with pytest.raises(PUBLISH.PublishError, match="reviewed full-scope inventory"):
        PUBLISH.verify_reviewed_bundle(bundle)


def test_hf_proxy_is_forced_in_upper_and_lower_case(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(name, "http://bypass.invalid:1")
    monkeypatch.setenv("NO_PROXY", "huggingface.co")
    PUBLISH.enforce_hf_proxy()
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        assert os.environ[name] == "http://127.0.0.1:7890"
    assert "huggingface.co" not in os.environ["NO_PROXY"]
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert os.environ["HF_HUB_DISABLE_HF_TRANSFER"] == "1"
    assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "0"
    assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "30"
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "120"


def test_hf_proxy_rejects_nonofficial_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.invalid")
    with pytest.raises(PUBLISH.PublishError, match="endpoint override is forbidden"):
        PUBLISH.enforce_hf_proxy()


def test_publisher_rejects_drifted_endpoint_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PUBLISH, "HF_ENDPOINT", "https://mirror.invalid")
    with pytest.raises(PUBLISH.PublishError, match="endpoint pin drifted"):
        PUBLISH.admit_hf_network()


def admitted_network() -> dict:
    return {
        "schema_version": 2,
        "endpoint": "https://huggingface.co",
        "proxy_url": PUBLISH.PROXY_URL,
        "route_group": PUBLISH.HF_ROUTE_GROUP,
        "route_group_type": "Selector",
        "runtime_group_type": "select",
        "selected_leaf_sha256": "a" * 64,
        "selected_namespace": "BW",
        "selected_region": "Hong Kong",
        "selected_residential": False,
        "selected_protocol": "ss",
        "controller_protocol": "Shadowsocks",
        "selector_projection_verified": True,
        "runtime_group_identity_verified": True,
        "runtime_proxy_identity_verified": True,
        "required_hosts_verified": list(PUBLISH.HF_ROUTE_REQUIRED_HOSTS),
        "connection_hosts_verified": [],
    }


def write_mihomo_test_config(
    tmp_path: Path,
    selected: str = "[BW] 🇭🇰 香港_03",
    *,
    server: str = "198.51.100.1",
    runtime_type: str = "ss",
    group_members: list[str] | None = None,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "runtime.yaml"
    path.write_text(
        json.dumps(
            {
                "mixed-port": 7890,
                "external-controller": "127.0.0.1:9090",
                "secret": "test-controller-secret",
                "proxy-groups": [
                    {
                        "name": PUBLISH.HF_ROUTE_GROUP,
                        "type": "select",
                        "proxies": group_members or [selected],
                    }
                ],
                "proxies": [
                    {
                        "name": selected,
                        "type": runtime_type,
                        "server": server,
                        "port": 443,
                        "password": "test-only-secret",
                    }
                ],
            },
            ensure_ascii=False,
        )
    )
    return path


def route_fetcher(
    selected: str,
    *,
    misroute: str | None = None,
    leaf: dict | None = None,
    controller_members: list[str] | None = None,
):
    controller_leaf = leaf if leaf is not None else {
        "name": selected,
        "type": "Shadowsocks",
        "id": "leaf-id-1",
        "alive": True,
        "history": [],
    }

    def fetch(base_url: str, secret: str, path: str) -> dict:
        assert base_url == "http://127.0.0.1:9090"
        assert secret == "test-controller-secret"
        if path.startswith("/proxies/") and "%E5%A4%A7%E6%B5%81%E9%87%8F" in path:
            return {
                "type": "Selector",
                "now": selected,
                "all": controller_members or [selected],
            }
        if path.startswith("/proxies/"):
            return controller_leaf
        if path == "/providers/proxies":
            raise AssertionError("selector-backed inline proxies must not be treated as providers")
        if path == "/rules":
            return {
                "rules": [
                    {
                        "type": "DOMAIN",
                        "payload": host,
                        "proxy": "DIRECT" if host == misroute else PUBLISH.HF_ROUTE_GROUP,
                    }
                    for host in PUBLISH.HF_ROUTE_REQUIRED_HOSTS
                ]
            }
        raise AssertionError(path)

    return fetch


def make_httpx_test_backend(handler, observed_config: dict[str, object]):
    state: dict[str, object] = {}

    class AuditedTestClient(httpx.Client):
        def __init__(self, **kwargs):
            observed_config.update(kwargs)
            assert kwargs.pop("proxy") == PUBLISH.PROXY_URL
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    def set_client_factory(factory):
        state.clear()
        state["factory"] = factory

    def get_session():
        if "session" not in state:
            state["session"] = state["factory"]()
        return state["session"]

    hub = SimpleNamespace(
        __version__="1.8.0",
        set_client_factory=set_client_factory,
        get_session=get_session,
    )
    return hub, SimpleNamespace(Client=AuditedTestClient), get_session


def test_hf_route_admission_accepts_hong_kong_non_residential_leaf(tmp_path: Path) -> None:
    result = PUBLISH.validate_hf_route_admission(
        write_mihomo_test_config(tmp_path),
        fetch_json=route_fetcher("[BW] 🇭🇰 香港_03"),
    )
    assert result["schema_version"] == 2
    assert result["route_group"] == "大流量"
    assert result["route_group_type"] == "Selector"
    assert result["runtime_group_type"] == "select"
    assert result["selected_namespace"] == "BW"
    assert result["selected_protocol"] == "ss"
    assert result["controller_protocol"] == "Shadowsocks"
    assert result["selector_projection_verified"] is True
    assert result["selected_region"] == "Hong Kong"
    assert result["selected_residential"] is False
    assert result["required_hosts_verified"] == list(PUBLISH.HF_ROUTE_REQUIRED_HOSTS)


def test_hf_route_admission_rejects_residential_leaf(tmp_path: Path) -> None:
    with pytest.raises(PUBLISH.PublishError, match="non-residential"):
        PUBLISH.validate_hf_route_admission(
            write_mihomo_test_config(tmp_path, "[BW] 🇭🇰 香港_家宽"),
            fetch_json=route_fetcher("[BW] 🇭🇰 香港_家宽"),
        )


def test_hf_route_admission_rejects_misrouted_hf_host(tmp_path: Path) -> None:
    with pytest.raises(PUBLISH.PublishError, match="do not route"):
        PUBLISH.validate_hf_route_admission(
            write_mihomo_test_config(tmp_path),
            fetch_json=route_fetcher(
                "[BW] 🇭🇰 香港_03",
                misroute="cas-bridge.xethub.hf.co",
            ),
        )


def test_hf_route_admission_rejects_empty_leaf(tmp_path: Path) -> None:
    with pytest.raises(PUBLISH.PublishError, match="leaf identity"):
        PUBLISH.validate_hf_route_admission(
            write_mihomo_test_config(tmp_path),
            fetch_json=route_fetcher("[BW] 🇭🇰 香港_03", leaf={}),
        )


def test_hf_route_admission_requires_matching_runtime_selector_membership(tmp_path: Path) -> None:
    with pytest.raises(PUBLISH.PublishError, match="runtime 大流量 group"):
        PUBLISH.validate_hf_route_admission(
            write_mihomo_test_config(
                tmp_path,
                group_members=["[BW] 🇭🇰 香港_03", "[BW] 🇭🇰 香港_04"],
            ),
            fetch_json=route_fetcher("[BW] 🇭🇰 香港_03"),
        )


def test_hf_route_admission_rejects_unknown_direct_protocol(tmp_path: Path) -> None:
    selected = "[BW] 🇭🇰 香港_03"
    with pytest.raises(PUBLISH.PublishError, match="concrete remote proxy leaf"):
        PUBLISH.validate_hf_route_admission(
            write_mihomo_test_config(tmp_path, runtime_type="MysteryProtocol"),
            fetch_json=route_fetcher(
                selected,
                leaf={
                    "name": selected,
                    "type": "MysteryProtocol",
                    "id": "mystery-id",
                    "alive": True,
                },
            ),
        )


def test_hf_route_admission_accepts_current_anytls_protocol(tmp_path: Path) -> None:
    selected = "[BW] 🇭🇰 香港_03"
    result = PUBLISH.validate_hf_route_admission(
        write_mihomo_test_config(tmp_path, runtime_type="anytls"),
        fetch_json=route_fetcher(
            selected,
            leaf={
                "name": selected,
                "type": "AnyTLS",
                "id": "anytls-id",
                "alive": True,
            },
        ),
    )
    assert result["selected_protocol"] == "anytls"
    assert result["controller_protocol"] == "AnyTLS"


def test_leaf_fingerprint_changes_when_same_name_changes_server(tmp_path: Path) -> None:
    selected = "[BW] 🇭🇰 香港_03"
    first = PUBLISH.validate_hf_route_admission(
        write_mihomo_test_config(tmp_path / "first", selected, server="198.51.100.1"),
        fetch_json=route_fetcher(selected),
    )
    second = PUBLISH.validate_hf_route_admission(
        write_mihomo_test_config(tmp_path / "second", selected, server="203.0.113.9"),
        fetch_json=route_fetcher(selected),
    )
    assert first["selected_leaf_sha256"] != second["selected_leaf_sha256"]


def test_route_admission_rejects_rule_set_before_hf_rule(tmp_path: Path) -> None:
    selected = "[BW] 🇭🇰 香港_03"
    normal_fetch = route_fetcher(selected)

    def fetch(base_url: str, secret: str, path: str) -> dict:
        if path == "/rules":
            return {
                "rules": [
                    {"type": "RULE_SET", "payload": "unknown", "proxy": "DIRECT"},
                    {
                        "type": "DOMAIN-SUFFIX",
                        "payload": "huggingface.co",
                        "proxy": PUBLISH.HF_ROUTE_GROUP,
                    },
                ]
            }
        return normal_fetch(base_url, secret, path)

    with pytest.raises(PUBLISH.PublishError, match="unevaluable domain rule"):
        PUBLISH.validate_hf_route_admission(
            write_mihomo_test_config(tmp_path, selected),
            fetch_json=fetch,
        )


def test_dynamic_redirect_host_misroute_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PUBLISH.PublishError, match="do not route to 大流量"):
        PUBLISH.validate_hf_route_admission(
            write_mihomo_test_config(tmp_path),
            fetch_json=route_fetcher("[BW] 🇭🇰 香港_03"),
            connection_hosts=("redirect-cdn.example",),
        )


def test_httpx_request_hook_admits_actual_redirect_host_before_send() -> None:
    events: list[str] = []
    observed_config: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        events.append(f"send:{host}")
        if host == "huggingface.co":
            return httpx.Response(
                302,
                headers={"location": "https://cdn-lfs.huggingface.co/blob?token=redirect-secret"},
            )
        return httpx.Response(200, content=b"ok")

    hub, httpx_module, get_session = make_httpx_test_backend(handler, observed_config)

    def admit(*, connection_hosts=()):
        host = connection_hosts[0]
        events.append(f"admit:{host}")
        result = admitted_network()
        result["connection_hosts_verified"] = [host]
        return result

    observer = PUBLISH.configure_hf_http_observer(
        admitted_network(),
        admit=admit,
        hub_module=hub,
        httpx_module=httpx_module,
        hub_request_hook=lambda request: events.append(f"hf-hook:{request.url.host}"),
    )
    response = get_session().get("https://huggingface.co/start?token=origin-secret")
    assert response.status_code == 200
    assert events == [
        "admit:huggingface.co",
        "hf-hook:huggingface.co",
        "send:huggingface.co",
        "admit:huggingface.co",
        "admit:cdn-lfs.huggingface.co",
        "hf-hook:cdn-lfs.huggingface.co",
        "send:cdn-lfs.huggingface.co",
        "admit:cdn-lfs.huggingface.co",
    ]
    assert observer.observed_hosts == ["cdn-lfs.huggingface.co", "huggingface.co"]
    assert observed_config["trust_env"] is False
    assert observed_config["follow_redirects"] is True
    assert observed_config["timeout"] is None


def test_dynamic_redirect_host_misroute_blocks_before_send() -> None:
    events: list[str] = []
    observed_config: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        events.append(f"send:{request.url.host}")
        if request.url.host == "huggingface.co":
            return httpx.Response(302, headers={"location": "https://misrouted.example/blob"})
        return httpx.Response(200)

    hub, httpx_module, get_session = make_httpx_test_backend(handler, observed_config)

    def admit(*, connection_hosts=()):
        host = connection_hosts[0]
        events.append(f"admit:{host}")
        if host == "misrouted.example":
            raise PUBLISH.RouteAdmissionError("dynamic host does not route to 大流量")
        result = admitted_network()
        result["connection_hosts_verified"] = [host]
        return result

    PUBLISH.configure_hf_http_observer(
        admitted_network(),
        admit=admit,
        hub_module=hub,
        httpx_module=httpx_module,
        hub_request_hook=lambda request: None,
    )
    with pytest.raises(PUBLISH.RouteAdmissionError, match="does not route"):
        get_session().get("https://huggingface.co/start")
    assert events == [
        "admit:huggingface.co",
        "send:huggingface.co",
        "admit:huggingface.co",
        "admit:misrouted.example",
    ]


def test_observer_records_hostname_without_query_credentials() -> None:
    observed_config: dict[str, object] = {}
    hub, httpx_module, _ = make_httpx_test_backend(
        lambda request: httpx.Response(200), observed_config
    )

    def admit(*, connection_hosts=()):
        result = admitted_network()
        result["connection_hosts_verified"] = list(connection_hosts)
        return result

    observer = PUBLISH.configure_hf_http_observer(
        admitted_network(),
        admit=admit,
        hub_module=hub,
        httpx_module=httpx_module,
        hub_request_hook=lambda request: None,
    )
    observer.admit_url("https://CDN-LFS.HUGGINGFACE.CO/blob?token=must-not-be-recorded")
    assert observer.observed_hosts == ["cdn-lfs.huggingface.co"]
    assert "must-not-be-recorded" not in repr(observer.observed_hosts)


def test_unknown_http_backend_fails_closed() -> None:
    unknown = SimpleNamespace(__version__="2.0.0")
    with pytest.raises(PUBLISH.PublishError, match="no audited HTTP backend"):
        PUBLISH.configure_hf_http_observer(admitted_network(), hub_module=unknown)


def test_httpx_factory_rejects_preexisting_untracked_client() -> None:
    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    stale_client = FakeClient()
    hub = SimpleNamespace(
        __version__="1.8.0",
        set_client_factory=lambda factory: None,
        get_session=lambda: stale_client,
    )
    with pytest.raises(PUBLISH.PublishError, match="did not activate the audited httpx Client"):
        PUBLISH.configure_hf_http_observer(
            admitted_network(),
            hub_module=hub,
            httpx_module=SimpleNamespace(Client=FakeClient),
            hub_request_hook=lambda request: None,
        )


def test_requests_compat_backend_forces_exact_proxy_and_disables_env() -> None:
    events: list[str] = []
    state: dict[str, object] = {}

    class FakeSession:
        def __init__(self):
            self.trust_env = True
            self.proxies = {}

        def send(self, request, **kwargs):
            events.append("send")
            return SimpleNamespace(status_code=200)

    def configure_http_backend(*, backend_factory):
        state.clear()
        state["factory"] = backend_factory

    def get_session():
        if "session" not in state:
            state["session"] = state["factory"]()
        return state["session"]

    def admit(*, connection_hosts=()):
        events.append(f"admit:{connection_hosts[0]}")
        result = admitted_network()
        result["connection_hosts_verified"] = list(connection_hosts)
        return result

    hub = SimpleNamespace(
        __version__="0.36.2",
        configure_http_backend=configure_http_backend,
        get_session=get_session,
    )
    observer = PUBLISH.configure_hf_http_observer(
        admitted_network(),
        admit=admit,
        hub_module=hub,
        base_backend_factory=FakeSession,
    )
    session = get_session()
    session.send(SimpleNamespace(url="https://huggingface.co/api/datasets"))
    assert session.trust_env is False
    assert session.proxies == {"http": PUBLISH.PROXY_URL, "https": PUBLISH.PROXY_URL}
    assert events == ["admit:huggingface.co", "send", "admit:huggingface.co"]
    assert observer.observed_hosts == ["huggingface.co"]


def test_main_admits_network_before_first_hf_api_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = make_bundle(tmp_path)
    args = SimpleNamespace(
        bundle=bundle,
        repo_id="AlexGeek/RLdataset",
        revision=PUBLISH.COMPLETED_PRIVATE_REVISION,
    )
    events: list[str] = []

    class MainApi:
        def __init__(self, *values, **kwargs):
            assert kwargs.get("endpoint") == PUBLISH.HF_ENDPOINT
            events.append("api_init")

    def admit() -> dict:
        events.append("admit")
        return admitted_network()

    monkeypatch.setattr(PUBLISH, "parse_args", lambda: args)
    monkeypatch.setattr(PUBLISH, "verify_reviewed_bundle", lambda bundle: None)
    monkeypatch.setattr(PUBLISH, "admit_hf_network", admit)
    monkeypatch.setattr(
        PUBLISH,
        "configure_hf_http_observer",
        lambda admission: SimpleNamespace(observed_hosts=[]),
    )
    monkeypatch.setattr(
        PUBLISH,
        "completed_private_preflight",
        lambda api, parsed, checksums, admission: events.append("preflight") or {"ok": True},
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=MainApi))
    assert PUBLISH.main() == 0
    assert events == ["admit", "api_init", "preflight", "admit"]


def test_guarded_api_rejects_route_change_before_hub_call(monkeypatch: pytest.MonkeyPatch) -> None:
    initial = admitted_network()
    changed = dict(initial)
    changed["selected_leaf_sha256"] = "b" * 64
    calls: list[str] = []
    monkeypatch.setattr(PUBLISH, "admit_hf_network", lambda: changed)
    raw_api = SimpleNamespace(repo_info=lambda *args, **kwargs: calls.append("repo_info"))
    api = PUBLISH.GuardedHfApi(raw_api, initial)
    with pytest.raises(PUBLISH.PublishError, match="selector changed"):
        api.repo_info("AlexGeek/RLdataset")
    assert calls == []


def test_publisher_raw_download_pins_official_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = tmp_path / "payload"
    payload.write_bytes(b"reviewed")
    observed: dict[str, object] = {}

    def fake_download(*args, **kwargs):
        observed.update(kwargs)
        return str(payload)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_download),
    )
    monkeypatch.setattr(PUBLISH, "admit_hf_network", lambda: admitted_network())
    guarded_api = PUBLISH.GuardedHfApi(SimpleNamespace(), admitted_network())
    digest = PUBLISH.download_repo_file_sha256(
        guarded_api,
        "AlexGeek/RLdataset",
        "README.md",
        "d" * 40,
    )
    assert observed["endpoint"] == "https://huggingface.co"
    assert digest == hashlib.sha256(b"reviewed").hexdigest()


def test_reviewed_full_bundle_matches_pinned_inventory_and_manifest() -> None:
    bundle = Path("/data-1/tmp/verl_agent_scratch/rlvr_full_upload_candidate_20260729_v3")
    if not bundle.is_dir():
        pytest.skip("operational full-scope bundle is not materialized on this host")
    checksums = PUBLISH.load_and_verify_bundle(bundle)
    PUBLISH.verify_reviewed_bundle(bundle)
    assert len(checksums) == 17
    assert checksums["metadata/publication_inventory.json"] == PUBLISH.REVIEWED_PRIVATE_INVENTORY_SHA256
    assert checksums["metadata/checksums.sha256"] == PUBLISH.REVIEWED_CHECKSUMS_SHA256


def fake_repo_info(sha: str | None, private: bool, paths: tuple[str, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(
        id="AlexGeek/RLdataset",
        sha=sha,
        private=private,
        gated=False,
        last_modified=None,
        siblings=[
            SimpleNamespace(rfilename=path, size=1, blob_id="b" * 40, lfs=None)
            for path in paths
        ],
    )


def fake_refs(
    branches: tuple[str, ...] = ("main",),
    tags: tuple[str, ...] = (),
    converts: tuple[tuple[str, str, str], ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        branches=[SimpleNamespace(name=name) for name in branches],
        converts=[
            SimpleNamespace(name=name, ref=ref, target_commit=target)
            for name, ref, target in converts
        ],
        tags=[SimpleNamespace(name=name) for name in tags],
        pull_requests=[],
    )


class FakePublishApi:
    def __init__(
        self,
        repo_infos: list[SimpleNamespace | Exception] | None = None,
        commit_lists: list[list[str]] | None = None,
        refs: list[SimpleNamespace] | None = None,
        upload_error: BaseException | None = None,
        squash_error: BaseException | None = None,
        settings_errors: list[BaseException | None] | None = None,
    ) -> None:
        self.repo_infos = list(repo_infos or [])
        self.commit_lists = list(commit_lists or [])
        self.refs = list(refs or [])
        self.upload_error = upload_error
        self.squash_error = squash_error
        self.settings_errors = list(settings_errors or [])
        self.upload_kwargs = None
        self.settings_calls: list[dict] = []
        self.squash_calls = 0
        self.token = "fake-token"

    def upload_folder(self, **kwargs):
        self.upload_kwargs = kwargs
        if self.upload_error is not None:
            raise self.upload_error
        return SimpleNamespace(oid="a" * 40, commit_url="https://example.invalid/commit")

    def repo_info(self, *args, **kwargs):
        assert self.repo_infos
        value = self.repo_infos.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def list_repo_commits(self, *args, **kwargs):
        assert self.commit_lists
        return [SimpleNamespace(commit_id=value) for value in self.commit_lists.pop(0)]

    def list_repo_refs(self, *args, **kwargs):
        assert self.refs
        return self.refs.pop(0)

    def super_squash_history(self, *args, **kwargs):
        self.squash_calls += 1
        if self.squash_error is not None:
            raise self.squash_error

    def update_repo_settings(self, *args, **kwargs):
        self.settings_calls.append(kwargs)
        if self.settings_errors:
            error = self.settings_errors.pop(0)
            if error is not None:
                raise error


def test_post_squash_policy_purges_only_legacy_and_verifies_safe_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_probes: list[dict[str, str]] = []
    verified: list[tuple[str, dict[str, str]]] = []

    def fake_unreachable(api, repo_id, probes):
        captured_probes.extend(probes)
        return [
            {
                "revision": item["revision"],
                "api": "not_found",
                "resolve": "not_found",
                "probe_path": item["path"],
            }
            for item in probes
        ]

    monkeypatch.setattr(PUBLISH, "assert_old_revisions_unreachable", fake_unreachable)
    monkeypatch.setattr(
        PUBLISH,
        "verify_remote",
        lambda api, repo_id, revision, checksums: verified.append((revision, checksums)),
    )
    checksums = {"README.md": "1" * 64, "data/reviewed.parquet": "2" * 64}
    api = FakePublishApi(
        repo_infos=[
            fake_repo_info(PUBLISH.SAFE_BOOTSTRAP_REVISION, True, (".gitattributes",)),
            fake_repo_info(PUBLISH.SAFE_UPLOADED_REVISION, True, tuple(checksums)),
        ],
        commit_lists=[
            [PUBLISH.SAFE_BOOTSTRAP_REVISION],
            [PUBLISH.SAFE_UPLOADED_REVISION, PUBLISH.SAFE_BOOTSTRAP_REVISION],
        ],
    )
    legacy, retained = PUBLISH.verify_post_squash_history_policy(
        api,
        "AlexGeek/RLdataset",
        checksums,
    )
    assert captured_probes == [
        {"revision": revision, "path": path}
        for revision, path in PUBLISH.REQUIRED_PURGED_LEGACY_PROBES
    ]
    assert [item["revision"] for item in legacy] == [
        revision for revision, _ in PUBLISH.REQUIRED_PURGED_LEGACY_PROBES
    ]
    assert [item["status"] for item in retained] == ["verified_safe", "verified_safe"]
    assert verified == [
        (
            PUBLISH.SAFE_BOOTSTRAP_REVISION,
            {".gitattributes": PUBLISH.SAFE_BOOTSTRAP_GITATTRIBUTES_SHA256},
        ),
        (PUBLISH.SAFE_UPLOADED_REVISION, checksums),
    ]


def test_post_squash_policy_rejects_retained_bootstrap_with_legacy_ancestry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(PUBLISH, "assert_old_revisions_unreachable", lambda *values: [])
    api = FakePublishApi(
        repo_infos=[fake_repo_info(PUBLISH.SAFE_BOOTSTRAP_REVISION, True, (".gitattributes",))],
        commit_lists=[
            [
                PUBLISH.SAFE_BOOTSTRAP_REVISION,
                PUBLISH.REQUIRED_PURGED_LEGACY_PROBES[0][0],
            ]
        ],
    )
    with pytest.raises(PUBLISH.PublishError, match="unreviewed ancestry"):
        PUBLISH.verify_post_squash_history_policy(
            api,
            "AlexGeek/RLdataset",
            {"README.md": "1" * 64},
        )


def test_post_squash_policy_rejects_uploaded_bundle_with_unreviewed_ancestry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(PUBLISH, "assert_old_revisions_unreachable", lambda *values: [])
    monkeypatch.setattr(PUBLISH, "verify_remote", lambda *values: None)
    api = FakePublishApi(
        repo_infos=[
            fake_repo_info(PUBLISH.SAFE_BOOTSTRAP_REVISION, True, (".gitattributes",)),
            fake_repo_info(PUBLISH.SAFE_UPLOADED_REVISION, True, ("README.md",)),
        ],
        commit_lists=[
            [PUBLISH.SAFE_BOOTSTRAP_REVISION],
            [
                PUBLISH.SAFE_UPLOADED_REVISION,
                PUBLISH.REQUIRED_PURGED_LEGACY_PROBES[0][0],
            ],
        ],
    )
    with pytest.raises(PUBLISH.PublishError, match="unreviewed ancestry"):
        PUBLISH.verify_post_squash_history_policy(
            api,
            "AlexGeek/RLdataset",
            {"README.md": "1" * 64},
        )


def test_post_squash_policy_allows_safe_retained_revisions_to_be_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NotFound(RuntimeError):
        response = SimpleNamespace(status_code=404)

    monkeypatch.setattr(PUBLISH, "assert_old_revisions_unreachable", lambda *values: [])
    api = FakePublishApi(repo_infos=[NotFound("missing"), NotFound("missing")])
    _, retained = PUBLISH.verify_post_squash_history_policy(
        api,
        "AlexGeek/RLdataset",
        {"README.md": "1" * 64},
    )
    assert [item["status"] for item in retained] == ["not_found", "not_found"]


def test_history_squash_rejects_tags_even_when_main_is_present() -> None:
    api = FakePublishApi(
        commit_lists=[["d" * 40]],
        refs=[fake_refs(tags=("release",))],
    )
    with pytest.raises(PUBLISH.PublishError, match="only the main branch"):
        PUBLISH.validate_private_parent(
            api,
            fake_repo_info("d" * 40, True),
            "AlexGeek/RLdataset",
            "d" * 40,
            {"README.md": "1" * 64},
        )


def test_exact_root_only_parquet_convert_ref_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    target = "f" * 40
    convert_info = fake_repo_info(target, True)
    convert_info.siblings = [
        SimpleNamespace(rfilename=".gitattributes", size=46, blob_id="1" * 40, lfs=None),
        SimpleNamespace(
            rfilename="converted/reviewed.parquet",
            size=10,
            blob_id="2" * 40,
            lfs=SimpleNamespace(sha256="2" * 64, size=10, pointer_size=128),
        ),
    ]
    api = FakePublishApi(
        repo_infos=[convert_info],
        commit_lists=[["d" * 40], [target]],
        refs=[fake_refs(converts=(("parquet", "refs/convert/parquet", target),))],
    )
    monkeypatch.setattr(PUBLISH, "download_repo_file_sha256", lambda *values: "1" * 64)
    commits, refs = PUBLISH.validate_private_parent(
        api,
        fake_repo_info("d" * 40, True),
        "AlexGeek/RLdataset",
        "d" * 40,
        {
            ".gitattributes": "1" * 64,
            "data/reviewed.parquet": "2" * 64,
        },
    )
    assert commits == ["d" * 40]
    assert refs["converts"] == [
        {
            "name": "parquet",
            "ref": "refs/convert/parquet",
            "target_commit": target,
            "commit_ids": [target],
            "payload_lfs_sha256": ["2" * 64],
        }
    ]


def test_parquet_convert_ref_rejects_unreviewed_lfs_oid(monkeypatch: pytest.MonkeyPatch) -> None:
    target = "f" * 40
    convert_info = fake_repo_info(target, True)
    convert_info.siblings = [
        SimpleNamespace(rfilename=".gitattributes", size=46, blob_id="1" * 40, lfs=None),
        SimpleNamespace(
            rfilename="converted/legacy.parquet",
            size=10,
            blob_id="2" * 40,
            lfs=SimpleNamespace(sha256="9" * 64, size=10, pointer_size=128),
        ),
    ]
    api = FakePublishApi(
        repo_infos=[convert_info],
        commit_lists=[["d" * 40], [target]],
        refs=[fake_refs(converts=(("parquet", "refs/convert/parquet", target),))],
    )
    monkeypatch.setattr(PUBLISH, "download_repo_file_sha256", lambda *values: "1" * 64)
    with pytest.raises(PUBLISH.PublishError, match="unreviewed LFS object"):
        PUBLISH.validate_private_parent(
            api,
            fake_repo_info("d" * 40, True),
            "AlexGeek/RLdataset",
            "d" * 40,
            {
                ".gitattributes": "1" * 64,
                "data/reviewed.parquet": "2" * 64,
            },
        )


@pytest.mark.parametrize(
    "convert",
    [
        ("parquet", "refs/convert/unreviewed", "f" * 40),
        ("other", "refs/convert/parquet", "f" * 40),
    ],
)
def test_any_non_exact_convert_ref_is_rejected(convert: tuple[str, str, str]) -> None:
    api = FakePublishApi(
        commit_lists=[["d" * 40]],
        refs=[fake_refs(converts=(convert,))],
    )
    with pytest.raises(PUBLISH.PublishError, match="exact refs/convert/parquet"):
        PUBLISH.validate_private_parent(
            api,
            fake_repo_info("d" * 40, True),
            "AlexGeek/RLdataset",
            "d" * 40,
            {"README.md": "1" * 64},
        )


def test_parquet_convert_ref_with_non_root_history_is_rejected() -> None:
    target = "f" * 40
    api = FakePublishApi(
        commit_lists=[["d" * 40], [target, "9" * 40]],
        refs=[fake_refs(converts=(("parquet", "refs/convert/parquet", target),))],
    )
    with pytest.raises(PUBLISH.PublishError, match="exactly its one root target commit"):
        PUBLISH.validate_private_parent(
            api,
            fake_repo_info("d" * 40, True),
            "AlexGeek/RLdataset",
            "d" * 40,
            {"README.md": "1" * 64},
        )


def test_completed_publisher_ast_contains_no_hf_mutator() -> None:
    for name in (
        "apply_private_publication",
        "reconcile_private_publication",
        "rollback_private",
        "squash_and_finish_private",
    ):
        assert not hasattr(PUBLISH, name)
    source = (REPO_ROOT / "scripts/publish_rebuttal_rlvr_public_dataset.py").read_text()
    tree = ast.parse(source)
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert called.isdisjoint(
        {
            "create_repo",
            "delete_repo",
            "upload_folder",
            "super_squash_history",
            "update_repo_settings",
        }
    )
    assert not any(
        keyword.arg in {"delete_patterns", "private"}
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
    )


@pytest.mark.parametrize(
    "forbidden",
    ["--apply", "--confirm-repo-id", "--expected-parent", "--reconcile", "--public-transition"],
)
def test_completed_cli_does_not_expose_mutation_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forbidden: str
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish_rebuttal_rlvr_public_dataset.py",
            "--bundle",
            str(tmp_path),
            forbidden,
        ],
    )
    with pytest.raises(SystemExit) as error:
        PUBLISH.parse_args()
    assert error.value.code == 2


def test_completed_preflight_revision_is_immutable() -> None:
    good = SimpleNamespace(
        repo_id=PUBLISH.DEFAULT_REPO_ID,
        revision=PUBLISH.COMPLETED_PRIVATE_REVISION,
    )
    PUBLISH.validate_cli_args(good)
    with pytest.raises(PUBLISH.PublishError, match="completed private revision"):
        PUBLISH.validate_cli_args(
            SimpleNamespace(repo_id=PUBLISH.DEFAULT_REPO_ID, revision="f" * 40)
        )


def test_handoff_tmux_command_records_pipefail_completion_status() -> None:
    guide = (
        REPO_ROOT / "docs/joint_training/guides/rebuttal_rlvr_hf_dataset_handoff.md"
    ).read_text()
    assert "bash -lc" in guide
    assert "set -euo pipefail" in guide
    assert "trap record_status EXIT" in guide
    assert 'test "$(cat "$DOWNLOAD_STATUS")" = 0' in guide
    assert "HTTP_PROXY=http://127.0.0.1:7890" in guide


@pytest.mark.parametrize(
    "value",
    ["/absolute", "../escape", "a/../b", "a//b", "a/./b", "a/", ".", "bad\npath"],
)
def test_prepare_and_publish_reject_unsafe_repo_paths(value: str) -> None:
    with pytest.raises(PREPARE.BundleError, match="unsafe"):
        PREPARE.safe_relative_path(value)
    with pytest.raises(PUBLISH.PublishError, match="unsafe"):
        PUBLISH.safe_relative(value)


def test_public_inventory_removes_host_paths_and_remote_credentials() -> None:
    source = {
        "schema_version": 1,
        "inventory_id": "test",
        "layout_version": "rebuttal-rlvr-dataset-v1",
        "generated_at": "2026-07-29",
        "target_repository": {"current_credential_identity": "secret-user"},
        "public_assets": [
            {
                "asset_id": "allowed",
                "local_path": "/host/private/data.parquet",
                "path_in_repo": "data/allowed.parquet",
                "sha256": "a" * 64,
            }
        ],
        "restricted_assets": [
            {
                "asset_id": "blocked",
                "local_path": "/host/private/blocked.parquet",
                "expected_path_below_dataset_root": "data/blocked.parquet",
                "size_bytes": 10,
                "row_count": 1,
                "sha256": "b" * 64,
                "publication_status": "blocked",
                "reason": "test",
            }
        ],
        "evaluator_source_pins": [],
        "license_files": [{"license_id": "x", "path_in_repo": "LICENSES/x", "url": "https://x", "sha256": "c" * 64}],
        "processing_files": [
            {
                "local_path": "/host/private/process.py",
                "path_in_repo": "processing/process.py",
                "sha256": "d" * 64,
            }
        ],
    }
    result = PREPARE.public_inventory(source)
    rendered = json.dumps(result)
    assert "/host/private" not in rendered
    assert "secret-user" not in rendered
    assert "https://x" not in rendered
    assert result["excluded_assets"][0]["publication_status"] == "blocked"
    assert result["processing_files"] == [
        {"path_in_repo": "processing/process.py", "sha256": "d" * 64}
    ]


def write_checksum_file(bundle: Path) -> None:
    checksum = bundle / "metadata/checksums.sha256"
    lines = []
    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or path == checksum:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(bundle).as_posix()}\n")
    checksum.write_text("".join(lines))


def make_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    (bundle / "metadata").mkdir(parents=True)
    (bundle / "data").mkdir()
    (bundle / "README.md").write_text("readme\n")
    (bundle / ".gitattributes").write_text("*.parquet filter=lfs diff=lfs merge=lfs -text\n")
    categories = (
        ["math_training"]
        + ["math_evaluation"] * 7
        + ["code_training"]
        + ["code_evaluation"] * 4
    )
    files = []
    for index, category in enumerate(categories):
        relative = f"data/{category}/{index:02d}.parquet"
        payload = f"payload-{index}".encode()
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        files.append(
            {
                "category": category,
                "relative_path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    (bundle / "metadata/publication_inventory.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "layout_version": "meituan-handoff-data-v1",
                "bundle_purpose": PUBLISH.REVIEWED_BUNDLE_PURPOSE,
                "publication_status": "private_handoff_only",
                "payload_summary": {
                    "file_count": 13,
                    "math_training_files": 1,
                    "math_evaluation_files": 7,
                    "code_training_files": 1,
                    "code_evaluation_files": 4,
                },
                "files": files,
            }
        )
    )
    write_checksum_file(bundle)
    return bundle


def test_publish_bundle_verifier_accepts_exact_allowlist(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    checksums = PUBLISH.load_and_verify_bundle(bundle)
    assert len(checksums) == 17
    assert {name for name in checksums if name.startswith("data/")} == {
        item["relative_path"]
        for item in json.loads((bundle / "metadata/publication_inventory.json").read_text())["files"]
    }


def test_publish_bundle_verifier_rejects_unlisted_data(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    (bundle / "data/restricted.parquet").write_bytes(b"restricted")
    write_checksum_file(bundle)
    with pytest.raises(PUBLISH.PublishError, match="file allowlist mismatch"):
        PUBLISH.load_and_verify_bundle(bundle)


def test_publish_bundle_verifier_rejects_unlisted_processing_file(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    (bundle / "processing").mkdir()
    (bundle / "processing/unreviewed.py").write_text("print('not reviewed')\n")
    write_checksum_file(bundle)
    with pytest.raises(PUBLISH.PublishError, match="file allowlist mismatch"):
        PUBLISH.load_and_verify_bundle(bundle)


def test_publish_bundle_verifier_rejects_symlink(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    (bundle / "linked").symlink_to(bundle / "README.md")
    write_checksum_file(bundle)
    with pytest.raises(PUBLISH.PublishError, match="symlinks"):
        PUBLISH.load_and_verify_bundle(bundle)
