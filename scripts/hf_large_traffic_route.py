#!/usr/bin/env python3
"""Fail-closed Hugging Face admission for the audited Mihomo large-traffic route."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
from pathlib import Path
import threading
from typing import Any, Callable
from urllib.parse import quote, urlsplit
from urllib.request import ProxyHandler, Request, build_opener


PROXY_URL = "http://127.0.0.1:7890"
HF_ENDPOINT = "https://huggingface.co"
HF_METADATA_TIMEOUT_SECONDS = 30
HF_DOWNLOAD_TIMEOUT_SECONDS = 120
DEFAULT_MIHOMO_RUNTIME_CONFIG = Path("/root/clashctl/resources/runtime.yaml")
HF_ROUTE_GROUP = "大流量"
HF_ROUTE_REQUIRED_HOSTS = (
    "huggingface.co",
    "hf.co",
    "xethub.hf.co",
    "cas-bridge.xethub.hf.co",
    "transfer.xethub.hf.co",
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.huggingface.co",
    "cdn-lfs-eu-1.huggingface.co",
    "hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com",
    "hf-hub-lfs-us-east-1.s3.amazonaws.com",
)
RESIDENTIAL_NODE_MARKERS = ("家宽", "住宅", "家庭", "residential", "home broadband")
DIRECT_PROXY_PROTOCOL_ALIASES = {
    # The current BW subscription contains only these two audited remote-leaf
    # protocols. New protocols must be reviewed and added explicitly.
    "anytls": frozenset({"anytls"}),
    "ss": frozenset({"shadowsocks", "ss"}),
}
VOLATILE_PROXY_FIELDS = {
    "alive",
    "delay",
    "history",
    "lastDelay",
    "subscriptionInfo",
    "updatedAt",
}
INDETERMINATE_DOMAIN_RULE_TYPES = {
    "geosite",
    "logical",
    "ruleprovider",
    "ruleset",
    "script",
    "subrules",
}


class RouteAdmissionError(RuntimeError):
    pass


def validate_route_admission(admission: Any) -> dict[str, Any]:
    if (
        not isinstance(admission, dict)
        or admission.get("schema_version") != 2
        or admission.get("endpoint") != HF_ENDPOINT
        or admission.get("proxy_url") != PROXY_URL
        or admission.get("route_group") != HF_ROUTE_GROUP
        or admission.get("route_group_type") != "Selector"
        or admission.get("runtime_group_type") != "select"
        or admission.get("selected_namespace") != "BW"
        or admission.get("selected_region") != "Hong Kong"
        or admission.get("selected_residential") is not False
        or admission.get("required_hosts_verified") != list(HF_ROUTE_REQUIRED_HOSTS)
        or not isinstance(admission.get("selected_leaf_sha256"), str)
        or len(admission["selected_leaf_sha256"]) != 64
        or any(ch not in "0123456789abcdef" for ch in admission["selected_leaf_sha256"])
        or not isinstance(admission.get("selected_protocol"), str)
        or admission["selected_protocol"] not in DIRECT_PROXY_PROTOCOL_ALIASES
        or not isinstance(admission.get("controller_protocol"), str)
        or admission["controller_protocol"].lower()
        not in DIRECT_PROXY_PROTOCOL_ALIASES[admission["selected_protocol"]]
        or admission.get("selector_projection_verified") is not True
        or admission.get("runtime_group_identity_verified") is not True
        or admission.get("runtime_proxy_identity_verified") is not True
        or not isinstance(admission.get("connection_hosts_verified"), list)
        or admission["connection_hosts_verified"] != sorted(set(admission["connection_hosts_verified"]))
        or any(
            not isinstance(host, str) or _normalize_hostname(host) != host
            for host in admission["connection_hosts_verified"]
        )
    ):
        raise RouteAdmissionError("Hugging Face operation requires a complete verified route admission")
    return admission


def enforce_hf_proxy(*, anonymous: bool = False) -> None:
    """Keep Hub traffic in the standard HTTP client on the audited mixed port."""
    configured_endpoint = os.environ.get("HF_ENDPOINT")
    if configured_endpoint and configured_endpoint.rstrip("/") != HF_ENDPOINT:
        raise RouteAdmissionError(
            f"Hugging Face endpoint override is forbidden: expected {HF_ENDPOINT}"
        )
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ[name] = PROXY_URL
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_HF_TRANSFER"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(HF_METADATA_TIMEOUT_SECONDS)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(HF_DOWNLOAD_TIMEOUT_SECONDS)
    if anonymous:
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"


def mihomo_controller_json(base_url: str, secret: str, path: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {secret}"} if secret else {}
    request = Request(f"{base_url}{path}", headers=headers)
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=3) as response:
            value = json.load(response)
    except Exception as exc:
        raise RouteAdmissionError("Mihomo route admission failed: local controller is unavailable") from exc
    if not isinstance(value, dict):
        raise RouteAdmissionError("Mihomo route admission failed: controller returned a non-object response")
    return value


def first_domain_route(rules: list[Any], hostname: str) -> str | None:
    host = hostname.lower().rstrip(".")
    for raw in rules:
        if not isinstance(raw, dict):
            raise RouteAdmissionError("Mihomo route admission failed: malformed rule entry")
        rule_type = str(raw.get("type", "")).lower().replace("_", "").replace("-", "")
        payload = str(raw.get("payload", "")).lower().lstrip(".").rstrip(".")
        matched = False
        if rule_type == "domain":
            matched = host == payload
        elif rule_type == "domainsuffix":
            matched = host == payload or host.endswith(f".{payload}")
        elif rule_type == "domainkeyword":
            matched = payload in host
        elif rule_type == "match":
            matched = True
        elif rule_type in INDETERMINATE_DOMAIN_RULE_TYPES:
            raise RouteAdmissionError(
                "Mihomo route admission failed: an unevaluable domain rule precedes the required HF route"
            )
        else:
            raise RouteAdmissionError(
                "Mihomo route admission failed: an unsupported rule type precedes the required HF route"
            )
        if matched:
            proxy = raw.get("proxy")
            return proxy if isinstance(proxy, str) else None
    return None


def _stable_proxy_view(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in VOLATILE_PROXY_FIELDS}


def _leaf_fingerprint(
    *,
    runtime_group: dict[str, Any],
    runtime_proxy: dict[str, Any],
    controller_leaf: dict[str, Any],
) -> str:
    identity = {
        "route_group": HF_ROUTE_GROUP,
        "runtime_group": {
            "name": runtime_group.get("name"),
            "type": runtime_group.get("type"),
            "selected": runtime_proxy.get("name"),
        },
        "runtime_proxy": runtime_proxy,
        "controller_leaf": _stable_proxy_view(controller_leaf),
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_hostname(value: str) -> str:
    if not isinstance(value, str) or not value or any(character.isspace() for character in value):
        raise RouteAdmissionError("Hugging Face connection target has an invalid hostname")
    try:
        normalized = value.rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise RouteAdmissionError("Hugging Face connection target has an invalid hostname") from exc
    if not normalized or len(normalized) > 253 or "/" in normalized or ":" in normalized:
        raise RouteAdmissionError("Hugging Face connection target has an invalid hostname")
    try:
        ipaddress.ip_address(normalized)
    except ValueError:
        pass
    else:
        raise RouteAdmissionError("Hugging Face connection target must not be an IP literal")
    return normalized


def validate_hf_route_admission(
    config_path: Path | None = None,
    *,
    fetch_json: Callable[[str, str, str], dict[str, Any]] = mihomo_controller_json,
    connection_hosts: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Prove the live HF route is 大流量 -> Hong Kong non-residential leaf."""
    if config_path is None:
        config_path = Path(os.environ.get("MIHOMO_RUNTIME_CONFIG", str(DEFAULT_MIHOMO_RUNTIME_CONFIG)))
    if not config_path.is_absolute() or config_path.is_symlink() or not config_path.is_file():
        raise RouteAdmissionError("Mihomo route admission failed: runtime config must be an absolute regular file")
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RouteAdmissionError("Mihomo route admission failed: runtime config cannot be parsed") from exc
    if not isinstance(config, dict) or config.get("mixed-port") != 7890:
        raise RouteAdmissionError("Mihomo route admission failed: runtime mixed-port is not 7890")
    controller = config.get("external-controller")
    if not isinstance(controller, str) or not controller.strip():
        raise RouteAdmissionError("Mihomo route admission failed: external controller is not configured")
    controller_url = controller if "://" in controller else f"http://{controller}"
    parsed = urlsplit(controller_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.port is None:
        raise RouteAdmissionError("Mihomo route admission failed: controller must be loopback HTTP")
    base_url = f"http://{parsed.netloc}"
    secret = config.get("secret", "")
    if not isinstance(secret, str):
        raise RouteAdmissionError("Mihomo route admission failed: controller credential has invalid type")

    group = fetch_json(base_url, secret, f"/proxies/{quote(HF_ROUTE_GROUP, safe='')}")
    selected = group.get("now")
    members = group.get("all")
    if group.get("type") != "Selector" or not isinstance(selected, str) or not isinstance(members, list):
        raise RouteAdmissionError("Mihomo route admission failed: 大流量 is not a valid Selector")
    lowered = selected.lower()
    if (
        selected not in members
        or not selected.lstrip().startswith("[BW]")
        or "香港" not in selected
        or any(marker in lowered for marker in RESIDENTIAL_NODE_MARKERS)
    ):
        raise RouteAdmissionError("Mihomo route admission failed: 大流量 is not on a Hong Kong non-residential BW node")
    runtime_groups = [
        item
        for item in (config.get("proxy-groups") or [])
        if isinstance(item, dict) and item.get("name") == HF_ROUTE_GROUP
    ]
    if len(runtime_groups) != 1:
        raise RouteAdmissionError(
            "Mihomo route admission failed: 大流量 must map to exactly one runtime selector group"
        )
    runtime_group = runtime_groups[0]
    runtime_members = runtime_group.get("proxies")
    if (
        runtime_group.get("type") != "select"
        or not isinstance(runtime_members, list)
        or not runtime_members
        or any(not isinstance(item, str) or not item for item in runtime_members)
        or len(runtime_members) != len(set(runtime_members))
        or runtime_members != members
        or selected not in runtime_members
    ):
        raise RouteAdmissionError(
            "Mihomo route admission failed: controller selector does not match the runtime 大流量 group"
        )
    runtime_matches = [
        item
        for item in (config.get("proxies") or [])
        if isinstance(item, dict) and item.get("name") == selected
    ]
    if len(runtime_matches) != 1:
        raise RouteAdmissionError(
            "Mihomo route admission failed: selected leaf must map to exactly one runtime proxy"
        )
    runtime_proxy = runtime_matches[0]
    runtime_type = runtime_proxy.get("type")
    runtime_server = runtime_proxy.get("server")
    runtime_port = runtime_proxy.get("port")
    if (
        not isinstance(runtime_type, str)
        or runtime_type not in DIRECT_PROXY_PROTOCOL_ALIASES
        or not isinstance(runtime_server, str)
        or not runtime_server
        or not isinstance(runtime_port, int)
        or isinstance(runtime_port, bool)
        or not 1 <= runtime_port <= 65535
    ):
        raise RouteAdmissionError(
            "Mihomo route admission failed: runtime selection is not a concrete remote proxy leaf"
        )

    leaf = fetch_json(base_url, secret, f"/proxies/{quote(selected, safe='')}")
    leaf_id = leaf.get("id") if isinstance(leaf, dict) else None
    if (
        not isinstance(leaf, dict)
        or not leaf
        or leaf.get("name") != selected
        or not isinstance(leaf.get("type"), str)
        or leaf["type"].lower() not in DIRECT_PROXY_PROTOCOL_ALIASES[runtime_type]
        or not isinstance(leaf_id, str)
        or not leaf_id
        or leaf.get("alive") is not True
    ):
        raise RouteAdmissionError(
            "Mihomo route admission failed: controller leaf identity is incomplete or inconsistent"
        )

    rules_response = fetch_json(base_url, secret, "/rules")
    rules = rules_response.get("rules")
    if not isinstance(rules, list):
        raise RouteAdmissionError("Mihomo route admission failed: controller rules are unavailable")
    dynamic_hosts = sorted({_normalize_hostname(host) for host in connection_hosts})
    verified_hosts = list(HF_ROUTE_REQUIRED_HOSTS) + dynamic_hosts
    misrouted = [host for host in verified_hosts if first_domain_route(rules, host) != HF_ROUTE_GROUP]
    if misrouted:
        raise RouteAdmissionError("Mihomo route admission failed: one or more HF domains do not route to 大流量")

    return {
        "schema_version": 2,
        "endpoint": HF_ENDPOINT,
        "proxy_url": PROXY_URL,
        "route_group": HF_ROUTE_GROUP,
        "route_group_type": "Selector",
        "runtime_group_type": "select",
        "selected_leaf_sha256": _leaf_fingerprint(
            runtime_group=runtime_group,
            runtime_proxy=runtime_proxy,
            controller_leaf=leaf,
        ),
        "selected_namespace": "BW",
        "selected_region": "Hong Kong",
        "selected_residential": False,
        "selected_protocol": runtime_type,
        "controller_protocol": leaf["type"],
        "selector_projection_verified": True,
        "runtime_group_identity_verified": True,
        "runtime_proxy_identity_verified": True,
        "required_hosts_verified": list(HF_ROUTE_REQUIRED_HOSTS),
        "connection_hosts_verified": dynamic_hosts,
    }


def admit_hf_network(
    *,
    anonymous: bool = False,
    connection_hosts: tuple[str, ...] = (),
) -> dict[str, Any]:
    enforce_hf_proxy(anonymous=anonymous)
    return validate_hf_route_admission(connection_hosts=connection_hosts)


class HfHttpConnectionObserver:
    """Admit and record the hostname of every actual Hub HTTP request."""

    def __init__(
        self,
        initial_admission: dict[str, Any],
        *,
        admit: Callable[..., dict[str, Any]] = admit_hf_network,
    ) -> None:
        self._expected_leaf = validate_route_admission(initial_admission)["selected_leaf_sha256"]
        self._admit = admit
        self._observed_hosts: set[str] = set()
        self._lock = threading.Lock()

    @property
    def observed_hosts(self) -> list[str]:
        with self._lock:
            return sorted(self._observed_hosts)

    def admit_url(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, str):
            raise RouteAdmissionError("Hugging Face connection target URL is unavailable")
        parsed = urlsplit(value)
        try:
            port = parsed.port
        except ValueError as exc:
            raise RouteAdmissionError("Hugging Face connection target has an invalid port") from exc
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.hostname is None
            or port not in {None, 443}
        ):
            raise RouteAdmissionError("Hugging Face connection target must be credential-free HTTPS")
        host = _normalize_hostname(parsed.hostname)
        admission = validate_route_admission(self._admit(connection_hosts=(host,)))
        if admission["selected_leaf_sha256"] != self._expected_leaf:
            raise RouteAdmissionError("large-traffic selector changed during the Hugging Face connection")
        if host not in admission["connection_hosts_verified"]:
            raise RouteAdmissionError("Hugging Face connection target was not admitted")
        with self._lock:
            self._observed_hosts.add(host)
        return admission


def configure_hf_http_observer(
    initial_admission: dict[str, Any],
    *,
    admit: Callable[..., dict[str, Any]] = admit_hf_network,
    hub_module: Any | None = None,
    httpx_module: Any | None = None,
    hub_request_hook: Callable[[Any], Any] | None = None,
    configure_backend: Callable[..., Any] | None = None,
    base_backend_factory: Callable[[], Any] | None = None,
) -> HfHttpConnectionObserver:
    """Install the one audited Hub backend; unknown versions fail closed."""
    if hub_module is None:
        try:
            import huggingface_hub as hub_module
        except ImportError as exc:
            raise RouteAdmissionError(
                "huggingface_hub does not expose the required observable HTTP backend"
            ) from exc

    version = getattr(hub_module, "__version__", None)
    if not isinstance(version, str):
        raise RouteAdmissionError("huggingface_hub version is unavailable; refusing an unknown backend")
    version_parts = version.split(".")
    try:
        version_family = (int(version_parts[0]), int(version_parts[1]))
    except (IndexError, ValueError) as exc:
        raise RouteAdmissionError("huggingface_hub version is unsupported") from exc

    observer = HfHttpConnectionObserver(initial_admission, admit=admit)

    if version_family == (1, 8):
        set_client_factory = getattr(hub_module, "set_client_factory", None)
        get_session = getattr(hub_module, "get_session", None)
        if not callable(set_client_factory) or not callable(get_session):
            raise RouteAdmissionError(
                "huggingface_hub 1.8.x lacks the required httpx client factory API"
            )
        if httpx_module is None:
            try:
                import httpx as httpx_module
            except ImportError as exc:
                raise RouteAdmissionError("httpx is required for huggingface_hub 1.8.x") from exc
        client_class = getattr(httpx_module, "Client", None)
        if not isinstance(client_class, type):
            raise RouteAdmissionError("httpx Client is unavailable")
        if hub_request_hook is None:
            try:
                from huggingface_hub.utils._http import hf_request_event_hook
            except ImportError as exc:
                raise RouteAdmissionError(
                    "huggingface_hub 1.8.x request hook is unavailable"
                ) from exc
            hub_request_hook = hf_request_event_hook

        def guarded_httpx_request(request: Any) -> None:
            raw_url = getattr(request, "url", None)
            if raw_url is None:
                raise RouteAdmissionError("Hugging Face httpx request URL is unavailable")
            observer.admit_url(str(raw_url))

        def guarded_httpx_response(response: Any) -> None:
            request = getattr(response, "request", None)
            raw_url = getattr(request, "url", None)
            if raw_url is None:
                raise RouteAdmissionError("Hugging Face httpx response request URL is unavailable")
            observer.admit_url(str(raw_url))

        activated_clients: list[Any] = []

        def client_factory() -> Any:
            client = client_class(
                proxy=PROXY_URL,
                trust_env=False,
                follow_redirects=True,
                timeout=None,
                event_hooks={
                    "request": [guarded_httpx_request, hub_request_hook],
                    "response": [guarded_httpx_response],
                },
            )
            activated_clients.append(client)
            return client

        set_client_factory(client_factory)
        session = get_session()
        if (
            not isinstance(session, client_class)
            or not any(session is activated for activated in activated_clients)
        ):
            raise RouteAdmissionError(
                "huggingface_hub did not activate the audited httpx Client"
            )
        return observer

    if version_family != (0, 36):
        raise RouteAdmissionError(
            f"huggingface_hub {version} has no audited HTTP backend implementation"
        )

    if configure_backend is None:
        configure_backend = getattr(hub_module, "configure_http_backend", None)
    if base_backend_factory is None:
        try:
            from huggingface_hub.utils._http import _default_backend_factory
        except ImportError as exc:
            raise RouteAdmissionError(
                "huggingface_hub 0.36.x requests backend factory is unavailable"
            ) from exc
        base_backend_factory = _default_backend_factory
    if not callable(configure_backend) or not callable(base_backend_factory):
        raise RouteAdmissionError(
            "huggingface_hub 0.36.x lacks the required requests backend API"
        )
    get_session = getattr(hub_module, "get_session", None)
    if not callable(get_session):
        raise RouteAdmissionError(
            "huggingface_hub 0.36.x lacks the required requests session accessor"
        )
    activated_sessions: list[Any] = []

    def backend_factory() -> Any:
        session = base_backend_factory()
        original_send = getattr(session, "send", None)
        if not callable(original_send):
            raise RouteAdmissionError("Hugging Face HTTP backend has no observable send method")
        session.trust_env = False
        session.proxies = {"http": PROXY_URL, "https": PROXY_URL}

        def guarded_send(request: Any, **kwargs: Any) -> Any:
            url = getattr(request, "url", None)
            observer.admit_url(url)
            try:
                return original_send(request, **kwargs)
            finally:
                observer.admit_url(url)

        session.send = guarded_send
        activated_sessions.append(session)
        return session

    configure_backend(backend_factory=backend_factory)
    session = get_session()
    if (
        not any(session is activated for activated in activated_sessions)
        or getattr(session, "trust_env", None) is not False
        or getattr(session, "proxies", None) != {"http": PROXY_URL, "https": PROXY_URL}
    ):
        raise RouteAdmissionError(
            "huggingface_hub did not activate the audited requests Session"
        )
    return observer
