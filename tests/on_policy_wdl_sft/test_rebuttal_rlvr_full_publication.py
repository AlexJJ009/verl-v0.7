from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import publish_rebuttal_rlvr_full_dataset_v4 as PUBLISH  # noqa: E402
import verify_rebuttal_rlvr_public_release as VERIFY  # noqa: E402


def repo_info(sha: str, private: bool) -> SimpleNamespace:
    return SimpleNamespace(
        id=PUBLISH.REPO_ID,
        sha=sha,
        private=private,
        gated=False,
        last_modified=None,
        siblings=[],
    )


def refs(main_sha: str) -> SimpleNamespace:
    return SimpleNamespace(
        branches=[
            SimpleNamespace(
                name="main",
                ref="refs/heads/main",
                target_commit=main_sha,
            )
        ],
        converts=[],
        tags=[],
        pull_requests=[],
    )


class FakeApi:
    def __init__(self, infos: list[SimpleNamespace]) -> None:
        self.infos = list(infos)
        self.settings_calls: list[bool] = []

    def repo_info(self, *args, **kwargs):
        assert self.infos
        return self.infos.pop(0)

    def update_repo_settings(self, *args, **kwargs):
        self.settings_calls.append(kwargs["private"])

    def _admit_same_route(self, connection_hosts=()):
        return {"connection_hosts_verified": list(connection_hosts)}


def args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        expected_parent=PUBLISH.EXPECTED_PRIVATE_PARENT,
        state_log=tmp_path / "state.jsonl",
        receipt=tmp_path / "receipt.json",
        bundle=tmp_path / "bundle",
        anonymous_root=tmp_path / "anonymous",
        old_bundle=tmp_path / "old-bundle",
    )


def test_publisher_ast_forbids_history_deletion_and_rewrite() -> None:
    source = (SCRIPTS / "publish_rebuttal_rlvr_full_dataset_v4.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert called.isdisjoint(
        {
            "delete_repo",
            "create_repo",
            "super_squash_history",
            "upload_folder",
        }
    )
    assert "create_commit" in called
    assert "update_repo_settings" in called
    assert not any(
        keyword.arg in {"delete_patterns", "force"}
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
    )


def test_publisher_create_commit_is_compare_and_swap_bound() -> None:
    source = (SCRIPTS / "publish_rebuttal_rlvr_full_dataset_v4.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "create_commit"
    ]
    assert len(calls) == 1
    keywords = {item.arg: item.value for item in calls[0].keywords}
    parent = keywords["parent_commit"]
    assert isinstance(parent, ast.Attribute)
    assert isinstance(parent.value, ast.Name)
    assert (parent.value.id, parent.attr) == ("args", "expected_parent")
    assert isinstance(keywords["revision"], ast.Constant)
    assert keywords["revision"].value == "main"


def test_ref_snapshot_requires_main_to_target_expected_commit() -> None:
    sha = "a" * 40
    api = SimpleNamespace(list_repo_refs=lambda *args, **kwargs: refs(sha))
    snapshot = PUBLISH.ref_snapshot(api, sha)
    assert snapshot["branches"][0]["target_commit"] == sha
    with pytest.raises(PUBLISH.FullPublicationError, match="main ref"):
        PUBLISH.ref_snapshot(api, "b" * 40)


def test_public_verifier_requires_preserved_revision_to_remain_reachable() -> None:
    parent = PUBLISH.EXPECTED_PRIVATE_PARENT
    api = SimpleNamespace(
        repo_info=lambda *args, **kwargs: SimpleNamespace(
            id=PUBLISH.REPO_ID,
            sha=parent,
            private=False,
            gated=False,
            siblings=[SimpleNamespace(rfilename="README.md")],
        )
    )
    assert VERIFY.prove_preserved_revisions_anonymously(api, [parent]) == [
        {
            "revision": parent,
            "anonymous_api": "reachable",
            "file_count": 1,
            "files": ["README.md"],
        }
    ]


def test_public_verifier_cli_binds_reviewed_preserved_parent(tmp_path: Path) -> None:
    parsed = SimpleNamespace(
        revision="a" * 40,
        local_dir=tmp_path / "download",
        receipt=tmp_path / "receipt.json",
        preserved_revision=[],
    )
    with pytest.raises(VERIFY.PublicVerificationError, match="reviewed existing main parent"):
        VERIFY.validate_args(parsed)
    parsed.preserved_revision = [VERIFY.EXPECTED_PRESERVED_PARENT]
    VERIFY.validate_args(parsed)


def test_anonymous_failure_rolls_back_and_fully_verifies_private_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = args(tmp_path)
    parsed.state_log.touch()
    parent = parsed.expected_parent
    release = "a" * 40
    api = FakeApi([repo_info(parent, True), repo_info(release, True)])
    verified: list[tuple[bool, list[str]]] = []

    monkeypatch.setattr(
        PUBLISH,
        "read_only_private_preflight",
        lambda *values: {
            "repo": {"sha": parent},
            "verified_file_sha256": {"README.md": "b" * 64},
        },
    )
    monkeypatch.setattr(
        PUBLISH,
        "upload_or_recover_exact_commit",
        lambda *values: (release, {"repo": {"sha": release}}),
    )

    def verify(*values, expected_private, expected_history, **kwargs):
        verified.append((expected_private, expected_history))
        return {"repo": {"sha": release}}

    monkeypatch.setattr(PUBLISH, "verify_exact_release_state", verify)
    monkeypatch.setattr(
        PUBLISH,
        "run_anonymous_verifier",
        lambda *values: (_ for _ in ()).throw(RuntimeError("anonymous failed")),
    )
    observer = SimpleNamespace(observed_hosts=[])
    with pytest.raises(RuntimeError, match="anonymous failed"):
        PUBLISH.apply_publication(
            api,
            parsed,
            {"selected_leaf_sha256": "a" * 64},
            observer,
            {"README.md": "b" * 64, "validate_dataset.py": "c" * 64},
            {"file_count": 18, "payload_count": 13, "payload_rows": 22860},
        )
    assert api.settings_calls == [False, True]
    assert verified == [
        (False, [release, parent]),
        (True, [release, parent]),
    ]
    assert "public_gate_failed_rolled_back_private" in parsed.state_log.read_text()


def test_receipt_failure_after_anonymous_gate_does_not_make_repository_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = args(tmp_path)
    parsed.state_log.touch()
    parent = parsed.expected_parent
    release = "a" * 40
    api = FakeApi([repo_info(parent, True), repo_info(release, True)])
    anonymous_receipt = tmp_path / "anonymous-receipt.json"
    anonymous_receipt.write_text("{}\n", encoding="utf-8")
    anonymous_log = tmp_path / "anonymous.log"
    anonymous_log.write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(
        PUBLISH,
        "read_only_private_preflight",
        lambda *values: {
            "repo": {"sha": parent},
            "verified_file_sha256": {"README.md": "b" * 64},
        },
    )
    monkeypatch.setattr(
        PUBLISH,
        "upload_or_recover_exact_commit",
        lambda *values: (release, {"repo": {"sha": release}}),
    )
    monkeypatch.setattr(
        PUBLISH,
        "verify_exact_release_state",
        lambda *values, **kwargs: {"repo": {"sha": release}},
    )
    monkeypatch.setattr(
        PUBLISH,
        "run_anonymous_verifier",
        lambda *values: (
            {"ok": True, "revision": release},
            anonymous_receipt,
            anonymous_log,
        ),
    )
    monkeypatch.setattr(
        PUBLISH,
        "write_json_new",
        lambda *values: (_ for _ in ()).throw(OSError("disk full")),
    )
    observer = SimpleNamespace(observed_hosts=[])
    with pytest.raises(PUBLISH.FullPublicationError, match="remote release passed"):
        PUBLISH.apply_publication(
            api,
            parsed,
            {"selected_leaf_sha256": "a" * 64},
            observer,
            {"README.md": "b" * 64, "validate_dataset.py": "c" * 64},
            {"file_count": 18, "payload_count": 13, "payload_rows": 22860},
        )
    assert api.settings_calls == [False]
    state = parsed.state_log.read_text()
    assert "remote_release_verified" in state
    assert "release_receipt_write_failed_remote_public_verified" in state
