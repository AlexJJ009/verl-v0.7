from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
from datetime import datetime, timedelta, timezone


ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = ROOT / "recipe/on_policy_wdl_sft/code_task/stage123_preflight.py"
    spec = importlib.util.spec_from_file_location("stage123_preflight", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def normalized_manifest(tmp_path: Path) -> Path:
    tool_path = ROOT / "scripts/experiment_manifest.py"
    spec = importlib.util.spec_from_file_location("experiment_manifest_preflight", tool_path)
    tool = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(tool)
    manifest = ROOT / "recipe/on_policy_wdl_sft/experiment_manifest/stage123.yaml"
    path = tmp_path / "normalized.json"
    path.write_text(json.dumps(tool.normalize(tool.load(manifest)), sort_keys=True))
    return path


def fake_machine_commands(module, monkeypatch):
    real_command = module.command

    def fake(*args: str, env=None):
        if len(args) > 1 and args[1].endswith("check_official_scorer_dependencies.py"):
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps({"ok": True, "imports": ["evalplus.evaluate"], "lcb_index": "/data-2/index.sqlite", "pythonpath": []}), stderr="")
        if args[0] in {"python3", "bash"}:
            return real_command(*args, env=env)
        if args[0] == "nvidia-smi":
            return subprocess.CompletedProcess(args, 0, stdout="\n".join(["NVIDIA L40S, 46068"] * 8) + "\n", stderr="")
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(module, "command", fake)


def test_scorer_dependency_check_is_structured_and_fail_closed(monkeypatch):
    module = load_module()
    complete = {"ok": True, "imports": ["evalplus.evaluate"], "lcb_index": "/data-2/index.sqlite", "pythonpath": ["/workspace/verl"]}
    monkeypatch.setattr(module, "command", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=json.dumps(complete), stderr=""))
    ok, detail = module.scorer_dependency_check()
    assert ok and detail == complete
    failure = {"ok": False, "failure_class": "dependency_failure", "error": "missing scorer"}
    monkeypatch.setattr(module, "command", lambda *args, **kwargs: subprocess.CompletedProcess(args, 2, stdout="", stderr=json.dumps(failure)))
    ok, detail = module.scorer_dependency_check()
    assert not ok and detail == failure


def host_facts(tmp_path: Path) -> Path:
    path = tmp_path / "host_facts.json"
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    path.write_text(json.dumps({
        "schema_version": 1,
        "artifact_type": "stage123_host_facts",
        "generated_at": now,
        "completed_at": now,
        "ok": True,
        "docker_image": {"reference": "verl-harness:latest", "immutable_id": "sha256:test"},
        "tmux": {"sessions": [], "stage123_conflicts": []},
        "mounts": {"checkpoint_mount": "/data-2/checkpoints"},
    }))
    return path


def test_load_host_facts_accepts_old_bound_evidence_and_rejects_invalid_schema(tmp_path: Path):
    module = load_module()
    facts_path = host_facts(tmp_path)
    payload = json.loads(facts_path.read_text())
    payload["completed_at"] = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat().replace("+00:00", "Z")
    facts_path.write_text(json.dumps(payload))

    facts, facts_sha256 = module.load_host_facts(facts_path)

    assert facts["ok"] is True
    assert len(facts_sha256) == 64
    payload["artifact_type"] = "wrong_schema"
    facts_path.write_text(json.dumps(payload))
    try:
        module.load_host_facts(facts_path)
    except ValueError as exc:
        assert "unsupported host facts schema" in str(exc)
    else:
        raise AssertionError("invalid host facts schema was accepted")


def test_preflight_binds_repo_model_identity_and_reports_pending_stage3(tmp_path, monkeypatch, capsys):
    module = load_module()
    normalized = normalized_manifest(tmp_path)
    fake_machine_commands(module, monkeypatch)
    monkeypatch.setattr(module.sys, "argv", ["stage123_preflight.py", "--allow-active", "--normalized-manifest", str(normalized), "--host-facts", str(host_facts(tmp_path))])
    # The container checkout is /workspace/verl, so unrelated topology checks may fail.
    # This fixture owns only the model-identity check.
    module.main()
    report = json.loads(capsys.readouterr().out)
    check = next(item for item in report["checks"] if item["name"] == "model_identity")
    assert check["ok"] is True
    assert check["detail"]["normalized_matches_repo"] is True
    assert check["detail"]["phases"]["stage1"]["materialized"] is True
    assert check["detail"]["phases"]["stage3"]["materialized"] is False


def test_preflight_rejects_normalized_stage1_identity_drift(tmp_path, monkeypatch, capsys):
    module = load_module()
    normalized = normalized_manifest(tmp_path)
    fake_machine_commands(module, monkeypatch)
    data = json.loads(normalized.read_text())
    data["calibration_workloads"]["stage1"]["model_sources"][0]["artifact_sha256"] = "0" * 64
    normalized.write_text(json.dumps(data, sort_keys=True))
    monkeypatch.setattr(module.sys, "argv", ["stage123_preflight.py", "--allow-active", "--normalized-manifest", str(normalized), "--host-facts", str(host_facts(tmp_path))])
    assert module.main() == 1
    report = json.loads(capsys.readouterr().out)
    check = next(item for item in report["checks"] if item["name"] == "model_identity")
    assert check["ok"] is False
    assert check["detail"]["normalized_matches_repo"] is False
