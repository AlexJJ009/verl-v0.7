from __future__ import annotations

import configparser
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import jsonschema
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PLATFORM = REPO_ROOT / "platform/hope_rebuttal_rlvr"


def load_submitter():
    path = PLATFORM / "submit_manifest.py"
    spec = importlib.util.spec_from_file_location("rebuttal_rlvr_submitter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SUBMITTER = load_submitter()


def test_handoff_registry_freezes_known_and_pending_evidence() -> None:
    registry = json.loads((PLATFORM / "handoff_registry.json").read_text())
    assert registry["initializations"]["R01"]["model_name"] == "R01_ORDINARY_SFT_4B_AM1P4M"
    assert registry["initializations"]["R01"]["sft_dataset"] == "AM-1.4M"
    assert registry["initializations"]["R02"]["sft_dataset"] == "AM-1.4M"
    assert "initialization_seed" in registry["initializations"]["R02"]["pending_external_provenance"]
    assert registry["downstream_rlvr"]["dataset_id"] == "hendrycks_math_7500"
    assert registry["downstream_rlvr"]["source_rows"] == 7500
    assert registry["gates"]["G1b"] == "passed"
    assert registry["gates"]["G0"] == "conditional_checkpoint_assumption_accepted_model_paths_pending"
    assert registry["gates"]["G3"].endswith("pending")
    assert registry["gates"]["G4"].endswith("pending")


def with_receipt_hash(value: dict) -> dict:
    value = dict(value)
    value["receipt_sha256"] = hashlib.sha256(SUBMITTER.canonical_json(value)).hexdigest()
    return value


def make_job(arm: str = "sft", init_pair: str = "I1", seed: int = 20260727) -> dict:
    root = "/mnt/dolphinfs/ssd_pool/test/lgx"
    dataset_root = f"{root}/datasets"
    model_root = f"{root}/models/rebuttal_rlvr/init"
    state_root = f"{root}/state"
    digest = "a" * 64
    return {
        "arm": arm,
        "init_pair": init_pair,
        "rl_seed": seed,
        "init_model_path": f"{model_root}/{arm}/{init_pair}",
        "paired_init_manifest": f"{root}/receipts/pairs/{init_pair}.json",
        "paired_init_manifest_hash": digest,
        "checkpoint_receipt": f"{root}/receipts/models/{arm}-{init_pair}.json",
        "checkpoint_receipt_hash": digest,
        "train_receipt": f"{root}/receipts/data/train.json",
        "train_receipt_hash": digest,
        "math7_receipt": f"{root}/receipts/data/math7.json",
        "math7_receipt_hash": digest,
        "grader_receipt": f"{root}/receipts/grader.json",
        "grader_receipt_hash": digest,
        "image_reference": "registry.example/verl@sha256:" + "b" * 64,
        "image_digest": "sha256:" + "b" * 64,
        "h20_profile_path": f"{root}/receipts/h20.json",
        "h20_profile_hash": digest,
        "h20_calibration_receipt": f"{root}/receipts/h20-calibration.json",
        "h20_calibration_receipt_hash": digest,
        "root": root,
        "dataset_root": dataset_root,
        "model_root": model_root,
        "state_root": state_root,
        "repo_subpath": "repos/verl-rebuttal-rlvr",
        "repo_commit": "c" * 40,
        "repo_submodule_receipt": f"{root}/receipts/submodules.json",
        "repo_submodule_receipt_hash": digest,
        "submitter_source_hash": digest,
        "algorithm_config_hash": digest,
        "eval_config_hash": digest,
        "path_override_receipt": f"{root}/receipts/paths.json",
        "path_override_receipt_hash": digest,
        "output_policy_version": "rebuttal-output-v1",
        "attempt_policy": "rebuttal-attempt-v1",
        "attempt_id": "a1",
        "retry_of": None,
        "run_mode": "formal",
        "allow_base_placeholder": False,
    }


def formal_manifest() -> dict:
    return {
        "schema_version": 1,
        "mode": "formal",
        "jobs": [make_job(arm, pair, seed) for pair in ("I1", "I2", "I3") for seed in SUBMITTER.RL_SEEDS for arm in ("sft", "wdl")],
    }


def pilot_manifest() -> dict:
    return {
        "schema_version": 1,
        "mode": "pilot",
        "jobs": [make_job(arm, "I1", seed) for seed in SUBMITTER.RL_SEEDS for arm in ("sft", "wdl")],
    }


def test_json_schema_is_valid_and_formal_matrix_is_complete() -> None:
    schema = json.loads((PLATFORM / "manifest.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    manifest = formal_manifest()
    jsonschema.validate(manifest, schema)
    SUBMITTER.validate_matrix(manifest)


def test_smoke_render_cannot_claim_formal_h20_admission() -> None:
    schema = json.loads((PLATFORM / "manifest.schema.json").read_text())
    job = make_job()
    job.update(
        {
            "run_mode": "smoke",
            "allow_base_placeholder": True,
            "h20_calibration_receipt": None,
            "h20_calibration_receipt_hash": None,
        }
    )
    manifest = {"schema_version": 1, "mode": "smoke", "jobs": [job]}
    jsonschema.validate(manifest, schema)
    SUBMITTER.validate_matrix(manifest)
    rendered, *_ = SUBMITTER.render_job(job)
    assert b"afo.app.env.H20_CALIBRATION_RECEIPT = NONE" in rendered

    job["h20_calibration_receipt"] = f"{job['root']}/receipts/fake-g4.json"
    job["h20_calibration_receipt_hash"] = "a" * 64
    with pytest.raises(SUBMITTER.ManifestError, match="must not claim"):
        SUBMITTER.validate_matrix({"schema_version": 1, "mode": "smoke", "jobs": [job]})


def test_submitter_requires_independently_signed_h20_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.on_policy_wdl_sft.test_rebuttal_rlvr_contract import write_signed_h20_bundle

    image_digest = "sha256:" + "b" * 64
    _profile, profile_path, admission_path, allowlist, _rendered = write_signed_h20_bundle(
        tmp_path, image_digest
    )
    monkeypatch.setattr(SUBMITTER, "G3_REVIEWER_ALLOWLIST_PATH", allowlist)
    job = make_job()
    job.update(
        {
            "root": str(tmp_path),
            "image_digest": image_digest,
            "h20_profile_path": str(profile_path),
            "h20_profile_hash": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
            "h20_calibration_receipt": str(admission_path),
            "h20_calibration_receipt_hash": hashlib.sha256(admission_path.read_bytes()).hexdigest(),
        }
    )
    SUBMITTER.validate_h20_calibration_admission(job)

    admission = json.loads(admission_path.read_text())
    payload = Path(admission["attestation_payload_path"])
    attestation = json.loads(payload.read_text())
    attestation["calibration_submitter_identity"] = "uid:2000:tampered-after-review"
    payload.write_bytes(SUBMITTER.canonical_json(attestation) + b"\n")
    admission.pop("receipt_sha256")
    admission["attestation_payload_sha256"] = hashlib.sha256(payload.read_bytes()).hexdigest()
    admission = with_receipt_hash(admission)
    admission_path.write_text(json.dumps(admission, sort_keys=True))
    with pytest.raises(SUBMITTER.ManifestError, match="signature"):
        SUBMITTER.validate_h20_calibration_admission(job)


def test_all_eighteen_cells_render_unique_strict_ini() -> None:
    manifest = formal_manifest()
    identities = set()
    for job in manifest["jobs"]:
        rendered, cell_hash, job_tag, app_name = SUBMITTER.render_job(job)
        assert b"REPLACE_" not in rendered
        assert len(cell_hash) == 64
        identities.add((job_tag, app_name))
        parser = configparser.ConfigParser(interpolation=None, strict=True)
        parser.optionxform = str
        parser.read_string(rendered.decode())
        assert parser["roles"]["worker.script"] == "bash jupyter.sh"
        assert parser["others"]["afo.app.env.CELL_HASH"] == cell_hash
        assert parser["others"]["afo.app.env.DATASET_ROOT"] == job["dataset_root"]
        assert parser["others"]["afo.app.env.MODEL_ROOT"] == job["model_root"]
        assert parser["others"]["afo.app.env.STATE_ROOT"] == job["state_root"]
        assert parser["others"]["afo.app.env.PAIRED_INIT_MANIFEST_HASH"] == job["paired_init_manifest_hash"]
        assert parser["others"]["afo.app.env.GRADER_RECEIPT_HASH"] == job["grader_receipt_hash"]
    assert len(identities) == 18


def test_all_six_pilot_cells_render_unique_strict_ini() -> None:
    schema = json.loads((PLATFORM / "manifest.schema.json").read_text())
    manifest = pilot_manifest()
    jsonschema.validate(manifest, schema)
    SUBMITTER.validate_matrix(manifest)
    identities = set()
    for job in manifest["jobs"]:
        rendered, cell_hash, job_tag, app_name = SUBMITTER.render_job(job)
        assert b"REPLACE_" not in rendered
        assert len(cell_hash) == 64
        identities.add((job_tag, app_name))
        parser = configparser.ConfigParser(interpolation=None, strict=True)
        parser.optionxform = str
        parser.read_string(rendered.decode())
        assert parser["roles"]["worker.script"] == "bash jupyter.sh"
        assert parser["others"]["afo.app.env.CELL_HASH"] == cell_hash
    assert len(identities) == 6


def test_formal_manifest_rejects_placeholder_and_duplicate_cell() -> None:
    manifest = formal_manifest()
    manifest["jobs"][0]["allow_base_placeholder"] = True
    with pytest.raises(SUBMITTER.ManifestError, match="forbid"):
        SUBMITTER.validate_matrix(manifest)

    manifest = formal_manifest()
    manifest["jobs"][-1] = dict(manifest["jobs"][0])
    with pytest.raises(SUBMITTER.ManifestError, match="duplicate"):
        SUBMITTER.validate_matrix(manifest)


@pytest.mark.parametrize("field", ["dataset_root", "model_root", "state_root"])
def test_manifest_rejects_controlled_root_outside_storage_boundary(field: str) -> None:
    manifest = formal_manifest()
    manifest["jobs"][0][field] = "/mnt/dolphinfs/ssd_pool/test/outside"
    with pytest.raises(SUBMITTER.ManifestError, match="strict children"):
        SUBMITTER.validate_matrix(manifest)


def test_formal_manifest_rejects_init_model_outside_model_root() -> None:
    manifest = formal_manifest()
    manifest["jobs"][0]["init_model_path"] = f"{manifest['jobs'][0]['root']}/other/model"
    with pytest.raises(SUBMITTER.ManifestError, match="below model_root"):
        SUBMITTER.validate_matrix(manifest)


def test_resolved_path_containment_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "repo-link").symlink_to(outside, target_is_directory=True)
    assert not SUBMITTER.resolved_path_is_under(root / "repo-link", root)
    assert SUBMITTER.resolved_path_is_under(root / "real-repo", root)


def test_renderer_rejects_ini_injection() -> None:
    job = make_job()
    job["init_model_path"] = job["init_model_path"] + "\nmalicious=true"
    with pytest.raises(SUBMITTER.ManifestError, match="unsafe INI"):
        SUBMITTER.render_job(job)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("init_model_path", "/mnt/dolphinfs/ssd_pool/test/lgx/../../escape"),
        ("dataset_root", "/mnt/dolphinfs/ssd_pool/test/lgx/../escape"),
        ("train_receipt", "/mnt/dolphinfs/ssd_pool/test/lgx/receipts/../escape.json"),
        ("repo_subpath", "repos/../other-checkout"),
    ],
)
def test_schema_and_matrix_reject_path_traversal(field: str, value: str) -> None:
    schema = json.loads((PLATFORM / "manifest.schema.json").read_text())
    manifest = formal_manifest()
    manifest["jobs"][0][field] = value
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(manifest, schema)
    with pytest.raises(SUBMITTER.ManifestError):
        SUBMITTER.validate_matrix(manifest)


def test_renderer_rejects_unapproved_template_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    template = tmp_path / "run.hope"
    template.write_text((PLATFORM / "run.hope").read_text() + "afo.app.env.UNAPPROVED = injected\n")
    monkeypatch.setattr(SUBMITTER, "TEMPLATE_PATH", template)
    with pytest.raises(SUBMITTER.ManifestError, match="allowlist"):
        SUBMITTER.render_job(make_job())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda text: text.replace("queue = root.shxs_training_cluster.hadoop-fridayagi.friday_h20_train", "queue = changed queue"),
        lambda text: text.replace("[resource]\n", "[resource]\nunapproved = value\n"),
        lambda text: text + "\n[unapproved]\nkey = value\n",
    ],
)
def test_renderer_rejects_any_non_allowlisted_ini_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate
) -> None:
    template = tmp_path / "run.hope"
    template.write_text(mutate((PLATFORM / "run.hope").read_text()))
    monkeypatch.setattr(SUBMITTER, "TEMPLATE_PATH", template)
    with pytest.raises(SUBMITTER.ManifestError, match="allowlist"):
        SUBMITTER.render_job(make_job())


def test_render_archive_binds_exact_manifest_and_environment(tmp_path: Path) -> None:
    manifest_bytes = b'{"schema_version":1,"mode":"formal","jobs":[]}\n'
    rendered = SUBMITTER.archive_render(make_job(), tmp_path, manifest_bytes=manifest_bytes)
    stage = Path(rendered["stage_dir"])
    archive = Path(rendered["archive_dir"])
    assert sorted(path.name for path in stage.iterdir()) == ["jupyter.sh", "run.hope"]
    assert (archive / "approved_manifest.json").read_bytes() == manifest_bytes
    assert rendered["approved_manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    environment = json.loads((archive / "resolved_environment.json").read_text())
    assert environment["afo.app.env.TRAIN_RECEIPT_HASH"] == "a" * 64
    assert environment["afo.app.env.EXPERIMENT"] == "R01"
    assert environment["afo.app.env.DATASET_ROOT"] == make_job()["dataset_root"]
    assert environment["afo.app.env.MODEL_ROOT"] == make_job()["model_root"]
    assert environment["afo.app.env.STATE_ROOT"] == make_job()["state_root"]
    assert rendered["run_hope_sha256"] == hashlib.sha256((stage / "run.hope").read_bytes()).hexdigest()


def test_ledger_allows_only_one_bound_retry(tmp_path: Path) -> None:
    job = make_job()
    first = SUBMITTER.archive_render(job, tmp_path / "first")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "attempt_id": "a1",
                "cell_hash": first["cell_hash"],
                "job_tag": first["job_tag"],
                "state": "FAILED",
            }
        )
        + "\n"
    )
    retry_job = dict(job, attempt_id="a2", retry_of="a1")
    retry = SUBMITTER.archive_render(retry_job, tmp_path / "retry")
    SUBMITTER.validate_ledger_admission(ledger, [retry])
    retry["retry_of"] = "wrong"
    with pytest.raises(SUBMITTER.ManifestError, match="latest failed"):
        SUBMITTER.validate_ledger_admission(ledger, [retry])


def semantics_receipt() -> dict:
    return with_receipt_hash(
        {
            "approved": True,
            "run_command": ["hope", "run", "run.hope"],
            "returns_after": "scheduler_acceptance",
            "job_id_regex": r"job_id=(?P<job_id>[A-Za-z0-9-]+)",
            "status_command": ["hope-status", "{job_id}"],
            "status_regex": r"(?P<state>QUEUED|RUNNING|SUCCEEDED|FAILED)",
            "status_stderr_must_be_empty": True,
            "active_list_command": ["hope-list"],
            "active_list_regex": r"^(?P<job_id>[A-Za-z0-9-]+) (?P<state>QUEUED|RUNNING|SUCCEEDED|FAILED) (?P<app_name>[A-Za-z0-9-]+)$",
            "active_list_empty_line": "NO ACTIVE JOBS",
            "active_list_header_lines": ["JOB_ID STATE APP_NAME"],
            "active_list_stderr_must_be_empty": True,
            "active_list_scope": "all_user_active_jobs",
            "submission_ledger_path": "/tmp/rebuttal-rlvr-global-submission-ledger.jsonl",
            "state_map": {
                "QUEUED": "QUEUED",
                "RUNNING": "RUNNING",
                "SUCCEEDED": "SUCCEEDED",
                "FAILED": "FAILED",
            },
        }
    )


def write_g3_bundle(
    tmp_path: Path,
    semantics_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    owner_identity: str = "uid:99999:independent-reviewer",
) -> Path:
    key = tmp_path / "reviewer_ed25519"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    public_key = (tmp_path / "reviewer_ed25519.pub").read_text().strip()
    allowlist = tmp_path / "g3_reviewer_keys.json"
    allowlist.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reviewers": [
                    {
                        "key_id": "reviewer-test-1",
                        "principal": "reviewer-test",
                        "owner_identity": owner_identity,
                        "public_key": public_key,
                    }
                ],
            },
            sort_keys=True,
        )
    )
    monkeypatch.setattr(SUBMITTER, "G3_REVIEWER_ALLOWLIST_PATH", allowlist)

    evidence = tmp_path / "g3-review.md"
    evidence.write_text("Independent review: PASS\n")
    smoke = with_receipt_hash(
        {
            "schema_version": 1,
            "receipt_kind": "hope_smoke_terminal",
            "state": "SUCCEEDED",
            "job_id": "job-smoke-1",
            "app_name": "rebuttal-rlvr-SMOKE-I1-r20260727",
            "hope_semantics_receipt_sha256": hashlib.sha256(semantics_path.read_bytes()).hexdigest(),
            "run_hope_sha256": "1" * 64,
            "jupyter_sha256": "2" * 64,
            "image_digest": "sha256:" + "3" * 64,
            "path_override_receipt_sha256": "4" * 64,
        }
    )
    smoke_path = tmp_path / "smoke-completion.json"
    smoke_path.write_text(json.dumps(smoke, sort_keys=True))
    attestation = {
        "schema_version": 1,
        "gate": "G3",
        "status": "passed",
        "approval_scope": "hope-g3-admission",
        "smoke_terminal_state": "SUCCEEDED",
        "path_image_smoke_passed": True,
        "worker_env_precedence_verified": True,
        "status_mapping_verified": True,
        "interrupt_reconciliation_verified": True,
        "hope_semantics_receipt_sha256": hashlib.sha256(semantics_path.read_bytes()).hexdigest(),
        "smoke_job_id": "job-smoke-1",
        "smoke_app_name": "rebuttal-rlvr-SMOKE-I1-r20260727",
        "submitter_identity": SUBMITTER.current_submitter_identity(),
        "reviewer_key_id": "reviewer-test-1",
        "review_evidence_path": str(evidence),
        "review_evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "smoke_completion_receipt_path": str(smoke_path),
        "smoke_completion_receipt_sha256": hashlib.sha256(smoke_path.read_bytes()).hexdigest(),
    }
    payload = tmp_path / "g3-attestation.json"
    payload.write_bytes(SUBMITTER.canonical_json(attestation) + b"\n")
    subprocess.run(
        ["ssh-keygen", "-Y", "sign", "-f", str(key), "-n", "rebuttal-rlvr-g3", str(payload)],
        check=True,
        capture_output=True,
    )
    signature = Path(str(payload) + ".sig")
    g3 = with_receipt_hash(
        {
            "approved": True,
            "schema_version": 1,
            "gate": "G3",
            "status": "passed",
            "approval_scope": "hope-g3-admission",
            "reviewer_key_id": "reviewer-test-1",
            "attestation_payload_path": str(payload),
            "attestation_payload_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            "attestation_signature_path": str(signature),
            "attestation_signature_sha256": hashlib.sha256(signature.read_bytes()).hexdigest(),
        }
    )
    g3_path = tmp_path / "g3.json"
    g3_path.write_text(json.dumps(g3, sort_keys=True))
    return g3_path


def test_formal_submission_requires_signed_independent_g3_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    semantics_path = tmp_path / "semantics.json"
    semantics_path.write_text(json.dumps(semantics_receipt(), sort_keys=True))
    with pytest.raises(SUBMITTER.ManifestError, match="G3"):
        SUBMITTER.load_submission_admission("formal", semantics_path, None)

    g3_path = write_g3_bundle(tmp_path, semantics_path, monkeypatch)
    semantics, admission = SUBMITTER.load_submission_admission("formal", semantics_path, g3_path)
    assert semantics["returns_after"] == "scheduler_acceptance"
    assert admission is not None and admission["gate"] == "G3"


def test_g3_rejects_reviewer_key_owned_by_live_submitter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    semantics_path = tmp_path / "semantics.json"
    semantics_path.write_text(json.dumps(semantics_receipt(), sort_keys=True))
    g3_path = write_g3_bundle(
        tmp_path,
        semantics_path,
        monkeypatch,
        owner_identity=SUBMITTER.current_submitter_identity(),
    )
    with pytest.raises(SUBMITTER.ManifestError, match="differ"):
        SUBMITTER.load_submission_admission("formal", semantics_path, g3_path)


def test_reconcile_counts_global_jobs_and_updates_old_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "attempt_id": "a1",
                "cell_hash": "c" * 64,
                "job_id": "old-1",
                "job_tag": "old-cell",
                "state": "RUNNING",
            }
        )
        + "\n"
    )
    semantics = semantics_receipt()

    def fake_run(command, **kwargs):
        if command[0] == "hope-list":
            return subprocess.CompletedProcess(command, 0, "global-1 RUNNING unrelated-app\n", "")
        if command[0] == "hope-status":
            return subprocess.CompletedProcess(command, 0, "SUCCEEDED\n", "")
        raise AssertionError(command)

    monkeypatch.setattr(SUBMITTER.subprocess, "run", fake_run)
    active = SUBMITTER.reconcile_active_jobs(ledger, semantics, timeout=1)
    assert set(active) == {"global-1"}
    assert SUBMITTER.read_ledger(ledger)[-1]["state"] == "SUCCEEDED"


def test_global_active_list_rejects_partial_parse_and_mixed_empty_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantics = semantics_receipt()
    outputs = iter(
        [
            "global-1 RUNNING app-1\nunrecognized platform row\n",
            "global-1 RUNNING app-1\nNO ACTIVE JOBS\n",
        ]
    )
    monkeypatch.setattr(
        SUBMITTER.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, next(outputs), ""),
    )
    with pytest.raises(SUBMITTER.ManifestError, match="unrecognized line"):
        SUBMITTER.list_platform_jobs(semantics, timeout=1)
    with pytest.raises(SUBMITTER.ManifestError, match="mixes"):
        SUBMITTER.list_platform_jobs(semantics, timeout=1)


def test_semantics_and_runtime_reject_overlapping_active_list_literals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    semantics = semantics_receipt()
    semantics.pop("receipt_sha256")
    semantics["active_list_empty_line"] = "job-1 RUNNING app-1"
    semantics = with_receipt_hash(semantics)
    path = tmp_path / "semantics.json"
    path.write_text(json.dumps(semantics, sort_keys=True))
    with pytest.raises(SUBMITTER.ManifestError, match="overlaps"):
        SUBMITTER.load_semantics(path)

    monkeypatch.setattr(
        SUBMITTER.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "job-1 RUNNING app-1\n", ""),
    )
    with pytest.raises(SUBMITTER.ManifestError, match="overlapping"):
        SUBMITTER.list_platform_jobs(semantics, timeout=1)


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    [
        ("SUCCEEDED\nUNPARSED ERROR DETAIL\n", ""),
        ("SUCCEEDED\n", "warning on stderr\n"),
    ],
)
def test_status_parser_rejects_mixed_or_stderr_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    stderr: str,
) -> None:
    semantics = semantics_receipt()
    monkeypatch.setattr(
        SUBMITTER.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout, stderr),
    )
    item = {
        "attempt_id": "a1",
        "cell_hash": "c" * 64,
        "job_id": "job-1",
        "job_tag": "cell-1",
        "state": "RUNNING",
    }
    with pytest.raises(SUBMITTER.ManifestError):
        SUBMITTER.query_state(item, semantics, timeout=1, ledger=tmp_path / "ledger.jsonl")


@pytest.mark.parametrize("tamper", ["extra", "run", "shim", "symlink"])
def test_submit_revalidates_exact_stage_before_hope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    render = SUBMITTER.archive_render(make_job(), tmp_path / "render")
    stage = Path(render["stage_dir"])
    if tamper == "extra":
        (stage / ".hidden").write_text("extra")
    elif tamper == "run":
        (stage / "run.hope").write_text("tampered")
    elif tamper == "shim":
        (stage / "jupyter.sh").write_text("tampered")
    else:
        (stage / "jupyter.sh").unlink()
        (stage / "jupyter.sh").symlink_to(stage / "run.hope")

    called = False

    def unexpected_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("Hope must not run after a stage-integrity failure")

    monkeypatch.setattr(SUBMITTER.subprocess, "run", unexpected_run)
    semantics = {
        "run_command": ["hope", "run", "run.hope"],
        "job_id_regex": r"job_id=(?P<job_id>[A-Za-z0-9-]+)",
        "_receipt_file_sha256": "d" * 64,
    }
    with pytest.raises(SUBMITTER.ManifestError, match="submission stage"):
        SUBMITTER.submit_one(render, semantics, timeout=1, ledger=tmp_path / "ledger.jsonl")
    assert not called


def test_submit_batch_waits_for_global_capacity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snapshots = iter([{"global-1": {"state": "RUNNING"}}, {}])
    submitted: list[str] = []
    sleeps: list[int] = []
    monkeypatch.setattr(SUBMITTER, "reconcile_active_jobs", lambda *args, **kwargs: next(snapshots))
    monkeypatch.setattr(SUBMITTER.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(
        SUBMITTER,
        "submit_one",
        lambda item, *args, **kwargs: submitted.append(item["job_tag"]) or item,
    )
    semantics = {"returns_after": "scheduler_acceptance"}
    SUBMITTER.submit_batch(
        [{"job_tag": "new-cell"}],
        semantics,
        max_active=1,
        submit_timeout=1,
        status_timeout=1,
        poll_seconds=7,
        ledger=tmp_path / "ledger.jsonl",
    )
    assert sleeps == [7]
    assert submitted == ["new-cell"]


def test_completion_semantics_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(SUBMITTER.ManifestError, match="completion-blocking"):
        SUBMITTER.submit_batch([], {"returns_after": "completion"}, 1, 1, 1, 1, tmp_path / "ledger")


def test_submission_timeout_is_unknown_not_retryable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    render = SUBMITTER.archive_render(make_job(), tmp_path / "render")
    semantics = {
        "run_command": ["hope", "run", "run.hope"],
        "job_id_regex": r"job_id=(?P<job_id>[A-Za-z0-9-]+)",
        "_receipt_file_sha256": "d" * 64,
    }
    monkeypatch.setattr(
        SUBMITTER.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(args[0], timeout=1, output="job_id=maybe-running")
        ),
    )
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(SUBMITTER.ManifestError, match="UNKNOWN"):
        SUBMITTER.submit_one(render, semantics, timeout=1, ledger=ledger)
    event = SUBMITTER.read_ledger(ledger)[-1]
    assert event["state"] == "UNKNOWN"
    assert event["job_id"] == "maybe-running"
    assert (Path(render["archive_dir"]) / "submitted.run.hope").is_file()
    assert (Path(render["archive_dir"]) / "submitted.jupyter.sh").is_file()
    retry = dict(render, attempt_id="a2", retry_of="a1")
    with pytest.raises(SUBMITTER.ManifestError, match="unknown"):
        SUBMITTER.validate_ledger_admission(ledger, [retry])
