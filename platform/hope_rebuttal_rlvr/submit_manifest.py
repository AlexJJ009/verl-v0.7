#!/usr/bin/env python3
"""Validate, render, and safely submit a manifest-driven Hope batch."""

from __future__ import annotations

import argparse
import configparser
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import pwd
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any

import jsonschema


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SCHEMA_PATH = HERE / "manifest.schema.json"
TEMPLATE_PATH = HERE / "run.hope"
SHIM_PATH = HERE / "jupyter.sh"
G3_REVIEWER_ALLOWLIST_PATH = HERE / "g3_reviewer_keys.json"
FROZEN_CONFIG_PATH = REPO_ROOT / "recipe/on_policy_wdl_sft/rebuttal_rlvr/frozen_grpo_v2.env"
H20_PROFILE_SCHEMA_PATH = REPO_ROOT / "recipe/on_policy_wdl_sft/rebuttal_rlvr/h20_profile.schema.json"
H20_CALIBRATION_SCHEMA_PATH = REPO_ROOT / "recipe/on_policy_wdl_sft/rebuttal_rlvr/h20_calibration_receipt.schema.json"
H20_TERMINAL_SCHEMA_PATH = REPO_ROOT / "recipe/on_policy_wdl_sft/rebuttal_rlvr/h20_calibration_terminal.schema.json"
H20_WORKER_EVIDENCE_SCHEMA_PATH = REPO_ROOT / "recipe/on_policy_wdl_sft/rebuttal_rlvr/h20_worker_evidence.schema.json"
RL_SEEDS = (20260727, 20260728, 20260729)
EXPERIMENT_BY_ARM = {"sft": "R01", "wdl": "R02"}
SAFE_VALUE = re.compile(r"^[A-Za-z0-9_./:@+-]+$")
TERMINAL_STATES = {"SUCCEEDED", "FAILED"}
ACTIVE_STATES = {"QUEUED", "RUNNING"}
FIXED_TEMPLATE_ENV = {
    "afo.app.env.YARN_CONTAINER_RUNTIME_DOCKER_SHM_SIZE_BYTES": "549755813888",
}
STATIC_INI = {
    "base": {"type": "ml-easy-job"},
    "resource": {
        "usergroup": "hadoop-ai-search",
        "queue": "root.shxs_training_cluster.hadoop-fridayagi.friday_h20_train",
    },
    "roles": {
        "workers": "1",
        "worker.memory": "1920000",
        "worker.vcore": "128",
        "worker.gcoresh20-141g": "8",
        "worker.script": "bash jupyter.sh",
    },
    "user_args": {},
    "am": {"afo.app.am.resource.mb": "4096"},
    "tensorboard": {"with.tensor.board": "false"},
    "data": {"afo.data.prefetch": "false"},
    "failover": {"afo.app.support.engine.failover": "false"},
    "others": {
        "client.git.revision.publish": "false",
        **FIXED_TEMPLATE_ENV,
        "with_requirements": "false",
        "afo.dolphinfs.otherusers": "hadoop-mtsearch-assistant",
        "afo.role.worker.task.attempt.max.retry": "0",
    },
}
BOUND_RECEIPT_BINDINGS = (
    ("paired_init_manifest", "paired_init_manifest_hash"),
    ("checkpoint_receipt", "checkpoint_receipt_hash"),
    ("train_receipt", "train_receipt_hash"),
    ("math7_receipt", "math7_receipt_hash"),
    ("grader_receipt", "grader_receipt_hash"),
    ("h20_profile_path", "h20_profile_hash"),
    ("h20_calibration_receipt", "h20_calibration_receipt_hash"),
    ("repo_submodule_receipt", "repo_submodule_receipt_hash"),
    ("path_override_receipt", "path_override_receipt_hash"),
)

CELL_FIELDS = (
    "arm",
    "init_pair",
    "rl_seed",
    "init_model_path",
    "paired_init_manifest",
    "paired_init_manifest_hash",
    "checkpoint_receipt",
    "checkpoint_receipt_hash",
    "train_receipt",
    "train_receipt_hash",
    "math7_receipt",
    "math7_receipt_hash",
    "grader_receipt",
    "grader_receipt_hash",
    "image_reference",
    "image_digest",
    "h20_profile_path",
    "h20_profile_hash",
    "h20_calibration_receipt",
    "h20_calibration_receipt_hash",
    "root",
    "repo_subpath",
    "repo_commit",
    "repo_submodule_receipt",
    "repo_submodule_receipt_hash",
    "submitter_source_hash",
    "algorithm_config_hash",
    "eval_config_hash",
    "path_override_receipt",
    "path_override_receipt_hash",
    "output_policy_version",
    "attempt_policy",
    "run_mode",
    "allow_base_placeholder",
)


class ManifestError(ValueError):
    pass


def canonical_json(value: Any) -> bytes:
    # The schema restricts execution-affecting values to strings, integers,
    # booleans, nulls, arrays, and objects; sorted compact JSON is canonical for
    # this subset and avoids float-number normalization ambiguity.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contains_replace_marker(value: Any) -> bool:
    if isinstance(value, str):
        return "REPLACE_" in value
    if isinstance(value, list):
        return any(contains_replace_marker(item) for item in value)
    if isinstance(value, dict):
        return any(contains_replace_marker(item) for item in value.values())
    return False


def is_safe_absolute_posix_path(value: str) -> bool:
    if not value.startswith("/") or value == "/" or value.endswith("/") or "//" in value:
        return False
    return all(component not in {"", ".", ".."} for component in value[1:].split("/"))


def is_safe_repo_subpath(value: str) -> bool:
    if value.startswith("/") or value.endswith("/") or "//" in value:
        return False
    return bool(value) and all(component not in {"", ".", ".."} for component in value.split("/"))


def path_is_under(path: str, root: str) -> bool:
    if not is_safe_absolute_posix_path(path) or not is_safe_absolute_posix_path(root):
        return False
    candidate = PurePosixPath(path)
    anchor = PurePosixPath(root)
    try:
        candidate.relative_to(anchor)
    except ValueError:
        return False
    return candidate != anchor


def load_manifest_bytes(payload: bytes, path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"cannot read manifest: {path}") from exc
    try:
        jsonschema.validate(raw, json.loads(SCHEMA_PATH.read_text()))
    except jsonschema.ValidationError as exc:
        raise ManifestError(f"manifest schema validation failed: {exc.message}") from exc
    if contains_replace_marker(raw):
        raise ManifestError("manifest contains unresolved REPLACE_ marker")
    return raw


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ManifestError(f"cannot read manifest: {path}") from exc
    return load_manifest_bytes(payload, path)


def validate_matrix(raw: dict[str, Any]) -> None:
    mode = raw["mode"]
    jobs = raw["jobs"]
    cells = [(job["arm"], job["init_pair"], job["rl_seed"]) for job in jobs]
    if len(cells) != len(set(cells)):
        raise ManifestError("duplicate scientific cell")

    if mode == "formal":
        pairs = sorted({job["init_pair"] for job in jobs})
        expected = {(arm, pair, seed) for arm in ("sft", "wdl") for pair in pairs for seed in RL_SEEDS}
        if len(pairs) != 3 or len(jobs) != 18 or set(cells) != expected:
            raise ManifestError("formal mode requires 2 arms x 3 init pairs x 3 frozen RL seeds")
    elif mode == "pilot":
        pairs = {job["init_pair"] for job in jobs}
        expected = {(arm, next(iter(pairs)), seed) for arm in ("sft", "wdl") for seed in RL_SEEDS} if len(pairs) == 1 else set()
        if len(jobs) != 6 or set(cells) != expected:
            raise ManifestError("pilot mode requires one init pair x 2 arms x 3 frozen RL seeds")

    for job in jobs:
        if not is_safe_repo_subpath(job["repo_subpath"]):
            raise ManifestError("repo_subpath contains an empty, dot, or parent component")
        if not path_is_under(f"{job['root']}/{job['repo_subpath']}", job["root"]):
            raise ManifestError("resolved repository path must be below ROOT")
        if mode in {"formal", "pilot"}:
            if job["run_mode"] != "formal" or job["allow_base_placeholder"]:
                raise ManifestError("formal/pilot manifests forbid placeholder or smoke rows")
            if not isinstance(job["h20_calibration_receipt"], str) or not isinstance(
                job["h20_calibration_receipt_hash"], str
            ):
                raise ManifestError("formal/pilot manifests require a signed H20 calibration receipt")
            formal_model_root = f"{job['root']}/models/rebuttal_rlvr/init"
            if not path_is_under(job["init_model_path"], formal_model_root):
                raise ManifestError("formal/pilot init_model_path must be below ROOT/models/rebuttal_rlvr/init")
        elif job["run_mode"] != "smoke":
            raise ManifestError("smoke manifest rows must use run_mode=smoke")
        elif job["h20_calibration_receipt"] is not None or job["h20_calibration_receipt_hash"] is not None:
            raise ManifestError("smoke manifests must not claim formal H20 calibration admission")

        path_fields = (
            "init_model_path",
            "paired_init_manifest",
            "checkpoint_receipt",
            "train_receipt",
            "math7_receipt",
            "grader_receipt",
            "h20_profile_path",
            "repo_submodule_receipt",
            "path_override_receipt",
        )
        optional_path_fields = ("h20_calibration_receipt",)
        outside = [field for field in path_fields if not path_is_under(job[field], job["root"])]
        outside.extend(
            field
            for field in optional_path_fields
            if job[field] is not None and not path_is_under(job[field], job["root"])
        )
        if outside:
            raise ManifestError(f"job paths must be below ROOT; outside fields: {outside}")


def validate_file_binding(job: dict[str, Any], path_field: str, hash_field: str) -> None:
    path = Path(job[path_field])
    if not path.is_file():
        raise ManifestError(f"bound file does not exist: {path}")
    actual = sha256_file(path)
    if actual != job[hash_field]:
        raise ManifestError(f"bound file hash mismatch: {path_field}")


def validate_live_bindings(raw: dict[str, Any]) -> None:
    expected_source_hash = sha256_file(Path(__file__).resolve())
    expected_config_hash = sha256_file(FROZEN_CONFIG_PATH)
    current_commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    current_status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_status:
        raise ManifestError("submission checkout must be clean and immutable")

    for job in raw["jobs"]:
        if job["submitter_source_hash"] != expected_source_hash:
            raise ManifestError("submitter_source_hash does not match checked-in submitter")
        if job["algorithm_config_hash"] != expected_config_hash:
            raise ManifestError("algorithm_config_hash does not match frozen_grpo_v2.env")
        if job["repo_commit"] != current_commit:
            raise ManifestError("manifest repo_commit does not match submission checkout")
        if not Path(job["init_model_path"]).is_dir():
            raise ManifestError(f"init model directory does not exist: {job['init_model_path']}")
        for path_field, hash_field in (
            ("paired_init_manifest", "paired_init_manifest_hash"),
            ("checkpoint_receipt", "checkpoint_receipt_hash"),
            ("train_receipt", "train_receipt_hash"),
            ("math7_receipt", "math7_receipt_hash"),
            ("grader_receipt", "grader_receipt_hash"),
            ("h20_profile_path", "h20_profile_hash"),
            ("h20_calibration_receipt", "h20_calibration_receipt_hash"),
            ("repo_submodule_receipt", "repo_submodule_receipt_hash"),
            ("path_override_receipt", "path_override_receipt_hash"),
        ):
            if job[path_field] is not None:
                validate_file_binding(job, path_field, hash_field)
        validate_h20_calibration_admission(job)


def cell_hash(job: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json({"schema_version": 1, **{field: job[field] for field in CELL_FIELDS}}))


def identities(job: dict[str, Any]) -> tuple[str, str, str]:
    digest = cell_hash(job)
    prefix = "SMOKE" if job["run_mode"] == "smoke" else job["arm"]
    job_tag = f"{prefix}-{job['init_pair']}-r{job['rl_seed']}-{digest[:12]}"
    app_name = f"rebuttal-rlvr-{job_tag}-{job['attempt_id']}"
    if len(app_name) > 120:
        raise ManifestError("rendered AFO app name exceeds 120 characters")
    return digest, job_tag, app_name


def new_parser() -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None, strict=True, empty_lines_in_values=False)
    parser.optionxform = str
    return parser


def controlled_environment(job: dict[str, Any], digest: str, job_tag: str) -> dict[str, str]:
    return {
        "afo.app.env.ROOT": job["root"],
        "afo.app.env.REPO_SUBPATH": job["repo_subpath"],
        "afo.app.env.REPO_COMMIT": job["repo_commit"],
        "afo.app.env.REPO_SUBMODULE_RECEIPT": job["repo_submodule_receipt"],
        "afo.app.env.REPO_SUBMODULE_RECEIPT_HASH": job["repo_submodule_receipt_hash"],
        "afo.app.env.SUBMITTER_SOURCE_HASH": job["submitter_source_hash"],
        "afo.app.env.ARM": job["arm"],
        "afo.app.env.EXPERIMENT": EXPERIMENT_BY_ARM[job["arm"]],
        "afo.app.env.INIT_PAIR": job["init_pair"],
        "afo.app.env.RLVR_SEED": str(job["rl_seed"]),
        "afo.app.env.INIT_MODEL_PATH": job["init_model_path"],
        "afo.app.env.PAIRED_INIT_MANIFEST": job["paired_init_manifest"],
        "afo.app.env.PAIRED_INIT_MANIFEST_HASH": job["paired_init_manifest_hash"],
        "afo.app.env.CHECKPOINT_RECEIPT": job["checkpoint_receipt"],
        "afo.app.env.CHECKPOINT_RECEIPT_HASH": job["checkpoint_receipt_hash"],
        "afo.app.env.TRAIN_RECEIPT": job["train_receipt"],
        "afo.app.env.TRAIN_RECEIPT_HASH": job["train_receipt_hash"],
        "afo.app.env.MATH7_RECEIPT": job["math7_receipt"],
        "afo.app.env.MATH7_RECEIPT_HASH": job["math7_receipt_hash"],
        "afo.app.env.GRADER_RECEIPT": job["grader_receipt"],
        "afo.app.env.GRADER_RECEIPT_HASH": job["grader_receipt_hash"],
        "afo.app.env.H20_PROFILE_PATH": job["h20_profile_path"],
        "afo.app.env.H20_PROFILE_HASH": job["h20_profile_hash"],
        "afo.app.env.H20_CALIBRATION_RECEIPT": job["h20_calibration_receipt"] or "NONE",
        "afo.app.env.H20_CALIBRATION_RECEIPT_HASH": job["h20_calibration_receipt_hash"] or "NONE",
        "afo.app.env.IMAGE_DIGEST": job["image_digest"],
        "afo.app.env.ALGORITHM_CONFIG_HASH": job["algorithm_config_hash"],
        "afo.app.env.EVAL_CONFIG_HASH": job["eval_config_hash"],
        "afo.app.env.PATH_OVERRIDE_RECEIPT": job["path_override_receipt"],
        "afo.app.env.PATH_OVERRIDE_RECEIPT_HASH": job["path_override_receipt_hash"],
        "afo.app.env.JOB_TAG": job_tag,
        "afo.app.env.CELL_HASH": digest,
        "afo.app.env.ATTEMPT_ID": job["attempt_id"],
        "afo.app.env.ATTEMPT_POLICY": job["attempt_policy"],
        "afo.app.env.OUTPUT_POLICY_VERSION": job["output_policy_version"],
        "afo.app.env.RUN_MODE": job["run_mode"],
        "afo.app.env.ALLOW_BASE_PLACEHOLDER": "1" if job["allow_base_placeholder"] else "0",
    }


def render_job(job: dict[str, Any]) -> tuple[bytes, str, str, str]:
    digest, job_tag, app_name = identities(job)
    parser = new_parser()
    with TEMPLATE_PATH.open(encoding="utf-8") as handle:
        parser.read_file(handle)

    parser["base"]["afo.app.name"] = app_name
    parser["docker"]["afo.docker.image.name"] = job["image_reference"]
    controlled = controlled_environment(job, digest, job_tag)
    for key, value in controlled.items():
        if not SAFE_VALUE.fullmatch(value):
            raise ManifestError(f"unsafe INI value for {key}: {value!r}")
        parser["others"][key] = value

    from io import StringIO

    output = StringIO()
    parser.write(output, space_around_delimiters=True)
    rendered = output.getvalue().encode("utf-8")
    if b"REPLACE_" in rendered:
        raise ManifestError("rendered run.hope contains unresolved placeholder")

    verification = new_parser()
    verification.read_string(rendered.decode("utf-8"))
    expected = {
        **STATIC_INI,
        "base": {"afo.app.name": app_name, **STATIC_INI["base"]},
        "docker": {"afo.docker.image.name": job["image_reference"]},
        "others": {**STATIC_INI["others"], **controlled},
    }
    actual = {section: dict(verification[section].items()) for section in verification.sections()}
    if actual != expected:
        unexpected_sections = sorted(set(actual) - set(expected))
        missing_sections = sorted(set(expected) - set(actual))
        changed_sections = sorted(section for section in set(actual) & set(expected) if actual[section] != expected[section])
        raise ManifestError(
            "rendered INI allowlist mismatch; "
            f"unexpected_sections={unexpected_sections}, missing_sections={missing_sections}, "
            f"changed_sections={changed_sections}"
        )
    for section, options in actual.items():
        for key, value in options.items():
            if (section, key) == ("roles", "worker.script"):
                if value != "bash jupyter.sh":
                    raise ManifestError("worker.script must equal exactly 'bash jupyter.sh'")
            elif not SAFE_VALUE.fullmatch(value):
                raise ManifestError(f"unsafe static INI value for [{section}] {key}: {value!r}")
    return rendered, digest, job_tag, app_name


def archive_render(
    job: dict[str, Any],
    output_root: Path,
    manifest_bytes: bytes | None = None,
    archive_bound_files: bool = False,
) -> dict[str, Any]:
    rendered, digest, job_tag, app_name = render_job(job)
    target = output_root / job_tag / "attempts" / job["attempt_id"]
    stage = target / "stage"
    archive = target / "archive"
    stage.mkdir(parents=True, exist_ok=False)
    archive.mkdir()
    if manifest_bytes is None:
        manifest_bytes = canonical_json(
            {"schema_version": 1, "mode": job["run_mode"], "jobs": [job]}
        ) + b"\n"
    manifest_sha256 = sha256_bytes(manifest_bytes)
    (archive / "approved_manifest.json").write_bytes(manifest_bytes)
    (stage / "run.hope").write_bytes(rendered)
    shutil.copyfile(SHIM_PATH, stage / "jupyter.sh")
    canonical_cell = {"schema_version": 1, **{field: job[field] for field in CELL_FIELDS}}
    (archive / "canonical_cell.json").write_bytes(canonical_json(canonical_cell) + b"\n")
    resolved_environment = {
        **FIXED_TEMPLATE_ENV,
        **controlled_environment(job, digest, job_tag),
    }
    (archive / "resolved_environment.json").write_bytes(canonical_json(resolved_environment) + b"\n")
    archived_receipts: dict[str, str] = {}
    if archive_bound_files:
        receipt_dir = archive / "bound_receipts"
        receipt_dir.mkdir()
        for field, hash_field in BOUND_RECEIPT_BINDINGS:
            if job[field] is None:
                continue
            destination = receipt_dir / f"{field}.json"
            shutil.copyfile(Path(job[field]), destination)
            archived_hash = sha256_file(destination)
            if archived_hash != job[hash_field]:
                raise ManifestError(f"bound receipt changed during archive: {field}")
            archived_receipts[field] = archived_hash
        archived_receipts.update(
            {
                f"h20_evidence/{label}": digest
                for label, digest in archive_h20_calibration_evidence(job, archive / "h20_calibration_evidence").items()
            }
        )
    metadata = {
        "app_name": app_name,
        "attempt_id": job["attempt_id"],
        "attempt_policy": job["attempt_policy"],
        "algorithm_config_sha256": job["algorithm_config_hash"],
        "approved_manifest_sha256": manifest_sha256,
        "archived_receipt_sha256": archived_receipts,
        "cell_hash": digest,
        "job_tag": job_tag,
        "jupyter_sha256": sha256_file(stage / "jupyter.sh"),
        "resolved_environment_sha256": sha256_file(archive / "resolved_environment.json"),
        "retry_of": job["retry_of"],
        "run_hope_sha256": sha256_file(stage / "run.hope"),
        "submitter_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    (archive / "render_receipt.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {
        **metadata,
        "archive_dir": str(archive),
        "attempt_dir": str(target),
        "stage_dir": str(stage),
    }


def load_hash_valid_receipt(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read {label} receipt: {path}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"{label} receipt root must be an object")
    receipt_hash = value.get("receipt_sha256")
    actual = sha256_bytes(canonical_json({k: v for k, v in value.items() if k != "receipt_sha256"}))
    if receipt_hash != actual or value.get("approved") is not True:
        raise ManifestError(f"{label} receipt is not approved or hash-valid")
    return value


def load_self_hashed_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"{label} root must be an object")
    receipt_hash = value.get("receipt_sha256")
    actual = sha256_bytes(canonical_json({k: v for k, v in value.items() if k != "receipt_sha256"}))
    if receipt_hash != actual:
        raise ManifestError(f"{label} is not self-hash-valid")
    return value


def current_submitter_identity() -> str:
    uid = os.getuid()
    return f"uid:{uid}:{pwd.getpwuid(uid).pw_name}"


def load_reviewer_key(key_id: str) -> dict[str, str]:
    try:
        value = json.loads(G3_REVIEWER_ALLOWLIST_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError("cannot read the checked-in G3 reviewer-key allowlist") from exc
    if not isinstance(value, dict) or set(value) != {"schema_version", "reviewers"} or value["schema_version"] != 1:
        raise ManifestError("G3 reviewer-key allowlist has an invalid root schema")
    reviewers = value["reviewers"]
    if not isinstance(reviewers, list):
        raise ManifestError("G3 reviewer-key allowlist reviewers must be a list")
    found = []
    seen: set[str] = set()
    for reviewer in reviewers:
        required = {"key_id", "principal", "owner_identity", "public_key"}
        if not isinstance(reviewer, dict) or set(reviewer) != required:
            raise ManifestError("G3 reviewer-key allowlist entry has an invalid schema")
        if reviewer["key_id"] in seen:
            raise ManifestError("G3 reviewer-key allowlist contains a duplicate key_id")
        seen.add(reviewer["key_id"])
        if not re.fullmatch(r"[A-Za-z0-9_.@+-]+", reviewer["key_id"]):
            raise ManifestError("G3 reviewer key_id is unsafe")
        if not re.fullmatch(r"[A-Za-z0-9_.@+-]+", reviewer["principal"]):
            raise ManifestError("G3 reviewer principal is unsafe")
        if not isinstance(reviewer["owner_identity"], str) or not reviewer["owner_identity"]:
            raise ManifestError("G3 reviewer owner_identity is invalid")
        if not re.fullmatch(r"ssh-ed25519 [A-Za-z0-9+/=]+(?: [^\r\n]+)?", reviewer["public_key"]):
            raise ManifestError("G3 reviewer public key must be one SSH Ed25519 line")
        if reviewer["key_id"] == key_id:
            found.append(reviewer)
    if len(found) != 1:
        raise ManifestError(f"G3 reviewer key is not independently allowlisted: {key_id}")
    return found[0]


def verify_reviewer_signature(
    payload_path: Path,
    signature_path: Path,
    reviewer: dict[str, str],
    namespace: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="g3-verify-") as temporary:
        allowed_signers = Path(temporary) / "allowed_signers"
        allowed_signers.write_text(f"{reviewer['principal']} {reviewer['public_key']}\n")
        try:
            result = subprocess.run(
                [
                    "ssh-keygen",
                    "-Y",
                    "verify",
                    "-f",
                    str(allowed_signers),
                    "-I",
                    reviewer["principal"],
                    "-n",
                    namespace,
                    "-s",
                    str(signature_path),
                ],
                input=payload_path.read_bytes(),
                capture_output=True,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ManifestError("cannot execute G3 detached-signature verification") from exc
    if result.returncode != 0:
        raise ManifestError(f"detached reviewer signature is invalid for namespace {namespace}")


def validate_schema(value: dict[str, Any], schema_path: Path, label: str) -> None:
    try:
        jsonschema.validate(value, json.loads(schema_path.read_text()))
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        detail = exc.message if isinstance(exc, jsonschema.ValidationError) else str(exc)
        raise ManifestError(f"{label} schema validation failed: {detail}") from exc


def canonical_projection_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def h20_resources_from_ini(path: Path) -> tuple[dict[str, Any], str]:
    parser = new_parser()
    try:
        with path.open(encoding="utf-8") as handle:
            parser.read_file(handle)
    except (OSError, configparser.Error) as exc:
        raise ManifestError(f"cannot strictly parse staged H20 run.hope: {path}") from exc

    required_keys = {
        "resource": {"usergroup", "queue"},
        "roles": {"workers", "worker.memory", "worker.vcore", "worker.gcoresh20-141g", "worker.script"},
        "failover": {"afo.app.support.engine.failover"},
    }
    for section, keys in required_keys.items():
        if section not in parser or set(parser[section]) != keys:
            raise ManifestError(f"staged H20 run.hope has an unexpected [{section}] resource surface")
    for key in set(FIXED_TEMPLATE_ENV) | {"afo.role.worker.task.attempt.max.retry"}:
        if key not in parser["others"]:
            raise ManifestError(f"staged H20 run.hope is missing [others] {key}")
    if "docker" not in parser or set(parser["docker"]) != {"afo.docker.image.name"}:
        raise ManifestError("staged H20 run.hope has an unexpected [docker] surface")

    def integer(section: str, key: str) -> int:
        raw = parser[section][key]
        if not re.fullmatch(r"0|[1-9][0-9]*", raw):
            raise ManifestError(f"staged H20 run.hope has a non-canonical integer: [{section}] {key}")
        return int(raw)

    failover = parser["failover"]["afo.app.support.engine.failover"]
    if failover not in {"true", "false"}:
        raise ManifestError("staged H20 failover value is not canonical boolean text")
    resources = {
        "usergroup": parser["resource"]["usergroup"],
        "queue": parser["resource"]["queue"],
        "workers": integer("roles", "workers"),
        "worker_memory_mb": integer("roles", "worker.memory"),
        "worker_vcore": integer("roles", "worker.vcore"),
        "gpu_resource_key": "worker.gcoresh20-141g",
        "gpu_count": integer("roles", "worker.gcoresh20-141g"),
        "worker_script": parser["roles"]["worker.script"],
        "shm_size_bytes": integer("others", "afo.app.env.YARN_CONTAINER_RUNTIME_DOCKER_SHM_SIZE_BYTES"),
        "max_retry": integer("others", "afo.role.worker.task.attempt.max.retry"),
        "failover": failover == "true",
    }
    return resources, parser["docker"]["afo.docker.image.name"]


def validate_bound_evidence(binding: dict[str, Any], root: str, label: str) -> Path:
    path = Path(str(binding.get("path", "")))
    if not path_is_under(str(path), root) or not path.is_file():
        raise ManifestError(f"{label} is missing or outside ROOT")
    if sha256_file(path) != binding.get("sha256"):
        raise ManifestError(f"{label} hash mismatch")
    return path


def validate_h20_calibration_admission(job: dict[str, Any]) -> None:
    profile_path = Path(job["h20_profile_path"])
    profile = load_self_hashed_json(profile_path, "H20 profile")
    validate_schema(profile, H20_PROFILE_SCHEMA_PATH, "H20 profile")
    if profile["image_digest"] != job["image_digest"]:
        raise ManifestError("H20 profile image digest differs from the manifest")

    if job["run_mode"] == "smoke":
        if profile["profile_status"] != "smoke_candidate":
            raise ManifestError("smoke jobs require profile_status=smoke_candidate")
        return
    if profile["profile_status"] != "formal_frozen":
        raise ManifestError("formal jobs require profile_status=formal_frozen")

    admission_path = Path(str(job["h20_calibration_receipt"]))
    admission = load_self_hashed_json(admission_path, "H20 calibration admission")
    validate_schema(admission, H20_CALIBRATION_SCHEMA_PATH, "H20 calibration admission")
    if Path(admission["h20_profile_path"]).resolve() != profile_path.resolve():
        raise ManifestError("H20 calibration admission points at a different profile")
    if admission["h20_profile_sha256"] != sha256_file(profile_path):
        raise ManifestError("H20 calibration admission profile hash mismatch")

    terminals: dict[str, tuple[Path, dict[str, Any]]] = {}
    for arm in ("sft", "wdl"):
        terminal_path = validate_bound_evidence(admission["terminal_receipts"][arm], job["root"], f"H20 {arm} terminal receipt")
        terminal = load_self_hashed_json(terminal_path, f"H20 {arm} terminal receipt")
        validate_schema(terminal, H20_TERMINAL_SCHEMA_PATH, f"H20 {arm} terminal receipt")
        if terminal["arm"] != arm or terminal["calibration_id"] != admission["calibration_id"]:
            raise ManifestError(f"H20 {arm} terminal receipt identity mismatch")
        for field in ("image_digest", "runtime_versions", "platform_resources", "selected", "fixed"):
            if terminal[field] != profile[field]:
                raise ManifestError(f"H20 {arm} terminal receipt differs on {field}")
        expected_metrics = profile["arm_metrics"][arm]
        if any(terminal["metrics"][key] != value for key, value in expected_metrics.items()):
            raise ManifestError(f"H20 {arm} terminal metrics differ from the frozen profile")
        staged = validate_bound_evidence(terminal["staged_run_hope"], job["root"], f"H20 {arm} staged run.hope")
        resources, image_reference = h20_resources_from_ini(staged)
        if resources != profile["platform_resources"]:
            raise ManifestError(f"H20 {arm} staged run.hope resource profile mismatch")
        if not image_reference.endswith("@" + profile["image_digest"]):
            raise ManifestError(f"H20 {arm} staged run.hope image digest mismatch")
        validate_bound_evidence(terminal["status_evidence"], job["root"], f"H20 {arm} status evidence")
        worker_path = validate_bound_evidence(
            terminal["worker_evidence"], job["root"], f"H20 {arm} worker evidence"
        )
        worker = load_self_hashed_json(worker_path, f"H20 {arm} worker evidence")
        validate_schema(worker, H20_WORKER_EVIDENCE_SCHEMA_PATH, f"H20 {arm} worker evidence")
        expected_worker_identity = {
            "calibration_id": admission["calibration_id"],
            "arm": arm,
            "job_id": terminal["job_id"],
            "image_digest": profile["image_digest"],
        }
        worker_identity_mismatches = [
            key for key, wanted in expected_worker_identity.items() if worker.get(key) != wanted
        ]
        if worker_identity_mismatches:
            raise ManifestError(f"H20 {arm} worker evidence identity mismatch: {worker_identity_mismatches}")
        if worker["runtime_versions"] != terminal["runtime_versions"]:
            raise ManifestError(f"H20 {arm} worker runtime differs from the terminal receipt")
        if worker["runtime_versions_sha256"] != canonical_projection_hash(terminal["runtime_versions"]):
            raise ManifestError(f"H20 {arm} worker runtime_versions hash mismatch")
        if worker["metrics"] != terminal["metrics"]:
            raise ManifestError(f"H20 {arm} worker metrics differ from the terminal receipt")
        terminals[arm] = (terminal_path, terminal)

    if terminals["sft"][1]["job_id"] == terminals["wdl"][1]["job_id"]:
        raise ManifestError("H20 calibration arms cannot reuse one platform job")
    if terminals["sft"][1]["app_name"] == terminals["wdl"][1]["app_name"]:
        raise ManifestError("H20 calibration arms cannot reuse one app identity")

    payload_binding = {
        "path": admission["attestation_payload_path"],
        "sha256": admission["attestation_payload_sha256"],
    }
    signature_binding = {
        "path": admission["attestation_signature_path"],
        "sha256": admission["attestation_signature_sha256"],
    }
    payload_path = validate_bound_evidence(payload_binding, job["root"], "H20 calibration attestation payload")
    signature_path = validate_bound_evidence(signature_binding, job["root"], "H20 calibration attestation signature")
    try:
        attestation = json.loads(payload_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError("H20 calibration attestation payload is unreadable") from exc
    expected_fields = {
        "schema_version",
        "gate",
        "status",
        "approval_scope",
        "calibration_id",
        "h20_profile_path",
        "h20_profile_sha256",
        "sft_terminal_receipt_path",
        "sft_terminal_receipt_sha256",
        "wdl_terminal_receipt_path",
        "wdl_terminal_receipt_sha256",
        "image_digest",
        "platform_resources_sha256",
        "runtime_versions_sha256",
        "selected_system_knobs_sha256",
        "fixed_system_knobs_sha256",
        "selection_policy_version",
        "calibration_submitter_identity",
        "reviewer_key_id",
        "review_evidence_path",
        "review_evidence_sha256",
    }
    if not isinstance(attestation, dict) or set(attestation) != expected_fields:
        raise ManifestError("H20 calibration attestation has unexpected or missing fields")
    expected_values = {
        "schema_version": 1,
        "gate": "G4",
        "status": "passed",
        "approval_scope": "rebuttal-h20-common-v1",
        "calibration_id": admission["calibration_id"],
        "h20_profile_path": str(profile_path),
        "h20_profile_sha256": sha256_file(profile_path),
        "sft_terminal_receipt_path": str(terminals["sft"][0]),
        "sft_terminal_receipt_sha256": sha256_file(terminals["sft"][0]),
        "wdl_terminal_receipt_path": str(terminals["wdl"][0]),
        "wdl_terminal_receipt_sha256": sha256_file(terminals["wdl"][0]),
        "image_digest": profile["image_digest"],
        "platform_resources_sha256": canonical_projection_hash(profile["platform_resources"]),
        "runtime_versions_sha256": canonical_projection_hash(profile["runtime_versions"]),
        "selected_system_knobs_sha256": canonical_projection_hash(profile["selected"]),
        "fixed_system_knobs_sha256": canonical_projection_hash(profile["fixed"]),
        "selection_policy_version": "rebuttal-h20-common-selection-v1",
        "reviewer_key_id": admission["reviewer_key_id"],
    }
    mismatches = [key for key, wanted in expected_values.items() if attestation.get(key) != wanted]
    if mismatches:
        raise ManifestError(f"H20 calibration attestation binding mismatch: {mismatches}")
    review_evidence = validate_bound_evidence(
        {"path": attestation["review_evidence_path"], "sha256": attestation["review_evidence_sha256"]},
        job["root"],
        "H20 calibration review evidence",
    )
    if not review_evidence.read_bytes():
        raise ManifestError("H20 calibration review evidence is empty")
    reviewer = load_reviewer_key(admission["reviewer_key_id"])
    if reviewer["owner_identity"] == attestation["calibration_submitter_identity"]:
        raise ManifestError("H20 calibration reviewer key owner must differ from the calibration submitter")
    verify_reviewer_signature(payload_path, signature_path, reviewer, "rebuttal-rlvr-g4")


def archive_h20_calibration_evidence(job: dict[str, Any], destination: Path) -> dict[str, str]:
    if job["h20_calibration_receipt"] is None:
        return {}
    admission = load_self_hashed_json(Path(job["h20_calibration_receipt"]), "H20 calibration admission")
    destination.mkdir()
    bindings: dict[str, dict[str, Any]] = {
        "attestation_payload": {
            "path": admission["attestation_payload_path"],
            "sha256": admission["attestation_payload_sha256"],
        },
        "attestation_signature": {
            "path": admission["attestation_signature_path"],
            "sha256": admission["attestation_signature_sha256"],
        },
        "sft_terminal_receipt": admission["terminal_receipts"]["sft"],
        "wdl_terminal_receipt": admission["terminal_receipts"]["wdl"],
    }
    payload = json.loads(Path(admission["attestation_payload_path"]).read_text())
    bindings["review_evidence"] = {
        "path": payload["review_evidence_path"],
        "sha256": payload["review_evidence_sha256"],
    }
    for arm in ("sft", "wdl"):
        terminal = load_self_hashed_json(Path(admission["terminal_receipts"][arm]["path"]), f"H20 {arm} terminal")
        for label in ("staged_run_hope", "status_evidence", "worker_evidence"):
            bindings[f"{arm}_{label}"] = terminal[label]

    archived: dict[str, str] = {}
    for label, binding in bindings.items():
        source = validate_bound_evidence(binding, job["root"], f"H20 archive source {label}")
        target = destination / f"{label}.bin"
        shutil.copyfile(source, target)
        digest = sha256_file(target)
        if digest != binding["sha256"]:
            raise ManifestError(f"H20 calibration evidence changed during archive: {label}")
        archived[label] = digest
    return archived


def require_named_groups(pattern: str, groups: set[str], label: str) -> None:
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ManifestError(f"{label} regex is invalid") from exc
    missing = groups - set(compiled.groupindex)
    if missing:
        raise ManifestError(f"{label} regex lacks named groups: {sorted(missing)}")


def load_semantics(path: Path) -> dict[str, Any]:
    value = load_hash_valid_receipt(path, "Hope semantics")
    if value.get("run_command") != ["hope", "run", "run.hope"]:
        raise ManifestError("Hope semantics run_command must be exactly hope run run.hope")
    if value.get("returns_after") not in {"scheduler_acceptance", "completion"}:
        raise ManifestError("Hope semantics returns_after is invalid")
    if not isinstance(value.get("job_id_regex"), str):
        raise ManifestError("Hope semantics job_id_regex is required")
    require_named_groups(value["job_id_regex"], {"job_id"}, "job-ID")
    if value["returns_after"] == "scheduler_acceptance":
        status_command = value.get("status_command")
        if not isinstance(status_command, list) or not all(isinstance(part, str) for part in status_command):
            raise ManifestError("scheduler-acceptance semantics require a string-list status_command")
        if not any("{job_id}" in part for part in status_command):
            raise ManifestError("scheduler-acceptance semantics require status_command with {job_id}")
        active_list_command = value.get("active_list_command")
        if not isinstance(active_list_command, list) or not all(isinstance(part, str) for part in active_list_command):
            raise ManifestError("scheduler-acceptance semantics require active_list_command")
        if not isinstance(value.get("status_regex"), str) or not isinstance(value.get("active_list_regex"), str):
            raise ManifestError("scheduler-acceptance semantics require status and active-list regexes")
        if value.get("status_stderr_must_be_empty") is not True:
            raise ManifestError("status_stderr_must_be_empty must be explicitly true")
        empty_line = value.get("active_list_empty_line")
        if not isinstance(empty_line, str) or not empty_line.strip() or "\n" in empty_line or "\r" in empty_line:
            raise ManifestError("active_list_empty_line must be one non-empty literal line")
        header_lines = value.get("active_list_header_lines", [])
        if (
            not isinstance(header_lines, list)
            or not all(isinstance(line, str) and line.strip() and "\n" not in line and "\r" not in line for line in header_lines)
            or len(header_lines) != len(set(header_lines))
        ):
            raise ManifestError("active_list_header_lines must contain unique non-empty literal lines")
        if empty_line in header_lines:
            raise ManifestError("active-list empty and header literals must be disjoint")
        if value.get("active_list_stderr_must_be_empty") is not True:
            raise ManifestError("active_list_stderr_must_be_empty must be explicitly true")
        if not isinstance(value.get("state_map"), dict):
            raise ManifestError("scheduler-acceptance semantics require status_regex and state_map")
        require_named_groups(value["status_regex"], {"state"}, "status")
        require_named_groups(value["active_list_regex"], {"job_id", "state"}, "active-list")
        row_pattern = re.compile(value["active_list_regex"])
        overlapping_literals = [line for line in [empty_line, *header_lines] if row_pattern.fullmatch(line) is not None]
        if overlapping_literals:
            raise ManifestError("active-list row regex overlaps an empty/header literal")
        mapped_states = set(value["state_map"].values())
        if not mapped_states or not mapped_states <= ACTIVE_STATES | TERMINAL_STATES:
            raise ManifestError("Hope state_map contains unsupported canonical states")
        if value.get("active_list_scope") != "all_user_active_jobs":
            raise ManifestError("active_list_scope must be all_user_active_jobs")
        ledger_path = value.get("submission_ledger_path")
        if not isinstance(ledger_path, str) or not is_safe_absolute_posix_path(ledger_path):
            raise ManifestError("Hope semantics must freeze one absolute submission_ledger_path")
    return value


def load_g3_admission(path: Path, semantics_path: Path) -> dict[str, Any]:
    value = load_hash_valid_receipt(path, "G3 admission")
    required_receipt_fields = {
        "approved",
        "schema_version",
        "gate",
        "status",
        "approval_scope",
        "reviewer_key_id",
        "attestation_payload_path",
        "attestation_payload_sha256",
        "attestation_signature_path",
        "attestation_signature_sha256",
        "receipt_sha256",
    }
    if set(value) != required_receipt_fields:
        raise ManifestError("G3 admission receipt has unexpected or missing fields")
    required_exact = {
        "schema_version": 1,
        "gate": "G3",
        "status": "passed",
        "approval_scope": "hope-g3-admission",
    }
    if any(value.get(key) != wanted for key, wanted in required_exact.items()):
        raise ManifestError("G3 admission receipt has an invalid gate identity")

    bound_paths: dict[str, Path] = {}
    for path_field, hash_field in (
        ("attestation_payload_path", "attestation_payload_sha256"),
        ("attestation_signature_path", "attestation_signature_sha256"),
    ):
        candidate = Path(str(value[path_field]))
        if not candidate.is_absolute() or not candidate.is_file():
            raise ManifestError(f"G3 bound file is missing or non-absolute: {path_field}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(value[hash_field])) or sha256_file(candidate) != value[hash_field]:
            raise ManifestError(f"G3 bound file hash mismatch: {path_field}")
        bound_paths[path_field] = candidate

    try:
        attestation = json.loads(bound_paths["attestation_payload_path"].read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError("G3 attestation payload is unreadable") from exc
    required_attestation_fields = {
        "schema_version",
        "gate",
        "status",
        "approval_scope",
        "smoke_terminal_state",
        "path_image_smoke_passed",
        "worker_env_precedence_verified",
        "status_mapping_verified",
        "interrupt_reconciliation_verified",
        "hope_semantics_receipt_sha256",
        "smoke_job_id",
        "smoke_app_name",
        "submitter_identity",
        "reviewer_key_id",
        "review_evidence_path",
        "review_evidence_sha256",
        "smoke_completion_receipt_path",
        "smoke_completion_receipt_sha256",
    }
    if not isinstance(attestation, dict) or set(attestation) != required_attestation_fields:
        raise ManifestError("G3 signed attestation has unexpected or missing fields")
    attestation_exact = {
        "schema_version": 1,
        "gate": "G3",
        "status": "passed",
        "approval_scope": "hope-g3-admission",
        "smoke_terminal_state": "SUCCEEDED",
        "path_image_smoke_passed": True,
        "worker_env_precedence_verified": True,
        "status_mapping_verified": True,
        "interrupt_reconciliation_verified": True,
        "hope_semantics_receipt_sha256": sha256_file(semantics_path),
        "reviewer_key_id": value["reviewer_key_id"],
        "submitter_identity": current_submitter_identity(),
    }
    mismatches = [key for key, wanted in attestation_exact.items() if attestation.get(key) != wanted]
    if mismatches:
        raise ManifestError(f"G3 signed attestation is incomplete or mismatched: {mismatches}")

    reviewer = load_reviewer_key(value["reviewer_key_id"])
    if reviewer["owner_identity"] == current_submitter_identity():
        raise ManifestError("G3 reviewer key owner must differ from the live submitter identity")
    verify_reviewer_signature(
        bound_paths["attestation_payload_path"],
        bound_paths["attestation_signature_path"],
        reviewer,
        "rebuttal-rlvr-g3",
    )

    for path_field, hash_field in (
        ("review_evidence_path", "review_evidence_sha256"),
        ("smoke_completion_receipt_path", "smoke_completion_receipt_sha256"),
    ):
        candidate = Path(str(attestation[path_field]))
        if not candidate.is_absolute() or not candidate.is_file() or sha256_file(candidate) != attestation[hash_field]:
            raise ManifestError(f"G3 signed evidence is missing or hash-mismatched: {path_field}")
        bound_paths[path_field] = candidate

    smoke = load_self_hashed_json(bound_paths["smoke_completion_receipt_path"], "G3 smoke completion receipt")
    required_smoke_fields = {
        "schema_version",
        "receipt_kind",
        "state",
        "job_id",
        "app_name",
        "hope_semantics_receipt_sha256",
        "run_hope_sha256",
        "jupyter_sha256",
        "image_digest",
        "path_override_receipt_sha256",
        "receipt_sha256",
    }
    if set(smoke) != required_smoke_fields:
        raise ManifestError("G3 smoke completion receipt has unexpected or missing fields")
    smoke_exact = {
        "schema_version": 1,
        "receipt_kind": "hope_smoke_terminal",
        "state": "SUCCEEDED",
        "job_id": attestation["smoke_job_id"],
        "app_name": attestation["smoke_app_name"],
        "hope_semantics_receipt_sha256": sha256_file(semantics_path),
    }
    if any(smoke.get(key) != wanted for key, wanted in smoke_exact.items()):
        raise ManifestError("G3 smoke completion receipt does not match the signed attestation")
    for field in ("run_hope_sha256", "jupyter_sha256", "path_override_receipt_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(smoke[field])):
            raise ManifestError(f"G3 smoke completion receipt has invalid {field}")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(smoke["image_digest"])):
        raise ManifestError("G3 smoke completion receipt has an invalid image_digest")

    value["_bound_files"] = {name: str(candidate) for name, candidate in bound_paths.items()}
    return value


def load_submission_admission(
    mode: str,
    semantics_path: Path,
    g3_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    semantics = load_semantics(semantics_path)
    semantics["_receipt_file_sha256"] = sha256_file(semantics_path)
    if semantics["returns_after"] == "completion":
        raise ManifestError("completion-blocking Hope semantics are not admitted")
    if mode in {"formal", "pilot"} and g3_path is None:
        raise ManifestError("formal/pilot submission requires G3 --g3-admission-receipt")
    g3_admission = load_g3_admission(g3_path, semantics_path) if g3_path is not None else None
    return semantics, g3_admission


def append_ledger(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_SH)
            lines = handle.read().splitlines()
            fcntl.flock(handle, fcntl.LOCK_UN)
        events = [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"submission ledger is unreadable: {path}") from exc
    if not all(isinstance(event, dict) for event in events):
        raise ManifestError(f"submission ledger contains a non-object event: {path}")
    return events


@contextmanager
def exclusive_ledger_session(path: Path):
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_ledger_admission(path: Path, renders: list[dict[str, Any]]) -> None:
    events = read_ledger(path)

    by_tag: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_tag.setdefault(str(event.get("job_tag", "")), []).append(event)

    for render in renders:
        prior = by_tag.get(render["job_tag"], [])
        prior_hashes = {event.get("cell_hash") for event in prior}
        if prior_hashes and prior_hashes != {render["cell_hash"]}:
            raise ManifestError(f"JOB_TAG collision with a different full CELL_HASH: {render['job_tag']}")
        latest = prior[-1] if prior else None
        if latest and latest.get("state") in ACTIVE_STATES | {"SUCCEEDED", "UNKNOWN"}:
            raise ManifestError(f"cell is already active, complete, or unknown: {render['job_tag']}")

        attempts = {event.get("attempt_id") for event in prior if event.get("attempt_id")}
        if render["attempt_id"] in attempts:
            raise ManifestError(f"ATTEMPT_ID already exists for cell: {render['attempt_id']}")
        if not prior and render["retry_of"] is not None:
            raise ManifestError("first attempt cannot declare retry_of")
        if prior:
            if len(attempts) >= 2:
                raise ManifestError(f"automatic retry budget exhausted: {render['job_tag']}")
            if latest.get("state") != "FAILED" or render["retry_of"] != latest.get("attempt_id"):
                raise ManifestError("retry must bind to the latest failed ATTEMPT_ID")


def receipt_payload(payload: dict[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value["receipt_sha256"] = sha256_bytes(canonical_json(value))
    return value


def write_receipt(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(json.dumps(receipt_payload(payload), indent=2, sort_keys=True) + "\n")
    return sha256_file(path)


def ledger_binding(item: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "app_name",
        "attempt_id",
        "cell_hash",
        "job_tag",
        "approved_manifest_sha256",
        "run_hope_sha256",
        "jupyter_sha256",
        "submitted_run_hope_sha256",
        "submitted_jupyter_sha256",
        "resolved_environment_sha256",
        "submitter_source_sha256",
        "hope_semantics_receipt_sha256",
        "g3_admission_receipt_sha256",
    )
    return {field: item[field] for field in fields if field in item}


def validate_submission_stage(render: dict[str, Any]) -> None:
    stage = Path(render["stage_dir"])
    archive = Path(render["archive_dir"])
    try:
        entries = list(stage.iterdir())
    except OSError as exc:
        raise ManifestError(f"cannot inspect submission stage: {stage}") from exc
    if {entry.name for entry in entries} != {"run.hope", "jupyter.sh"} or len(entries) != 2:
        raise ManifestError("submission stage must contain exactly run.hope and jupyter.sh")

    expected = {
        "run.hope": render["run_hope_sha256"],
        "jupyter.sh": render["jupyter_sha256"],
    }
    for name, wanted_hash in expected.items():
        source = stage / name
        if source.is_symlink() or not source.is_file():
            raise ManifestError(f"submission stage entry must be a regular non-symlink file: {name}")
        if sha256_file(source) != wanted_hash:
            raise ManifestError(f"submission stage byte hash changed after render: {name}")
        destination = archive / f"submitted.{name}"
        if destination.exists():
            raise ManifestError(f"submitted-byte archive already exists: {destination}")
        shutil.copyfile(source, destination)
        if sha256_file(destination) != wanted_hash:
            raise ManifestError(f"submitted-byte archive hash mismatch: {name}")

    render["submitted_run_hope_sha256"] = expected["run.hope"]
    render["submitted_jupyter_sha256"] = expected["jupyter.sh"]


def submit_one(render: dict[str, Any], semantics: dict[str, Any], timeout: int, ledger: Path) -> dict[str, Any]:
    stage = Path(render["stage_dir"])
    archive = Path(render["archive_dir"])
    validate_submission_stage(render)
    try:
        result = subprocess.run(
            semantics["run_command"],
            cwd=stage,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        def as_text(value: str | bytes | None) -> str:
            if value is None:
                return ""
            return value.decode(errors="replace") if isinstance(value, bytes) else value

        stdout = as_text(exc.stdout)
        stderr = as_text(exc.stderr)
        (archive / "submit.stdout").write_text(stdout)
        (archive / "submit.stderr").write_text(stderr)
        (archive / "submit.returncode").write_text("TIMEOUT\n")
        match = re.search(semantics["job_id_regex"], stdout + "\n" + stderr)
        unknown = {
            **ledger_binding(render),
            "hope_semantics_receipt_sha256": semantics["_receipt_file_sha256"],
            "job_id": match.group("job_id") if match is not None else None,
            "return_code": None,
            "state": "UNKNOWN",
            "stderr_sha256": sha256_file(archive / "submit.stderr"),
            "stdout_sha256": sha256_file(archive / "submit.stdout"),
            "time": int(time.time()),
        }
        receipt_sha256 = write_receipt(archive / "submission_unknown.json", unknown)
        unknown["submission_receipt_sha256"] = receipt_sha256
        append_ledger(ledger, unknown)
        raise ManifestError(f"Hope submission timed out with UNKNOWN state for {render['job_tag']}") from exc
    (archive / "submit.stdout").write_text(result.stdout)
    (archive / "submit.stderr").write_text(result.stderr)
    (archive / "submit.returncode").write_text(f"{result.returncode}\n")
    match = re.search(semantics["job_id_regex"], result.stdout + "\n" + result.stderr)
    job_id = match.group("job_id") if match is not None else None
    common = {
        **ledger_binding(render),
        "hope_semantics_receipt_sha256": semantics["_receipt_file_sha256"],
        "job_id": job_id,
        "return_code": result.returncode,
        "stderr_sha256": sha256_file(archive / "submit.stderr"),
        "stdout_sha256": sha256_file(archive / "submit.stdout"),
        "time": int(time.time()),
    }
    if result.returncode != 0 or match is None or "job_id" not in match.groupdict():
        failure = {**common, "state": "FAILED"}
        receipt_sha256 = write_receipt(archive / "submission_failure.json", failure)
        failure["submission_receipt_sha256"] = receipt_sha256
        append_ledger(ledger, failure)
        raise ManifestError(f"Hope submission failed or yielded no named job_id for {render['job_tag']}")
    event = {**common, "state": "QUEUED"}
    receipt_sha256 = write_receipt(archive / "submission_receipt.json", event)
    event["submission_receipt_sha256"] = receipt_sha256
    append_ledger(ledger, event)
    return {**render, **event}


def query_state(item: dict[str, Any], semantics: dict[str, Any], timeout: int, ledger: Path) -> str:
    command = [part.replace("{job_id}", item["job_id"]) for part in semantics["status_command"]]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise ManifestError(f"Hope status command failed for {item['job_id']}")
    if result.stderr.strip():
        raise ManifestError(f"Hope status command wrote unreviewed stderr for {item['job_id']}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ManifestError(f"Hope status output must contain exactly one recognized line for {item['job_id']}")
    match = re.fullmatch(semantics["status_regex"], lines[0])
    if match is None or "state" not in match.groupdict():
        raise ManifestError(f"Hope status output is unrecognized for {item['job_id']}")
    raw_state = match.group("state")
    state = semantics["state_map"].get(raw_state, "UNKNOWN")
    if state not in ACTIVE_STATES | TERMINAL_STATES:
        raise ManifestError(f"Hope state maps to UNKNOWN for {item['job_id']}: {raw_state}")
    if state != item["state"]:
        item["state"] = state
        append_ledger(
            ledger,
            {
                **ledger_binding(item),
                "job_id": item["job_id"],
                "state": state,
                "time": int(time.time()),
            },
        )
    return state


def list_platform_jobs(semantics: dict[str, Any], timeout: int) -> dict[str, dict[str, Any]]:
    result = subprocess.run(
        semantics["active_list_command"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise ManifestError("Hope global active-job list command failed")
    if result.stderr.strip():
        raise ManifestError("Hope global active-job list wrote unreviewed stderr")
    row_pattern = re.compile(semantics["active_list_regex"])
    empty_line = semantics["active_list_empty_line"]
    header_lines = set(semantics.get("active_list_header_lines", []))
    matches: list[re.Match[str]] = []
    saw_empty_marker = False
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        row_match = row_pattern.fullmatch(line)
        classifications = int(line == empty_line) + int(line in header_lines) + int(row_match is not None)
        if classifications > 1:
            raise ManifestError(f"Hope global active-job line matches overlapping semantics: {line!r}")
        if line == empty_line:
            saw_empty_marker = True
        elif line in header_lines:
            continue
        else:
            if row_match is not None:
                matches.append(row_match)
            else:
                raise ManifestError(f"Hope global active-job output has an unrecognized line: {line!r}")
    if not matches and not saw_empty_marker:
        raise ManifestError("Hope global active-job output is unrecognized")
    if matches and saw_empty_marker:
        raise ManifestError("Hope global active-job output mixes job rows with an empty marker")
    jobs: dict[str, dict[str, Any]] = {}
    for match in matches:
        raw_state = match.group("state")
        state = semantics["state_map"].get(raw_state, "UNKNOWN")
        if state not in ACTIVE_STATES | TERMINAL_STATES:
            raise ManifestError(f"Hope global state maps to UNKNOWN: {raw_state}")
        job_id = match.group("job_id")
        prior = jobs.get(job_id)
        if prior is not None:
            raise ManifestError(f"Hope global list reports duplicate job ID: {job_id}")
        jobs[job_id] = {
            "app_name": match.groupdict().get("app_name"),
            "job_id": job_id,
            "state": state,
        }
    return jobs


def latest_attempts(events: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        job_tag = event.get("job_tag")
        attempt_id = event.get("attempt_id")
        if job_tag and attempt_id:
            latest[(str(job_tag), str(attempt_id))] = dict(event)
    return latest


def reconcile_active_jobs(ledger: Path, semantics: dict[str, Any], timeout: int) -> dict[str, dict[str, Any]]:
    platform_jobs = list_platform_jobs(semantics, timeout)
    active = {job_id: item for job_id, item in platform_jobs.items() if item["state"] in ACTIVE_STATES}
    for item in latest_attempts(read_ledger(ledger)).values():
        if item.get("state") not in ACTIVE_STATES:
            continue
        job_id = item.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ManifestError("active ledger entry lacks a job_id")
        if job_id in platform_jobs:
            state = platform_jobs[job_id]["state"]
            if state != item["state"]:
                item["state"] = state
                append_ledger(
                    ledger,
                    {**ledger_binding(item), "job_id": job_id, "state": state, "time": int(time.time())},
                )
        else:
            state = query_state(item, semantics, timeout, ledger)
        if state in ACTIVE_STATES:
            active[job_id] = {**active.get(job_id, {}), **item, "state": state}
        else:
            active.pop(job_id, None)
    return active


def submit_batch(
    renders: list[dict[str, Any]],
    semantics: dict[str, Any],
    max_active: int,
    submit_timeout: int,
    status_timeout: int,
    poll_seconds: int,
    ledger: Path,
) -> None:
    if semantics["returns_after"] == "completion":
        raise ManifestError("completion-blocking Hope semantics are unsupported; G3 must provide safe child management")

    pending = list(renders)
    while pending:
        active = reconcile_active_jobs(ledger, semantics, status_timeout)
        capacity = max_active - len(active)
        if capacity > 0 and pending:
            wave = [pending.pop(0) for _ in range(min(capacity, len(pending)))]
            with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                futures = [pool.submit(submit_one, item, semantics, submit_timeout, ledger) for item in wave]
                for future in as_completed(futures):
                    future.result()
        elif pending:
            time.sleep(poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--render-only", action="store_true")
    action.add_argument("--submit", action="store_true")
    parser.add_argument("--render-output", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, default=Path("/tmp/rebuttal_rlvr_submit"))
    parser.add_argument("--hope-semantics-receipt", type=Path)
    parser.add_argument("--g3-admission-receipt", type=Path)
    parser.add_argument("--submission-ledger", type=Path)
    parser.add_argument("--max-active-jobs", type=int, default=8)
    parser.add_argument("--submit-timeout", type=int, default=172800)
    parser.add_argument("--status-timeout", type=int, default=60)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if not 1 <= args.max_active_jobs <= 10:
            raise ManifestError("max-active-jobs must be between 1 and 10")
        try:
            manifest_bytes = args.manifest.read_bytes()
        except OSError as exc:
            raise ManifestError(f"cannot read manifest: {args.manifest}") from exc
        raw = load_manifest_bytes(manifest_bytes, args.manifest)
        validate_matrix(raw)
        validate_live_bindings(raw)
        semantics: dict[str, Any] | None = None
        g3_admission: dict[str, Any] | None = None
        if args.submit:
            if args.hope_semantics_receipt is None:
                raise ManifestError("real submission requires --hope-semantics-receipt")
            if args.submission_ledger is None or not args.submission_ledger.is_absolute():
                raise ManifestError("real submission requires an absolute --submission-ledger")
            semantics, g3_admission = load_submission_admission(
                raw["mode"],
                args.hope_semantics_receipt,
                args.g3_admission_receipt,
            )
            if Path(semantics["submission_ledger_path"]).resolve() != args.submission_ledger.resolve():
                raise ManifestError("--submission-ledger differs from the G3-reviewed global ledger path")
        args.scratch_root.mkdir(parents=True, exist_ok=True)
        args.render_output.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="render-", dir=args.scratch_root) as temporary:
            temporary_root = Path(temporary)
            renders = [
                archive_render(job, temporary_root, manifest_bytes, archive_bound_files=True)
                for job in raw["jobs"]
            ]
            for item in renders:
                source = Path(item["attempt_dir"])
                destination = args.render_output / source.relative_to(temporary_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, destination)
                item["attempt_dir"] = str(destination)
                item["stage_dir"] = str(destination / "stage")
                item["archive_dir"] = str(destination / "archive")
                if semantics is not None and args.hope_semantics_receipt is not None:
                    semantics_copy = destination / "archive/hope_semantics_receipt.json"
                    shutil.copyfile(args.hope_semantics_receipt, semantics_copy)
                    item["hope_semantics_receipt_sha256"] = sha256_file(semantics_copy)
                if g3_admission is not None and args.g3_admission_receipt is not None:
                    g3_copy = destination / "archive/g3_admission_receipt.json"
                    shutil.copyfile(args.g3_admission_receipt, g3_copy)
                    item["g3_admission_receipt_sha256"] = sha256_file(g3_copy)
                    evidence_dir = destination / "archive/g3_bound_evidence"
                    evidence_dir.mkdir()
                    for label, source_text in g3_admission["_bound_files"].items():
                        source = Path(source_text)
                        evidence_copy = evidence_dir / f"{label}.bin"
                        shutil.copyfile(source, evidence_copy)
                        if sha256_file(evidence_copy) != sha256_file(source):
                            raise ManifestError(f"G3 evidence changed during archive: {label}")

        if args.submit:
            assert semantics is not None and args.submission_ledger is not None
            ledger = args.submission_ledger
            with exclusive_ledger_session(ledger):
                reconcile_active_jobs(ledger, semantics, args.status_timeout)
                validate_ledger_admission(ledger, renders)
                submit_batch(
                    renders,
                    semantics,
                    args.max_active_jobs,
                    args.submit_timeout,
                    args.status_timeout,
                    args.poll_seconds,
                    ledger,
                )
        print(json.dumps({"ok": True, "jobs": len(renders), "mode": raw["mode"], "submitted": args.submit}, sort_keys=True))
        return 0
    except (ManifestError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
