#!/usr/bin/env python3
"""Fail-closed Firejail and hidden-test confidentiality probes for KodCode."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import tempfile
from typing import Iterator


def wrap_code(code: str) -> str:
    return f"<think>verify the implementation against the hidden tests</think>\n<answer>\n```python\n{code}\n```\n</answer>"


@contextmanager
def temporary_env(updates: dict[str, str | None]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in updates}
    try:
        for name, value in updates.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--containment-only",
        action="store_true",
        help="validate Firejail home isolation but skip the formal hidden-test confidentiality gate",
    )
    args = parser.parse_args()

    if not os.path.isfile("/usr/bin/firejail"):
        raise RuntimeError("/usr/bin/firejail is missing")

    from recipe.on_policy_wdl_sft.code_task.official_aligned_reward import (
        compute_score_code_official_aligned,
    )

    ground_truth = {
        "verification_method": "kodcode_exec",
        "test": (
            "from solution import *\n\n"
            "def test_add():\n"
            "    assert add(2, 3) == 5\n"
        ),
    }
    formal_env = {"KODCODE_EXEC": "/usr/bin/firejail", "KODCODE_ALLOW_UNSANDBOXED": None}
    with temporary_env(formal_env):
        passing = compute_score_code_official_aligned(
            "kodcode_light_rl_10k",
            wrap_code("def add(a, b):\n    return a + b"),
            ground_truth,
        )

    if passing.get("code_reward_status") != "pass" or passing.get("code_reward_sandbox") != "firejail":
        raise RuntimeError(f"valid KodCode sample did not pass in Firejail: {passing}")

    home = Path.home().resolve()
    home.mkdir(parents=True, exist_ok=True)
    canary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".meituan-firejail-canary-",
            dir=home,
            delete=False,
        ) as canary:
            canary.write("host-only-canary\n")
            canary_path = Path(canary.name).resolve()

        canary_code = (
            "def host_canary_visible():\n"
            "    from pathlib import Path\n"
            f"    return Path({str(canary_path)!r}).read_text(encoding='utf-8') == 'host-only-canary\\n'"
        )
        canary_truth = {
            "verification_method": "kodcode_exec",
            "test": (
                "from solution import *\n\n"
                "def test_host_canary_visibility():\n"
                "    assert host_canary_visible() is True\n"
            ),
        }
        with temporary_env(
            {
                "KODCODE_EXEC": "/definitely/missing/meituan-firejail",
                "KODCODE_ALLOW_UNSANDBOXED": "1",
            }
        ):
            unsandboxed_canary = compute_score_code_official_aligned(
                "kodcode_light_rl_10k",
                wrap_code(canary_code),
                canary_truth,
            )
        with temporary_env(formal_env):
            sandboxed_canary = compute_score_code_official_aligned(
                "kodcode_light_rl_10k",
                wrap_code(canary_code),
                canary_truth,
            )
    finally:
        if canary_path is not None:
            canary_path.unlink(missing_ok=True)

    unsandboxed_control_ok = (
        unsandboxed_canary.get("code_reward_status") == "pass"
        and unsandboxed_canary.get("code_reward_sandbox") == "diagnostic_unsandboxed_hidden_tests"
    )
    containment_ok = (
        sandboxed_canary.get("code_reward_status") == "wrong_answer"
        and sandboxed_canary.get("score") == -1.0
        and sandboxed_canary.get("code_reward_sandbox") == "firejail"
    )

    containment_receipt = {
        "passing_status": passing.get("code_reward_status"),
        "sandbox": passing.get("code_reward_sandbox"),
        "unsandboxed_canary_status": unsandboxed_canary.get("code_reward_status"),
        "firejail_canary_status": sandboxed_canary.get("code_reward_status"),
        "unsandboxed_control_ok": unsandboxed_control_ok,
        "containment_ok": containment_ok,
    }
    if args.containment_only:
        if not unsandboxed_control_ok:
            raise RuntimeError(f"unsandboxed control could not read the host canary: {unsandboxed_canary}")
        if not containment_ok:
            raise RuntimeError(f"Firejail did not hide the host-home canary: {sandboxed_canary}")
        print(
            json.dumps(
                {
                    "ok": True,
                    "probe": "kodcode_firejail_basic_containment",
                    **containment_receipt,
                },
                sort_keys=True,
            )
        )
        return 0

    secret = secrets.token_hex(16)
    source_reader = (
        "def reveal_hidden_secret():\n"
        "    import base64, pathlib, re, sys\n"
        "    runner = pathlib.Path(sys.argv[0]).read_text(encoding='utf-8')\n"
        "    for encoded in re.findall(r\"base64\\.b64decode\\('([^']+)'\\)\", runner):\n"
        "        decoded = base64.b64decode(encoded).decode('utf-8')\n"
        "        match = re.search(r'SECRET:([0-9a-f]+)', decoded)\n"
        "        if match:\n"
        "            return match.group(1)\n"
        "    return 'not-found'\n"
    )
    secret_truth = {
        "verification_method": "kodcode_exec",
        "test": (
            "from solution import *\n\n"
            f"# SECRET:{secret}\n"
            "def test_hidden_secret():\n"
            f"    assert reveal_hidden_secret() == {secret!r}\n"
        ),
    }
    with temporary_env(formal_env):
        source_leak = compute_score_code_official_aligned(
            "kodcode_light_rl_10k",
            wrap_code(source_reader),
            secret_truth,
        )
    confidentiality_ok = (
        source_leak.get("code_reward_status") == "wrong_answer"
        and source_leak.get("score") == -1.0
    )
    failures = []
    if not unsandboxed_control_ok:
        failures.append("unsandboxed control could not read the host canary")
    if not containment_ok:
        failures.append("Firejail did not hide the host-home canary")
    if not confidentiality_ok:
        failures.append("runner exposes encoded test source")
    if failures:
        print(
            json.dumps(
                {
                    "ok": False,
                    "probe": "kodcode_hidden_test_confidentiality",
                    "hidden_source_attack_status": source_leak.get("code_reward_status"),
                    "confidentiality_ok": confidentiality_ok,
                    "failures": failures,
                    **containment_receipt,
                },
                sort_keys=True,
            )
        )
        raise RuntimeError(
            "; ".join(failures)
            + "; formal hidden-test reward is blocked until both platform containment "
            "and test materialization pass this probe"
        )

    print(
        json.dumps(
            {
                "ok": True,
                "probe": "kodcode_firejail_and_hidden_test_confidentiality",
                "hidden_source_attack_status": source_leak.get("code_reward_status"),
                **containment_receipt,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
