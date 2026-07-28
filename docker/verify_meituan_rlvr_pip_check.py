#!/usr/bin/env python3
"""Fail closed on dependency conflicts outside the reviewed vLLM override."""

from __future__ import annotations

import argparse
from collections import Counter
import subprocess
import sys


EXPECTED_VLLM_OVERRIDES = (
    "vllm 0.12.0 has requirement torch==2.9.0, but you have torch 2.9.1+cu126.",
    "vllm 0.12.0 has requirement torchaudio==2.9.0, but you have torchaudio 2.9.1+cu126.",
    "vllm 0.12.0 has requirement torchvision==0.24.0, but you have torchvision 0.24.1+cu126.",
)


def verify_pip_check(output: str, returncode: int) -> None:
    """Require exactly the three reviewed vLLM metadata mismatches."""
    observed = Counter(line.strip() for line in output.splitlines() if line.strip())
    expected = Counter(EXPECTED_VLLM_OVERRIDES)
    if returncode == 0:
        raise RuntimeError(
            "pip check unexpectedly passed; the reviewed torch 2.9.1/vLLM 0.12.0 "
            "override is absent, so this image is not the declared compatibility matrix"
        )
    if observed != expected:
        missing = sorted((expected - observed).elements())
        unexpected = sorted((observed - expected).elements())
        raise RuntimeError(
            "pip check differs from the exact reviewed allowlist; "
            f"missing={missing!r}; unexpected={unexpected!r}"
        )


def self_test() -> None:
    allowed = "\n".join(EXPECTED_VLLM_OVERRIDES) + "\n"
    verify_pip_check(allowed, 1)

    rejected = (
        allowed
        + "example-package 1.0 has requirement dependency==1.0, but you have dependency 2.0.\n"
    )
    try:
        verify_pip_check(rejected, 1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("unexpected dependency conflicts must fail the verifier")

    try:
        verify_pip_check("\n".join(EXPECTED_VLLM_OVERRIDES[:-1]), 1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a missing declared mismatch must fail the verifier")

    try:
        verify_pip_check("No broken requirements found.\n", 0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("an undeclared compatibility-matrix change must fail the verifier")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="exercise both the accepted fixture and intentional rejection cases",
    )
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("pip-check verifier self-test: PASS")
        return 0

    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(completed.stdout, end="")
    try:
        verify_pip_check(completed.stdout, completed.returncode)
    except RuntimeError as exc:
        print(f"pip-check verifier: FAIL: {exc}", file=sys.stderr)
        return 2
    print("pip-check verifier: PASS (exact reviewed vLLM metadata override only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
