#!/usr/bin/env python3
"""Resolve a Docker image's portable OCI config digest."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence


Run = Callable[..., subprocess.CompletedProcess[str]]
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


def _stdout(run: Run, command: Sequence[str]) -> str:
    result = run(command, check=True, capture_output=True, text=True)
    return result.stdout


def resolve_image_config_digest(image_ref: str, *, run: Run = subprocess.run) -> str:
    images = json.loads(_stdout(run, ["docker", "image", "inspect", image_ref]))
    if not isinstance(images, list) or len(images) != 1:
        raise RuntimeError(f"expected one Docker image for {image_ref!r}")
    image = images[0]
    descriptor = image.get("Descriptor")
    if descriptor:
        manifest_digest = descriptor.get("digest")
        if not isinstance(manifest_digest, str) or not SHA256.fullmatch(manifest_digest):
            raise RuntimeError("Docker image descriptor has no valid digest")
        manifest = json.loads(
            _stdout(run, ["ctr", "-n", "moby", "content", "get", manifest_digest])
        )
        config_digest = manifest.get("config", {}).get("digest")
    else:
        # The legacy overlay2 image store exposes the config digest as .Id.
        config_digest = image.get("Id")
    if not isinstance(config_digest, str) or not SHA256.fullmatch(config_digest):
        raise RuntimeError("Docker image has no valid OCI config digest")
    return config_digest


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("usage: resolve_image_config_digest.py IMAGE_REF")
    print(resolve_image_config_digest(args[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
