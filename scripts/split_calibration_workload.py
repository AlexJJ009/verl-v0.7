#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_COUNTS = {"HumanEval+": 16, "MBPP+": 16, "LiveCodeBench": 32}
EXPECTED_SHA256 = "c3eaf3374661fba71d1132f0de7a8dbdbd3d90295d4fabeb77b5e9dd7c221608"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_workload(source: Path, output_root: Path) -> dict[str, str]:
    import pandas as pd

    frame = pd.read_parquet(source)
    actual = frame["data_source"].value_counts().to_dict()
    if actual != EXPECTED_COUNTS or sha256(source) != EXPECTED_SHA256:
        raise RuntimeError(f"calibration workload mismatch: {actual}")
    outputs = {}
    for name in EXPECTED_COUNTS:
        path = output_root / f"{name.lower().replace('+', '_plus')}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame[frame["data_source"] == name].to_parquet(path, index=False)
        outputs[name] = str(path)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    outputs = split_workload(args.source, args.output_root)
    receipt = {
        "schema_version": 1,
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "outputs": {name: {"path": path, "sha256": sha256(Path(path))} for name, path in outputs.items()},
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "receipt": str(args.receipt)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
