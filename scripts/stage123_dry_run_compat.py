#!/usr/bin/env python3
"""Render legacy Stage123 dry-run evidence without launching lifecycle work."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-tool", type=Path, required=True)
    parser.add_argument("--python", dest="python_executable", default="python3")
    parser.add_argument("--scratch-root", type=Path, required=True)
    args = parser.parse_args()
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    normalized_path = args.scratch_root / "stage123.normalized.json"
    rendered = subprocess.check_output(
        [args.python_executable, str(args.manifest_tool), "render", str(args.manifest), "--format", "json"],
        text=True,
    )
    normalized_path.write_text(rendered)
    normalized = json.loads(rendered)
    stage2_runs = [run for run in normalized["runs"] if run.get("phase") == "stage2"]
    stage3_runs = [run for run in normalized["runs"] if run.get("phase") == "stage3"]
    for run in stage2_runs:
        payload = {
            "schema_version": 1,
            "run_id": run["id"],
            "manifest_sha256": normalized["manifest_sha256"],
            "release_eligible": False,
            "source": "stage123_dry_run_compat",
        }
        output = args.scratch_root / f"{run['id']}.provenance.json"
        output.write_text(json.dumps(payload, sort_keys=True) + "\n")
    status = args.scratch_root / "status.tsv"
    rows = ["timestamp\tchain\tphase\tstatus\tdetail"]
    for run in stage3_runs:
        rows.append(f"dry-run\t{run['id']}\tstage3\tpending_producer\tcompatibility projection")
    status.write_text("\n".join(rows) + "\n")
    print(
        f"[STAGE123 QUEUE] DRY_RUN PASS; Stage3 blocked: pending current manifest_hash={normalized['manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
