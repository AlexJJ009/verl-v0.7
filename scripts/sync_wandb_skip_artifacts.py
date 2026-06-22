#!/usr/bin/env python3
"""Sync a W&B offline run while skipping artifact records.

This is a narrow recovery helper for offline runs whose `.wandb` event stream
contains artifact records pointing at missing local staging files.
"""

from __future__ import annotations

import argparse
import os
import sys

import wandb
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore, sender


def _find_wandb_file(path: str) -> str:
    if path.endswith(".wandb"):
        return path
    names = [name for name in os.listdir(path) if name.endswith(".wandb")]
    if len(names) != 1:
        raise RuntimeError(f"Expected exactly one .wandb file in {path}, found {names}")
    return os.path.join(path, names[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mark-synced", action="store_true")
    args = parser.parse_args()

    sync_file = _find_wandb_file(args.path)
    root_dir = os.path.dirname(sync_file)

    sm = sender.SendManager.setup(root_dir, resume=None)
    ds = datastore.DataStore()
    ds.open_for_scan(sync_file)

    exit_pb = None
    finished = False
    sent = 0
    skipped_artifacts = 0
    shown = False

    while True:
        data = ds.scan_data()
        if data is None:
            break

        pb = wandb_internal_pb2.Record()
        pb.ParseFromString(data)
        record_type = pb.WhichOneof("record_type")

        if record_type == "run":
            pb.run.run_id = args.run_id
            pb.run.entity = args.entity
            pb.run.project = args.project
            pb.control.req_resp = True
        elif record_type == "artifact":
            skipped_artifacts += 1
            continue
        elif record_type == "exit":
            exit_pb = pb
            finished = True
            continue
        elif record_type == "final":
            if exit_pb is None:
                raise RuntimeError("final record seen without an exit record")
            pb = exit_pb
            exit_pb = None

        sm.send(pb)
        sent += 1

        while not sm._record_q.empty():
            sm.send(sm._record_q.get(block=True))

        if pb.control.req_resp:
            result = sm._result_q.get(block=True)
            if not shown and result.WhichOneof("result_type") == "run_result":
                r = result.run_result.run
                print(f"Syncing without artifacts: {r.entity}/{r.project}/{r.run_id}")
                shown = True

    sm.finish()

    if args.mark_synced and finished:
        with open(f"{sync_file}.synced", "w"):
            pass

    print(
        f"done sent_records={sent} skipped_artifact_records={skipped_artifacts} "
        f"finished={finished}"
    )
    return 0 if finished else 2


if __name__ == "__main__":
    sys.exit(main())
