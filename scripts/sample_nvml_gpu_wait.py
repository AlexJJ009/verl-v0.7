#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Sample all-GPU idle wall time with NVML without spawning nvidia-smi."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from pathlib import Path


class Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--idle-threshold", type=int, default=2)
    args = parser.parse_args()

    lib = ctypes.CDLL("libnvidia-ml.so.1")
    if lib.nvmlInit_v2() != 0:
        raise RuntimeError("nvmlInit_v2 failed")
    try:
        count = ctypes.c_uint()
        if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0 or count.value != 8:
            raise RuntimeError(f"expected 8 GPUs, got {count.value}")
        handles = []
        for index in range(count.value):
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != 0:
                raise RuntimeError(f"failed to get GPU {index}")
            handles.append(handle)

        readiness_wait_start = time.monotonic()
        while not args.ready_file.is_file():
            try:
                os.kill(args.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(args.interval)
        readiness_wait = time.monotonic() - readiness_wait_start
        measurement_started = args.ready_file.is_file()
        samples = idle = 0
        if measurement_started:
            while True:
                try:
                    os.kill(args.pid, 0)
                except ProcessLookupError:
                    break
                values = []
                for handle in handles:
                    util = Utilization()
                    if lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(util)) != 0:
                        raise RuntimeError("nvmlDeviceGetUtilizationRates failed")
                    values.append(util.gpu)
                samples += 1
                idle += int(all(value <= args.idle_threshold for value in values))
                time.sleep(args.interval)
        rendered = (
            json.dumps(
                {
                    "gpu_sample_count": samples,
                    "gpu_idle_sample_count": idle,
                    "gpu_wait_fraction": idle / samples if samples else None,
                    "gpu_idle_threshold_pct": args.idle_threshold,
                    "gpu_sample_interval_seconds": args.interval,
                    "measurement_started": measurement_started,
                    "readiness_wait_seconds": readiness_wait,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        tmp = args.output.with_name(f"{args.output.name}.tmp.{os.getpid()}")
        tmp.write_text(rendered)
        os.replace(tmp, args.output)
    finally:
        lib.nvmlShutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
