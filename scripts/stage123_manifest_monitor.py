#!/usr/bin/env python3
"""Manifest-native Stage123 monitor with reviewed lifecycle notifications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


def tmux_active(name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", name], capture_output=True).returncode == 0


def latest_checkpoint(root: Path, prefix: str) -> tuple[Path | None, int]:
    matches = sorted(root.glob(f"{prefix}_*"))
    if not matches: return None, 0
    ckpt = matches[-1]; marker = ckpt / "latest_checkpointed_iteration.txt"
    digits = "".join(ch for ch in marker.read_text() if ch.isdigit()) if marker.is_file() else ""
    return ckpt, int(digits or 0)


def emit(policy: Path, ledger: Path, sender: list[str] | None, state: dict, scratch: Path) -> None:
    scratch.write_text(json.dumps(state))
    cmd = ["python3", str(policy), "--state", str(scratch), "--ledger", str(ledger)]
    if sender: cmd += ["--sender", *sender]
    subprocess.run(cmd, check=False)


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--checkpoint-root',type=Path,required=True); p.add_argument('--queue-tmux',required=True); p.add_argument('--poll-seconds',type=float,default=60); p.add_argument('--ledger',type=Path,required=True); p.add_argument('--policy',type=Path,required=True); p.add_argument('--sender',nargs='+'); p.add_argument('--once',action='store_true'); args=p.parse_args()
    manifest=json.loads(args.manifest.read_text()); scratch=args.ledger.with_suffix('.state.json'); seen_active=set()
    while True:
        any_active=tmux_active(args.queue_tmux)
        for run in manifest['runs']:
            prefix=run['run_prefix']; active=tmux_active(run['tmux_name']); any_active |= active
            ckpt,step=latest_checkpoint(args.checkpoint_root,prefix)
            metrics=[] if ckpt is None else list(Path('/data-1/code/verl/recipe/on_policy_wdl_sft').glob(f'**/metrics/OnPolicyWDLSFT-CodeTask/{ckpt.name}.jsonl'))
            deadline=Path('/data-2/experiment_registry/validation_deadlines')/f'{prefix}.deadline.json'
            if active: seen_active.add(prefix)
            state={'run_id':prefix,'training_step':step,'complete_validation_metrics':bool(metrics),'local_paths':f'checkpoint={ckpt}; deadline={deadline}'}
            if prefix in seen_active and not active and step < int(run['final_step']):
                state.update({'terminal_failure':True,'cleanup_evidence':deadline.is_file(),'background':'Stage123 run stopped before final step','evidence':deadline.read_text() if deadline.is_file() else f'step={step}','cost':'GPU queue stopped','recommendation':'Inspect local deadline and training logs'})
            emit(args.policy,args.ledger,args.sender,state,scratch)
        if args.once or not any_active: return 0
        time.sleep(args.poll_seconds)

if __name__=='__main__': raise SystemExit(main())
