#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path

def fail(code,message,**context):
    print(json.dumps({"ok":False,"failure":{"code":code,"message":message,"context":context}},sort_keys=True),file=sys.stderr); return 1

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--resource-profile",type=Path,required=True); p.add_argument("--phases",required=True); p.add_argument("--repetitions",type=int,required=True); p.add_argument("--training-steps",type=int,required=True); p.add_argument("--scratch-root",type=Path,required=True); a=p.parse_args()
    phases=a.phases.split(",")
    if phases!=["stage2","stage3"]: return fail("phase_set","probe phases must be stage2,stage3",phases=phases)
    if not 1<=a.repetitions<=3: return fail("repetitions","probe repetitions must be 1..3",repetitions=a.repetitions)
    if a.training_steps!=0: return fail("training_steps","calibration cannot train",training_steps=a.training_steps)
    if not str(a.scratch_root).startswith("/data-1/tmp/verl_agent_scratch/"): return fail("scratch_root","outputs must stay in calibration scratch",path=str(a.scratch_root))
    root=Path(__file__).resolve().parents[1]
    rendered=json.loads(subprocess.check_output([sys.executable,str(root/"scripts/experiment_manifest.py"),"render",str(a.manifest),"--format","json"],text=True))
    profile_hash=subprocess.check_output(["bash","-lc",f"source {a.resource_profile!s}; stage123_profile_hash"],text=True).strip()
    if rendered["resource_profile"]["sha256"]!=profile_hash: return fail("profile_hash","manifest/profile mismatch")
    command=["bash",str(root/"recipe/on_policy_wdl_sft/code_task/run_code_task_operational_calibration_phase.sh"),"--manifest",str(a.manifest),"--phases",a.phases,"--repetitions",str(a.repetitions),"--training-steps","0","--optimizer-enabled","false","--scratch-root",str(a.scratch_root),"--manifest-sha256",rendered["manifest_sha256"],"--resource-profile-sha256",profile_hash]
    print(json.dumps(command,separators=(",",":"))); return 0
if __name__=="__main__": raise SystemExit(main())
