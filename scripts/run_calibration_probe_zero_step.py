#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--phases',required=True); p.add_argument('--repetitions',type=int,required=True); p.add_argument('--training-steps',type=int,required=True); p.add_argument('--optimizer-enabled',required=True); p.add_argument('--scratch-root',type=Path,required=True); p.add_argument('--manifest-sha256',required=True); p.add_argument('--resource-profile-sha256',required=True); a=p.parse_args()
    phases=a.phases.split(',')
    if phases!=['stage2','stage3']: raise SystemExit('phase_set')
    if a.training_steps!=0 or a.optimizer_enabled.lower()!='false': raise SystemExit('training_disabled')
    if not 1<=a.repetitions<=3: raise SystemExit('repetitions')
    if not str(a.scratch_root).startswith('/data-1/tmp/verl_agent_scratch/'): raise SystemExit('scratch_root')
    a.scratch_root.mkdir(parents=True,exist_ok=True)
    value={'schema_version':1,'driver':'stage123-zero-step-calibration-v1','manifest':str(a.manifest),'manifest_sha256':a.manifest_sha256,'resource_profile_sha256':a.resource_profile_sha256,'phases':phases,'repetitions':a.repetitions,'training_steps':0,'optimizer_enabled':False,'phase_specs':[{'phase':phase,'repetitions':a.repetitions,'training_steps':0,'optimizer_enabled':False} for phase in phases]}
    (a.scratch_root/'probe-spec.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'ok':True,'probe_spec':str(a.scratch_root/'probe-spec.json')},sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
