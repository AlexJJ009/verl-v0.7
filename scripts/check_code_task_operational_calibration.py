#!/usr/bin/env python3
"""Validate real local L40S calibration and return deployability."""

from __future__ import annotations
import argparse,json
from pathlib import Path
import sys

METRICS=('validation_elapsed_seconds','peak_rss_gib','gpu_wait_fraction')

def check(report:dict, manifest:dict)->dict:
    failures=[]
    if report.get('evidence_class')!='infrastructure_calibration': failures.append('wrong evidence class')
    if report.get('manifest_sha256')!=manifest.get('manifest_sha256'): failures.append('manifest hash mismatch')
    phases=report.get('phases',[])
    if [p.get('phase') for p in phases]!=['stage1','stage2','stage3']: failures.append('phase order mismatch')
    hashes={p.get('profile_hash') for p in phases}
    if hashes!={manifest.get('resource_profile',{}).get('sha256')}: failures.append('profile hash mismatch')
    for phase in phases:
        name=phase.get('phase','unknown'); observed=phase.get('observed',{}); predicted=phase.get('predicted',{})
        if observed.get('complete_validation_metrics') is not True: failures.append(f'{name}: incomplete validation metrics')
        if observed.get('validation_elapsed_seconds',10**9)>1800: failures.append(f'{name}: validation deadline exceeded')
        for metric in METRICS:
            actual=observed.get(metric); estimate=predicted.get(metric)
            if actual is None or estimate in (None,0): failures.append(f'{name}: missing {metric} prediction evidence'); continue
            if abs(actual-estimate)/abs(estimate)>0.20: failures.append(f'{name}: {metric} prediction error exceeds 20%')
        if phase.get('optimized') is not True: failures.append(f'{name}: optimization/safety budget not met')
    decision='deployable' if not failures and report.get('decision')=='deployable' else report.get('decision','inconclusive')
    if decision=='deployable' and failures: decision='blocked'
    return {'ok':decision=='deployable' and not failures,'decision':decision,'failures':failures}

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--report',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); a=p.parse_args()
    report=json.loads(a.report.read_text()); manifest=json.loads(a.manifest.read_text()) if a.manifest.suffix=='.json' else None
    if manifest is None:
        import subprocess
        raw=subprocess.check_output(['python3',str(Path(__file__).with_name('experiment_manifest.py')),'render',str(a.manifest),'--format','json'],text=True); manifest=json.loads(raw)
    result=check(report,manifest); print(json.dumps(result,indent=2,sort_keys=True)); return 0 if result['ok'] else 1
if __name__=='__main__': sys.exit(main())
