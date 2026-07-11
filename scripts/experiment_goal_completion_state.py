#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def state(calibration:dict|None, reviewer:dict|None)->str:
    if not calibration or calibration.get('decision')!='deployable': return 'PENDING OPERATIONAL CALIBRATION'
    if not reviewer or reviewer.get('verdict')!='ACCEPTED' or not reviewer.get('all_acceptance_criteria_pass'): return 'PENDING INDEPENDENT ACCEPTANCE'
    return 'GOAL COMPLETE'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--calibration',type=Path); p.add_argument('--reviewer',type=Path); a=p.parse_args()
    load=lambda x: json.loads(x.read_text()) if x and x.is_file() else None
    result=state(load(a.calibration),load(a.reviewer)); print(result); return 0 if result=='GOAL COMPLETE' else 1
if __name__=='__main__': raise SystemExit(main())
