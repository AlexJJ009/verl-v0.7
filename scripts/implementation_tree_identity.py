#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

EXPECTED={"schema_version":1,"algorithm":"stage123-implementation-boundary-v1","superproject_tree_roots":["config/experiment_execution","scripts","verl"],"submodules":[{"path":"recipe","coverage":"entire_gitlink_commit"}]}

def git(root:Path,*args:str)->str:
    return subprocess.check_output(["git","-C",str(root),*args],text=True).strip()

def canonical(records:list[dict])->bytes:
    return b"".join((json.dumps(item,sort_keys=True,separators=(",",":"))+"\n").encode() for item in sorted(records,key=lambda x:x["path"].encode()))

def compute(root:Path,boundary:Path)->tuple[bytes,str]:
    value=json.loads(boundary.read_text())
    if value!=EXPECTED: raise ValueError("boundary_manifest_mismatch")
    dirty=git(root,"status","--porcelain","--untracked-files=all","--",*value["superproject_tree_roots"])
    if dirty: raise ValueError("covered_superproject_tree_dirty")
    stage=git(root,"ls-files","--stage","recipe").split()
    if len(stage)<4 or stage[0]!="160000": raise ValueError("recipe_gitlink_missing")
    gitlink=stage[1]
    if git(root/"recipe","rev-parse","HEAD")!=gitlink: raise ValueError("recipe_head_gitlink_mismatch")
    if git(root/"recipe","status","--porcelain","--untracked-files=all"): raise ValueError("recipe_checkout_dirty")
    records=[{"kind":"git_tree","path":path,"tree_sha1":git(root,"rev-parse",f"HEAD:{path}")} for path in value["superproject_tree_roots"]]
    records.append({"gitlink_commit":gitlink,"kind":"gitlink","mode":"160000","path":"recipe"})
    payload=canonical(records)
    return payload,hashlib.sha256(payload).hexdigest()

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--repo-root",type=Path,required=True); p.add_argument("--boundary-manifest",type=Path,default=Path("config/experiment_execution/stage123_implementation_boundary_v1.json")); p.add_argument("--format",choices=("json",),default="json"); p.add_argument("--output",type=Path); p.add_argument("--compare",type=Path); a=p.parse_args()
    try: payload,digest=compute(a.repo_root,(a.repo_root/a.boundary_manifest) if not a.boundary_manifest.is_absolute() else a.boundary_manifest)
    except (OSError,ValueError,subprocess.CalledProcessError) as exc:
        print(json.dumps({"ok":False,"failure":{"code":"implementation_tree_identity","message":str(exc),"context":{}}},sort_keys=True)); return 1
    if a.compare and a.compare.read_bytes()!=payload: print(json.dumps({"ok":False,"failure":{"code":"implementation_tree_compare","message":"canonical identity mismatch","context":{}}},sort_keys=True)); return 1
    if a.output: a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_bytes(payload)
    else: sys.stdout.buffer.write(payload)
    print(json.dumps({"implementation_tree_sha256":digest},sort_keys=True),file=sys.stderr)
    return 0
if __name__=="__main__": raise SystemExit(main())
