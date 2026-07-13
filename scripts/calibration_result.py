#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any
REQUIRED=("decision","manifest_sha256","resource_profile_sha256","implementation_tree_sha256","evidence_commit","workload_identity","policy_id","policy_sha256","authorization_identity","started_at","completed_at","phase_evidence","prediction_comparison","cleanup","failures")
def validate(value:dict[str,Any],schema:dict[str,Any])->dict[str,Any]:
    failures=[]
    if value.get("schema_version")!=1 or value.get("result_type")!="calibration_result": failures.append({"code":"result_schema","message":"unsupported calibration result","context":{}})
    missing=[key for key in REQUIRED if key not in value]
    if missing: failures.append({"code":"result_fields","message":"calibration result is incomplete","context":{"missing":missing}})
    if value.get("decision") not in schema["decisions"]: failures.append({"code":"result_decision","message":"invalid calibration decision","context":{"decision":value.get("decision")}})
    if value.get("decision")=="passed" and value.get("failures")!=[]: failures.append({"code":"passed_with_failures","message":"passed result cannot contain failures","context":{}})
    return {"ok":not failures,"decision":value.get("decision") if not failures else "blocked","failures":failures}
def policy_sha256(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
