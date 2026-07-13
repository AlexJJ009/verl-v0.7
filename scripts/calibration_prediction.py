#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass,asdict
from typing import Any
@dataclass(frozen=True)
class PredictionDecision:
    qualified:bool; code:str; message:str; context:dict[str,Any]
    def as_dict(self): return asdict(self)
def qualify(history:list[float],predicted:float,observed:float,policy:dict[str,Any])->PredictionDecision:
    cfg=policy.get("prediction",{}); minimum=cfg.get("minimum_history_count"); ratio=cfg.get("maximum_observed_to_predicted_ratio")
    if policy.get("policy_id")!="stage123-calibration-policy-v1": return PredictionDecision(False,"policy_version","unsupported prediction policy",{})
    if not isinstance(minimum,int) or len(history)<minimum: return PredictionDecision(False,"insufficient_history","prediction history is insufficient",{"count":len(history),"required":minimum})
    if predicted<=0 or observed<0: return PredictionDecision(False,"prediction_values","prediction values are invalid",{})
    actual=observed/predicted
    if actual>ratio: return PredictionDecision(False,"prediction_exceeded","observed value exceeds qualified prediction",{"ratio":actual,"limit":ratio})
    return PredictionDecision(True,"qualified","prediction is supported by current probe evidence",{"ratio":actual,"history_count":len(history)})
