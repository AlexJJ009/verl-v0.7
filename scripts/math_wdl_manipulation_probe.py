#!/usr/bin/env python3
"""Fail-closed tiny-model manipulation receipt for the Math WDL C/D/D0 arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from verl.models.joint_model.configuration_joint_qwen3 import QwenJointConfig
from verl.models.joint_model.modeling_joint_qwen3 import QwenJointForCausalLM


def _tiny_model(*, fusion_lambda: float, fusion_mode: str) -> QwenJointForCausalLM:
    torch.manual_seed(20260727)
    config = QwenJointConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        fusion_lambda=fusion_lambda,
        fusion_mode=fusion_mode,
    )
    return QwenJointForCausalLM(config)


def _parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _grad_norm(model: torch.nn.Module) -> float:
    return sum(float(parameter.grad.detach().float().square().sum()) for parameter in model.parameters() if parameter.grad is not None) ** 0.5


def run_probe() -> dict:
    input_ids = torch.tensor([[1, 5, 11, 17, 23, 29], [1, 7, 13, 19, 31, 37]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[5, 11, 17, 23, 29, 41], [7, 13, 19, 31, 37, 43]])

    arms = {
        "C": _tiny_model(fusion_lambda=0.8, fusion_mode="mixture"),
        "D": _tiny_model(fusion_lambda=1.0, fusion_mode="mixture"),
        "D0": _tiny_model(fusion_lambda=0.8, fusion_mode="strong_scaled"),
    }
    diagnostics: dict[str, dict] = {}
    for arm, model in arms.items():
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        model1_hash_before = _parameter_hash(model.sub_models[0])
        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask, return_submodel_logits=True)
        direct_model2_logits = output.submodel_logits[1]
        fused_nll = F.cross_entropy(output.logits[:, :-1].reshape(-1, output.logits.shape[-1]), labels[:, 1:].reshape(-1))
        direct_model2_nll = F.cross_entropy(
            direct_model2_logits[:, :-1].reshape(-1, direct_model2_logits.shape[-1]),
            labels[:, 1:].reshape(-1),
        )
        fused_nll.backward()
        model1_grad_norm = _grad_norm(model.sub_models[0])
        model2_grad_norm = _grad_norm(model.sub_models[1])
        optimizer.step()
        model1_hash_after = _parameter_hash(model.sub_models[0])

        if arm == "D0":
            expected_logits = 0.8 * direct_model2_logits
        elif arm == "D":
            expected_logits = direct_model2_logits
        else:
            expected_logits = 0.2 * output.submodel_logits[0] + 0.8 * direct_model2_logits
        diagnostics[arm] = {
            "fusion_lambda": model.fusion_lambda,
            "fusion_mode": model.fusion_mode,
            "formula_max_abs_error": float((output.logits - expected_logits).abs().max().detach()),
            "model1_grad_norm": model1_grad_norm,
            "model2_grad_norm": model2_grad_norm,
            "model1_hash_unchanged_after_step": model1_hash_before == model1_hash_after,
            "fused_vs_direct_model2_nll_delta": float(fused_nll.detach() - direct_model2_nll.detach()),
        }

    # Replacement invariance: only C may depend on Model1.
    for arm, model in arms.items():
        model.eval()
        with torch.no_grad():
            before = model(input_ids=input_ids, attention_mask=attention_mask).logits.clone()
            for parameter in model.sub_models[0].parameters():
                parameter.add_(0.25)
            after = model(input_ids=input_ids, attention_mask=attention_mask).logits
        diagnostics[arm]["model1_replacement_max_abs_logit_change"] = float((after - before).abs().max())

    tolerance = 1e-6
    checks = {
        "C_depends_on_model1": diagnostics["C"]["model1_replacement_max_abs_logit_change"] > tolerance,
        "C_updates_both_models": diagnostics["C"]["model1_grad_norm"] > 0 and diagnostics["C"]["model2_grad_norm"] > 0,
        "D_is_direct_model2": diagnostics["D"]["formula_max_abs_error"] <= tolerance,
        "D_ignores_and_does_not_update_model1": diagnostics["D"]["model1_grad_norm"] == 0 and diagnostics["D"]["model1_hash_unchanged_after_step"],
        "D0_is_matched_scale_model2": diagnostics["D0"]["formula_max_abs_error"] <= tolerance,
        "D0_ignores_and_does_not_update_model1": diagnostics["D0"]["model1_grad_norm"] == 0 and diagnostics["D0"]["model1_hash_unchanged_after_step"],
        "D_and_D0_are_model1_invariant": diagnostics["D"]["model1_replacement_max_abs_logit_change"] <= tolerance
        and diagnostics["D0"]["model1_replacement_max_abs_logit_change"] <= tolerance,
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(checks.values()) else "fail",
        "tolerance": tolerance,
        "checks": checks,
        "diagnostics": diagnostics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
