# SPDX-License-Identifier: Apache-2.0

import torch

from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss_wdl_sft, kl_penalty
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config import ActorConfig, SubmodelKLConfig, SubmodelKLPairConfig


def _submodel_kl_loss(logprob, ref_logprob, mask, kl_type="low_var_kl", coef=1.0):
    kld = kl_penalty(logprob=logprob, ref_logprob=ref_logprob, kl_penalty=kl_type)
    return agg_loss(loss_mat=kld, loss_mask=mask, loss_agg_mode="token-mean") * coef


def test_submodel_kl_uses_existing_kl_penalty_types():
    logprob = torch.tensor([[-1.0, -0.5, -0.25], [-0.8, -0.7, -0.6]])
    ref_logprob = torch.tensor([[-1.1, -0.4, -0.5], [-0.9, -0.6, -0.55]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    for kl_type in ("kl", "mse", "low_var_kl"):
        actual = _submodel_kl_loss(logprob, ref_logprob, mask, kl_type=kl_type)
        expected = agg_loss(
            loss_mat=kl_penalty(logprob=logprob, ref_logprob=ref_logprob, kl_penalty=kl_type),
            loss_mask=mask,
            loss_agg_mode="token-mean",
        )
        assert torch.allclose(actual, expected, atol=1e-7)


def test_submodel_kl_off_matches_existing_wdl():
    old_log_prob = torch.tensor([[-0.9, -0.8, -0.7], [-0.6, -0.5, -0.4]])
    log_prob = torch.tensor([[-1.0, -0.7, -0.6], [-0.8, -0.45, -0.35]], requires_grad=True)
    advantages = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    response_mask = torch.ones_like(log_prob)

    legacy_config = ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1)

    disabled_config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        submodel_kl=SubmodelKLPairConfig(enabled=False),
    )

    legacy_loss, legacy_metrics = compute_policy_loss_wdl_sft(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=legacy_config,
    )
    disabled_loss, disabled_metrics = compute_policy_loss_wdl_sft(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=disabled_config,
    )

    assert torch.allclose(disabled_loss, legacy_loss, atol=1e-7)
    assert disabled_metrics == legacy_metrics


def test_independent_enable_disable_matrix():
    logprob1 = torch.tensor([[-1.0, -0.5]])
    ref1 = torch.tensor([[-1.2, -0.6]])
    logprob2 = torch.tensor([[-0.7, -0.4]])
    ref2 = torch.tensor([[-0.8, -0.45]])
    mask = torch.ones_like(logprob1)

    model1 = _submodel_kl_loss(logprob1, ref1, mask, kl_type="mse", coef=0.3)
    model2 = _submodel_kl_loss(logprob2, ref2, mask, kl_type="low_var_kl", coef=0.7)

    both_off = torch.zeros(())
    model1_only = model1
    model2_only = model2
    both_on = model1 + model2

    assert torch.allclose(both_off, torch.tensor(0.0))
    assert torch.allclose(model1_only, model1)
    assert torch.allclose(model2_only, model2)
    assert torch.allclose(both_on, model1 + model2)


def test_enabled_submodel_kl_selects_only_required_ref_tensors():
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.config = type(
        "Cfg",
        (),
        {
            "submodel_kl": SubmodelKLPairConfig(
                enabled=True,
                model1=SubmodelKLConfig(enabled=True, coef=0.1, kl_type="mse"),
                model2=SubmodelKLConfig(enabled=False, coef=0.0, kl_type="low_var_kl"),
            )
        },
    )()

    enabled = actor._enabled_submodel_kl_indices()

    assert enabled == [0]
    assert [actor._submodel_ref_logprob_key(i) for i in enabled] == ["model1_ref_log_probs"]


def test_disabled_submodel_kl_requires_no_ref_tensors():
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.config = type("Cfg", (), {"submodel_kl": SubmodelKLPairConfig()})()

    assert actor._enabled_submodel_kl_indices() == []
