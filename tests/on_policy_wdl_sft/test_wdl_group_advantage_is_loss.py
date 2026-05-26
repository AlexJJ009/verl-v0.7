import pytest
import torch

from verl.trainer.ppo.core_algos import get_policy_loss_fn
from verl.workers.config.actor import ActorConfig


def _config(clip_ratio_low=0.2, clip_ratio_high=0.27):
    config = ActorConfig.__new__(ActorConfig)
    config.clip_ratio = 0.2
    config.clip_ratio_low = clip_ratio_low
    config.clip_ratio_high = clip_ratio_high
    config.global_batch_info = {}
    config.policy_loss = {"loss_mode": "wdl_group_adv_is"}
    return config


def test_wdl_group_adv_is_gradient_only_flows_through_current_log_prob():
    old_log_prob = torch.zeros(2, 3, requires_grad=True)
    log_prob = torch.zeros(2, 3, requires_grad=True)
    advantages = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]], requires_grad=True)
    response_mask = torch.ones(2, 3)

    loss, metrics = get_policy_loss_fn("wdl_group_adv_is")(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="seq-mean-token-sum",
        config=_config(),
        rollout_is_weights=None,
    )
    loss.backward()

    assert log_prob.grad is not None
    assert old_log_prob.grad is None
    assert advantages.grad is None
    assert metrics["wdl_group_adv_is/ratio_mean"] == pytest.approx(1.0)
    assert metrics["wdl_group_adv_is/clipfrac_positive"] == pytest.approx(0.0)
    assert metrics["wdl_group_adv_is/clipfrac_negative"] == pytest.approx(0.0)


def test_wdl_group_adv_is_rejects_external_rollout_is_and_wrong_aggregation():
    old_log_prob = torch.zeros(2, 3)
    log_prob = torch.zeros(2, 3, requires_grad=True)
    advantages = torch.ones(2, 3)
    response_mask = torch.ones(2, 3)
    fn = get_policy_loss_fn("wdl_group_adv_is")

    with pytest.raises(ValueError, match="forbids rollout_is_weights"):
        fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="seq-mean-token-sum",
            config=_config(),
            rollout_is_weights=torch.ones(2, 3),
        )

    with pytest.raises(ValueError, match="seq-mean-token-sum"):
        fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="token-mean",
            config=_config(),
            rollout_is_weights=None,
        )


def test_wdl_group_adv_is_binary_mask_zeros_out_of_bound_tokens():
    old_log_prob = torch.zeros(2, 3)
    log_prob = torch.tensor(
        [
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    advantages = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    response_mask = torch.ones(2, 3)

    loss, metrics = get_policy_loss_fn("wdl_group_adv_is")(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode="seq-mean-token-sum",
        config=_config(),
        rollout_is_weights=None,
    )
    loss.backward()

    assert log_prob.grad[0, 0].item() == pytest.approx(0.0)
    assert log_prob.grad[1, 0].item() == pytest.approx(0.0)
    assert log_prob.grad[0, 1].item() < 0.0
    assert log_prob.grad[1, 1].item() > 0.0
    assert metrics["wdl_group_adv_is/clipfrac_positive"] == pytest.approx(1 / 3)
    assert metrics["wdl_group_adv_is/clipfrac_negative"] == pytest.approx(1 / 3)
