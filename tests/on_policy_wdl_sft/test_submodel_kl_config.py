from omegaconf import OmegaConf

from verl.trainer.ppo.utils import (
    is_submodel_kl_enabled,
    need_reference_policy,
    validate_submodel_kl_reference_paths,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config.actor import ActorConfig, SubmodelKLConfig, SubmodelKLPairConfig


def _base_ppo_config():
    return OmegaConf.create(
        {
            "algorithm": {"use_kl_in_reward": False},
            "actor_rollout_ref": {
                "actor": {
                    "use_kl_loss": False,
                    "submodel_kl": {
                        "enabled": False,
                        "model1": {"enabled": False, "coef": 0.0, "kl_type": "low_var_kl", "ref_path": None},
                        "model2": {"enabled": False, "coef": 0.0, "kl_type": "low_var_kl", "ref_path": None},
                    },
                }
            },
        }
    )


def test_default_submodel_kl_config_is_disabled():
    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
    )

    assert isinstance(config.submodel_kl, SubmodelKLPairConfig)
    assert config.submodel_kl.enabled is False
    assert config.submodel_kl.model1.enabled is False
    assert config.submodel_kl.model2.enabled is False
    assert config.submodel_kl.model1.coef == 0.0
    assert config.submodel_kl.model2.coef == 0.0
    assert config.submodel_kl.is_effective() is False


def test_submodel_kl_yaml_defaults_do_not_require_reference_lifecycle():
    from hydra import compose, initialize_config_dir
    import os

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
        cfg = compose(config_name="actor", overrides=["strategy=fsdp", "ppo_micro_batch_size_per_gpu=128"])

    config = omega_conf_to_dataclass(cfg)

    assert isinstance(config, ActorConfig)
    assert isinstance(config.submodel_kl, SubmodelKLPairConfig)
    assert isinstance(config.submodel_kl.model1, SubmodelKLConfig)
    assert isinstance(config.submodel_kl.model2, SubmodelKLConfig)
    assert config.submodel_kl.enabled is False
    assert config.submodel_kl.is_effective() is False


def test_submodel_kl_requires_reference_lifecycle():
    cfg = _base_ppo_config()
    cfg.actor_rollout_ref.actor.submodel_kl.enabled = True
    cfg.actor_rollout_ref.actor.submodel_kl.model2.enabled = True
    cfg.actor_rollout_ref.actor.submodel_kl.model2.coef = 0.05

    assert is_submodel_kl_enabled(cfg) is True
    assert need_reference_policy(cfg) is True


def test_submodel_kl_coef_zero_does_not_require_reference_lifecycle():
    cfg = _base_ppo_config()
    cfg.actor_rollout_ref.actor.submodel_kl.enabled = True
    cfg.actor_rollout_ref.actor.submodel_kl.model1.enabled = True
    cfg.actor_rollout_ref.actor.submodel_kl.model1.coef = 0.0

    assert is_submodel_kl_enabled(cfg) is False
    assert need_reference_policy(cfg) is False


def test_legacy_reference_lifecycle_still_works_without_submodel_kl():
    cfg = _base_ppo_config()

    cfg.algorithm.use_kl_in_reward = True
    assert need_reference_policy(cfg) is True

    cfg.algorithm.use_kl_in_reward = False
    cfg.actor_rollout_ref.actor.use_kl_loss = True
    assert need_reference_policy(cfg) is True


def test_invalid_submodel_kl_type_fails_fast():
    try:
        SubmodelKLConfig(enabled=True, coef=0.1, kl_type="not-a-kl")
    except ValueError as exc:
        assert "Invalid submodel KL type" in str(exc)
    else:
        raise AssertionError("invalid submodel KL type should fail")


def test_negative_submodel_kl_coef_fails_fast():
    try:
        SubmodelKLConfig(enabled=True, coef=-0.1, kl_type="low_var_kl")
    except ValueError as exc:
        assert "coef must be non-negative" in str(exc)
    else:
        raise AssertionError("negative submodel KL coef should fail")


def test_missing_enabled_reference_fails_fast():
    cfg = _base_ppo_config()
    cfg.actor_rollout_ref.actor.submodel_kl.enabled = True
    cfg.actor_rollout_ref.actor.submodel_kl.model2.enabled = True
    cfg.actor_rollout_ref.actor.submodel_kl.model2.coef = 0.05

    try:
        validate_submodel_kl_reference_paths(cfg, strict=True)
    except ValueError as exc:
        assert "model2.ref_path is missing" in str(exc)
    else:
        raise AssertionError("enabled model2 KL without a resolved reference path should fail")


def test_incompatible_enabled_reference_fails_fast():
    cfg = _base_ppo_config()
    cfg.actor_rollout_ref.actor.submodel_kl.enabled = True
    cfg.actor_rollout_ref.actor.submodel_kl.model1.enabled = True
    cfg.actor_rollout_ref.actor.submodel_kl.model1.coef = 0.05
    cfg.actor_rollout_ref.actor.submodel_kl.model1.ref_path = "/tmp/model1-ref"

    try:
        validate_submodel_kl_reference_paths(
            cfg,
            strict=True,
            path_exists=lambda path: True,
            compatible=lambda name, path: False,
        )
    except ValueError as exc:
        assert "model1 submodel KL reference path is incompatible" in str(exc)
    else:
        raise AssertionError("incompatible model1 KL reference path should fail")
