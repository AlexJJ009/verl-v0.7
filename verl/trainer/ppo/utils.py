# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from enum import Enum

from omegaconf import DictConfig

from verl.single_controller.base import Worker
from verl.trainer.ppo.core_algos import AdvantageEstimator

WorkerType = type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    Env = 7

    def __str__(self):
        return self._get_role_string()

    def _get_role_string(self):
        role_mapping = {
            Role.Actor: "actor",
            Role.Rollout: "rollout",
            Role.ActorRollout: "actor_rollout",
            Role.Critic: "critic",
            Role.RefPolicy: "ref",
            Role.RewardModel: "rm",
            Role.ActorRolloutRef: "actor_rollout_ref",
        }
        return role_mapping.get(self, self.name.lower())

    @classmethod
    def from_string(cls, name: str):
        string_mapping = {
            "actor": cls.Actor,
            "rollout": cls.Rollout,
            "actor_rollout": cls.ActorRollout,
            "critic": cls.Critic,
            "ref": cls.RefPolicy,
            "rm": cls.RewardModel,
            "actor_rollout_ref": cls.ActorRolloutRef,
        }
        role = string_mapping.get(name.lower())
        if role is None:
            raise ValueError(f"No Role found for string: {name}")
        return role


def need_reference_policy(
    config: DictConfig,
) -> bool:
    """Given the config, do we need ref policy."""
    return (
        config.algorithm.use_kl_in_reward
        or config.actor_rollout_ref.actor.use_kl_loss
        or is_submodel_kl_enabled(config)
    )


def _conf_get(config, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def is_submodel_kl_enabled(config: DictConfig) -> bool:
    """Return true when per-submodel KL requires reference log-prob computation."""
    actor = _conf_get(_conf_get(config, "actor_rollout_ref"), "actor")
    submodel_kl = _conf_get(actor, "submodel_kl")
    if not _conf_get(submodel_kl, "enabled", False):
        return False

    for name in ("model1", "model2"):
        model_cfg = _conf_get(submodel_kl, name)
        if _conf_get(model_cfg, "enabled", False) and float(_conf_get(model_cfg, "coef", 0.0) or 0.0) > 0.0:
            return True
    return False


def validate_submodel_kl_reference_paths(
    config: DictConfig,
    *,
    strict: bool = False,
    path_exists=None,
    compatible=None,
) -> None:
    """Validate enabled per-submodel KL reference paths.

    The training wrappers may leave ``ref_path`` empty while they resolve the
    default Stage2-start reference from launch provenance. In strict mode, used
    after that resolution step, every effective submodel KL must have a present
    and compatible reference path.
    """
    actor = _conf_get(_conf_get(config, "actor_rollout_ref"), "actor")
    submodel_kl = _conf_get(actor, "submodel_kl")
    if not _conf_get(submodel_kl, "enabled", False):
        return

    if path_exists is None:
        import os

        path_exists = os.path.exists

    for name in ("model1", "model2"):
        model_cfg = _conf_get(submodel_kl, name)
        if not (
            _conf_get(model_cfg, "enabled", False)
            and float(_conf_get(model_cfg, "coef", 0.0) or 0.0) > 0.0
        ):
            continue
        ref_path = _conf_get(model_cfg, "ref_path")
        if strict and not ref_path:
            raise ValueError(f"{name} submodel KL is enabled but {name}.ref_path is missing")
        if ref_path and not path_exists(ref_path):
            raise ValueError(f"{name} submodel KL reference path does not exist: {ref_path}")
        if ref_path and compatible is not None and not compatible(name, ref_path):
            raise ValueError(f"{name} submodel KL reference path is incompatible: {ref_path}")


def need_reward_model(
    config: DictConfig,
) -> bool:
    """Given the config, do we need reward model."""
    return config.reward.reward_model.enable


def need_critic(config: DictConfig) -> bool:
    """Given a config, do we need critic."""
    if config.critic.enable is not None:
        return bool(config.critic.enable)
    elif config.algorithm.adv_estimator == AdvantageEstimator.GAE:
        return True
    else:
        warnings.warn(
            "Disabled critic as algorithm.adv_estimator != gae. If it is not intended, please set critic.enable=True",
            stacklevel=2,
        )
        return False
