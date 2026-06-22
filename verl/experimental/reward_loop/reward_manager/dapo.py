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

import asyncio
import inspect
import logging

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score

logger = logging.getLogger(__file__)


def _code_reward_failure_info(
    *,
    score: float,
    status: str,
    response_str: str,
    stderr_excerpt: str,
    timeout: int,
    runtime_error: int,
) -> dict:
    return {
        "score": score,
        "acc": 0.0,
        "code_reward_status": status,
        "code_reward_extraction_fail": 0,
        "code_reward_compile_error": 0,
        "code_reward_runtime_error": runtime_error,
        "code_reward_timeout": timeout,
        "code_reward_dependency_error": 0,
        "code_reward_num_tests": 0,
        "code_reward_num_passed": 0,
        "code_reward_stderr_excerpt": stderr_excerpt[:1000],
        "pred": response_str[:4000],
        "verification_method": "reward_manager_fallback",
        "official_aligned": False,
        "code_reward_sandbox": "",
    }


@register("dapo")
class DAPORewardManager(RewardManagerBase):
    """DAPO Reward Manager."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)

        # DAPO Reward Config
        overlong_buffer_cfg = config.reward.get("reward_kwargs", {}).get("overlong_buffer_cfg", None)
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = config.reward.get("reward_kwargs", {}).get("max_resp_len", None)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        self.timeout = float(config.reward.get("timeout", 300.0))

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )
            assert not self.overlong_buffer_cfg.enable or self.overlong_buffer_cfg.len > 0, (
                "overlong_buffer.len must be positive when overlong penalty is enabled,"
                f"but got {self.overlong_buffer_cfg.len}."
                "To disable the overlong penalty, set overlong_buffer.enable = False"
            )

    async def _compute_reward(
        self, data_source: str, response_str: str, ground_truth, extra_info: dict, extra_reward_kwargs: dict
    ):
        if self.is_async_reward_score:
            return await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        return await self.loop.run_in_executor(
            None,
            lambda: self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            ),
        )

    def _timeout_result(self, data_source: str, response_str: str) -> dict:
        logger.warning(
            "Reward computation timed out after %.1fs for data_source=%s. Response preview: %r",
            self.timeout,
            data_source,
            response_str[:120],
        )
        return {
            "reward_score": -1.0,
            "reward_extra_info": _code_reward_failure_info(
                score=-1.0,
                status="timeout",
                response_str=response_str,
                stderr_excerpt=f"reward manager timeout after {self.timeout}s",
                timeout=1,
                runtime_error=0,
            ),
        }

    def _error_result(self, data_source: str, response_str: str, exc: Exception) -> dict:
        logger.exception(
            "Reward computation failed for data_source=%s. Response preview: %r",
            data_source,
            response_str[:120],
        )
        return {
            "reward_score": -1.0,
            "reward_extra_info": _code_reward_failure_info(
                score=-1.0,
                status="runtime_error",
                response_str=response_str,
                stderr_excerpt=str(exc),
                timeout=0,
                runtime_error=1,
            ),
        }

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        try:
            result = await asyncio.wait_for(
                self._compute_reward(data_source, response_str, ground_truth, extra_info, extra_reward_kwargs),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            return self._timeout_result(data_source, response_str)
        except Exception as exc:
            return self._error_result(data_source, response_str, exc)

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score

        if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
            overlong_buffer_len = self.overlong_buffer_cfg.len
            expected_len = self.max_resp_len - overlong_buffer_len
            exceed_len = valid_response_length - expected_len
            overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
            overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            reward += overlong_reward
            if self.overlong_buffer_cfg.log:
                reward_extra_info["overlong_reward"] = overlong_reward
                reward_extra_info["overlong"] = overlong_reward < 0

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
