# Joint Training Stabilization Experience Notes

## Purpose

This note captures the main engineering lessons from the Stage 1 bring-up and stabilization of joint GRPO training in `verl`. The goal is to preserve the useful patterns from the recent debugging cycle so Stage 2 can focus on algorithm logic instead of rediscovering infrastructure failure modes.

## Milestone Context

Stage 1 ended when `recipe/joint_training/run_joint_grpo_qwen3_1.7b.sh` completed a full `100`-step run in `recipe/joint_training/Joint-GRPO-Qwen3-1.7B-GSM8K_1773032262.log`, saved checkpoints on `/data-1`, and emitted periodic merged training-plus-validation metrics.

The path to that result was not one bug. It was a chain of interacting failures across distributed control flow, per-GPU memory pressure, filesystem topology, optional dependencies, and observability.

## Commit Landmarks And What They Taught

1. `384804fe` `feat(joint_training): add joint vLLM rollout support`
   - Lesson: the correct abstraction boundary is the model and weight-extraction layer. Keep rollout and PPO logic as unchanged as possible.

2. `6151c24` in `recipe/` `joint_training: default GRPO recipe to vLLM rollout`
   - Lesson: switching the default serving backend is not a cosmetic recipe change. It changes memory, startup, failure shape, and debug surface.

3. `d0c5d3a` in `recipe/` `fix(joint_training): harden rollout memory and checkpoint paths`
   - Lesson: recipe defaults must encode real host constraints. Generic defaults are not enough on a specific H800 machine with nontrivial mount topology.

4. `428a7e83` `fix(joint_training): stabilize fsdp actor rollout path`
   - Lesson: distributed correctness can fail before memory or math fails. DP-group consistency is a semantic invariant, not an optimization detail.

5. `73404180` `fix(checkpoint): harden fsdp shard saves against disk pressure`
   - Lesson: a checkpoint save must be treated as a transaction. Direct writes to final files are too fragile under disk pressure.

6. `5bd62896` `fix(vllm_rollout): parameterize colocated zmq socket paths`
   - Lesson: hidden root-mounted side effects like `/tmp` can kill a healthy training job. Runtime paths must be explicit.

7. `59a4c534` `docs(tests): refresh joint training recipe coverage`
   - Lesson: stabilization work is not finished until the guards are covered by regression tests and reflected in documentation.

8. `5b3aca2` in `recipe/` `fix(joint_training): persist local metric logs`
   - Lesson: experiment tracking cannot depend on one backend. Local persistent metrics are part of robustness.

9. `c4436d2b` `fix(trainer): print periodic test-step metrics`
   - Lesson: if critical metrics are only visible at teardown, the system is still under-instrumented.

## Recurrent Failure Classes

### 1. Per-GPU Memory, Not Cluster Memory

The H800 server has 8 cards, but each actor or colocated rollout process still lives within one card's memory budget. Several early misreads came from treating 8 GPUs as if their memory pooled into one large heap. They do not.

Practical rule:

1. Diagnose OOM by rank and by device.
2. Separate actor, ref, critic, and rollout residency on the same GPU.
3. Treat colocated vLLM as part of the same memory budget as the actor path.

### 2. Distributed Control-Flow Drift Is Fatal

The step-2 deadlock showed that if actor and critic dynamic batching disagree across DP ranks, NCCL hangs rather than producing a friendly error.

Practical rule:

1. DP-group context must be wired explicitly.
2. Dynamic batch construction must be deterministic within the distributed group.
3. When collectives hang, compare control flow and micro-batch schedules before assuming a network problem.

### 3. Dense Full-Vocab Operations Become Expensive Fast

Old-log-prob recompute and entropy calculation exposed how costly dense `batch x seq x vocab` tensors become, even on a 1.7B model.

Practical rule:

1. Avoid creating extra full-vocab temporaries in the fused joint-logit path.
2. Do not compute entropy unless the config truly requires it.
3. Chunk dense entropy paths when fallback code cannot use remove-padding.

### 4. Optional Dependencies Must Fail Safe

`flash_attn` absence caused a late crash only after training had already started.

Practical rule:

1. Optional acceleration dependencies need preflight checks.
2. If the fast path is unavailable, the recipe should automatically shift to a slower but safe configuration.
3. The runtime error should be explicit, not buried in a worker traceback.

### 5. Filesystem Topology Is Part Of The Runtime Design

Several failures had nothing to do with RL logic:

1. Checkpoints were initially writing to `/data-2`, which resolved to a small root filesystem on this host.
2. vLLM and ZMQ indirectly used root-mounted locations like `/tmp` and `/root/.config`.

Practical rule:

1. Treat checkpoint, temp, config, and IPC roots as first-class config.
2. Add preflight free-space checks for the checkpoint target.
3. Prefer `/data-1` for all large or high-churn runtime paths on this server.

### 6. Save Paths Must Be Atomic

Disk-full events left partial checkpoint outputs behind until shard saving was hardened.

Practical rule:

1. Save to temporary files first.
2. Rename atomically on success.
3. Clean partial artifacts on failure.

### 7. Observability Must Be Designed, Not Assumed

The metrics issue showed that data can exist in W&B history while still being effectively invisible during debugging.

Practical rule:

1. Important metrics must be printed at meaningful milestones during the run.
2. Use more than one sink: stdout, W&B, and a local jsonl file.
3. Joint training needs its own metrics, not only generic PPO/GRPO metrics.

## Practices That Worked Well

1. Use the model layer as the main place to encapsulate joint logit fusion.
2. Prefer narrowly targeted regression tests for each failure that was actually seen in GPU runs.
3. Harden the launcher with host-aware defaults instead of assuming the environment matches upstream examples.
4. Keep recipe fixes and framework fixes separate in reasoning:
   - recipe fix when the issue is host- or run-shape-specific
   - framework fix when the issue is semantic or reusable

## What Stage 2 Should Do Better

Stage 1 mostly answered: can the system run?

Stage 2 needs to answer:

1. Is the fused joint policy semantically correct across rollout, old-log-prob recompute, update, and eval-only validation?
2. Which metrics reveal whether joint training is helping or simply hiding problems behind generic GRPO outputs?
3. Where are the intrinsic incompatibilities between joint training and assumptions built for single-model PPO/GRPO?
4. Which fallbacks should stay in the recipe, and which should graduate into shared framework behavior?

## Suggested Stage 2 Joint-Specific Metrics

1. Fused-vs-model2 log-prob gap on validation batches.
2. Fused-vs-model1 and fused-vs-model2 KL estimates.
3. Mean absolute and max absolute logits disagreement between the two submodels.
4. Per-token contribution balance from model1 versus model2 under the current `lambda`.
5. Rollout-time fused-policy metrics versus eval-only model2 metrics in the same reporting block.
6. Model1 and model2 gradient norms, plus a ratio or gap metric to expose chronic update imbalance.
7. Reward-extraction failure buckets such as `[NO_BOXED]`, empty extraction, and other parse failures.

## Early Stage 2 Follow-Up Lessons

The first real Stage 2 instrumentation pass added two missing observability pieces:

1. validation prompt/response samples in stdout, tracking, and jsonl dumps
2. per-submodel gradient norms under `jointTraining/`

That work immediately produced three useful lessons.

### 1. Scalar Reward Collapse Hid Multiple Different Failures

Before the new logging, the run only told us that GSM8K validation reward was flat at `-1` and accuracy was `0`. That was not enough to tell whether the policy was:

1. producing unreadable junk
2. answering coherently but incorrectly
3. answering correctly but in the wrong format for reward extraction

The new validation examples showed all three patterns in one live run:

1. some outputs were clearly garbled or off-domain
2. some outputs were fluent but mathematically wrong
3. some outputs contained the correct final number but still failed extraction, e.g. `pred = [NO_BOXED]`
4. some outputs reached successful answer extraction, but still logged `answer_correct = false` under `verl_math_verify`, which points to a verifier mismatch rather than only a formatting mismatch

Lesson:

1. joint training cannot be debugged from scalar reward alone
2. answer-format failures must be separated from reasoning failures
3. reward-verifier mismatches must be separated from both formatting failures and reasoning failures
4. sampled validation generations are not optional debugging sugar; they are part of the algorithm observability surface

### 2. Real E2E Reruns Still Find Lifecycle Bugs That Unit Tests Miss

The first instrumentation rerun did not fail in the reward path. It failed earlier, during vLLM startup, with `EADDRINUSE` on `data_parallel_master_port`.

Root cause:

1. the async vLLM server reserved multiple startup sockets
2. only `_master_sock` was being closed before server launch
3. `_dp_rpc_sock` and `_dp_master_sock` could remain open long enough to collide with the real vLLM bind

Fix:

1. close all reserved startup sockets before the server is launched
2. keep that behavior covered by a focused regression test

Lesson:

1. startup resource lifecycle is part of runtime correctness
2. real reruns are still required even after unit tests pass
3. Stage 2 observability work should expect to uncover adjacent runtime bugs, not only metric gaps

### 3. Joint Gradient Metrics Need To Be Logged Even When They Are Zero

The first post-fix training step showed:

1. `actor/grad_norm = 0.0`
2. `jointTraining/model1_grad_norm = 0.0`
3. `jointTraining/model2_grad_norm = 0.0`

This is useful, not redundant.

Lesson:

1. the zero values confirm that the new metrics are wired into the real training path
2. they also prove that the current failure is upstream in reward/advantage formation, not only inside backprop logging
3. once reward starts moving, these metrics will be the baseline evidence for whether the two submodels are both receiving updates or one side is dominating

### 4. Full E2E Confirmation Still Matters After Observability Is Added

The second Stage 2 rerun did not only survive the early validation steps. It completed the full `100 / 100` schedule, saved checkpoints at `20`, `40`, `60`, `80`, and `100`, and kept printing sampled validation outputs the entire time.

That final rerun taught two practical lessons.

1. The observability patch itself is operationally safe:
   - the new validation logging path did not break checkpointing, validation cadence, or final teardown
   - separate `RUN_PREFIX` values preserved distinct log files and artifacts for each rerun
2. The algorithmic failure is persistent rather than a short warmup artifact:
   - the final validation still had `val-core/openai/gsm8k/acc/mean@1 = 0.0`
   - the final validation still had `val-aux/openai/gsm8k/reward/mean@1 = -1.0`
   - the final actor and joint submodel grad norms were still all `0.0`
   - the final samples still contained formatting failures, verifier mismatches, and off-domain or garbled generations

Lesson:

1. if a Stage 2 change is meant to improve observability, it still needs one full real rerun, not only a smoke test
2. a complete lifecycle run is now the baseline proof that a logging change is safe to keep
3. once that full rerun passes, the remaining blockers can be treated as algorithmic rather than infrastructural

## Bottom Line

The best outcome from Stage 1 is not only that the recipe runs. It is that the project now has a sharper definition of where joint training really becomes different:

1. memory shape
2. distributed control-flow consistency
3. validation semantics
4. environment-sensitive runtime defaults
5. observability requirements beyond the mature baseline algorithms

That is the starting point for the next phase.
