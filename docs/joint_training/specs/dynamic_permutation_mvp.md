# Target-Preserving Dynamic Permutation MVP

Status: GON-34 implementation merged into the formal training branch at merge commit `8209576c04d89c7d778a249e8458c608f747c764`. Candidate-bound CPU and 8xL40S Slurm GPU/FSDP engineering smoke evidence passed. Formal P20/P30/P60 experiments are not part of the GON-34 Delivery evidence and have not been launched.

## Scope

Dynamic Permutation is a teacher-forced training intervention on the weak branch logits of `QwenJointForCausalLM`. It is disabled for rollout, validation, old/reference log-prob computation, eval-only forwards, and diagnostic counterfactual forwards unless a later approved experiment changes that boundary.

The intervention is configured by `actor_rollout_ref.actor.weak_logit_permutation` and is independent of `freeze_model1`. Different `rho` values do not require rebuilding model caches or changing checkpoint topology.

## Transform contract

For each active supervised causal-logit row, let the hard target be `y` and vocabulary size be `V`.

1. `rho` must be finite and in `[0, 1]`.
2. `k = floor(rho * (V - 1))` non-target coordinates are selected.
3. The target coordinate is never selected and its logit is unchanged.
4. `k = 0` is an exact no-op: the original tensor object is returned before hashing, allocation, or process-global RNG use.
5. `k = 1` fails closed because no fixed-point-free permutation exists.
6. `k >= 2` applies one fixed-point-free cycle to the selected set.

The implementation in `verl/models/joint_model/dynamic_permutation.py` uses a stateless keyed cyclic selected-set:

- the selected non-target coordinates are a cyclic window over compact non-target coordinates, shifted by a hash of `(base_seed, global_step, actor_update_index, dynperm_sample_id, absolute_token_position)`;
- the cycle rotation is non-zero, so selected coordinates have no fixed points;
- selected compact coordinates are lifted back to real vocabulary coordinates by skipping the target index.

This is an exact bijection over the selected set. It preserves the target logit, weak-logit value multiset, weak entropy, and centered-logit norm up to normal floating-point comparison tolerance. It intentionally does not preserve fused entropy, fused target probability, or cross-model affinity.

## Memory and audit budget

The transform processes active rows in chunks. It materializes only `O(row_chunk_size * k)` integer indices, where `k = floor(rho * (V - 1))`; it never constructs a `[token_rows, vocab]` permutation tensor.

Runtime telemetry is bounded:

- always-on counters: requested/realized `rho`, active rows, selected count, selected coordinates, fixed points, target mismatches, audited rows, invariant failures;
- sampled audits: weak entropy error and sorted-value/multiset error over deterministic bounded rows.

Any sampled target, entropy, or multiset invariant failure raises immediately. Exhaustive full-vocabulary sorting belongs in CPU fixtures, not every production token row.

## Determinism identities

The mapping is fully stateless. The seed material is:

- actor base seed;
- restored global optimizer step;
- actor update index, derived from PPO epoch and mini-batch index;
- `dynperm_sample_id`, assigned by the trainer after rollout expansion and final actor-row construction but before worker dispatch, balancing, mini-batching, micro-batching, remove-padding, dynamic batching, or sequence-parallel slicing;
- absolute causal token position.

The mapping is stable across micro-batches, gradient accumulation, and gradient-checkpoint recomputation within the same optimizer update. It changes only when the restored global step or actor update index changes. Rank-local row ordinals, random UUIDs, and DP rank are not seed material.

## Actor and training boundary

`RayPPOTrainer` assigns `dynperm_sample_id` only when the weak-logit permutation config is enabled and passes `global_step` through actor metadata. `DataParallelPPOActor.update_policy` is the only path that sets `apply_weak_logit_permutation=True`.

`compute_log_prob`, reference-policy log-prob, rollout, validation, generation, eval-only, and diagnostic forwards do not enable the transform. Enabling the transform with fused kernels or PrefixGrouper is fail-closed; remove-padding depends on the repository attention backend gate.

The MVP is fail-closed to `policy_loss.loss_mode=wdl_sft`, with `entropy_coeff=0`, actor reference KL disabled, and effective per-submodel KL disabled. PPO/GRPO/IS losses, including `wdl_sft_is` and `wdl_group_adv_is`, form ratios against `old_log_prob`; because those old-log-prob forwards are intentionally unpermuted, enabling Dynamic Permutation would mix the intervention delta into the policy-staleness ratio. Actor KL and submodel KL have the same mismatch against their intentionally unpermuted reference log-probabilities. Supporting any such ratio or reference objective therefore requires a later approved algorithm contract that binds the same permutation identity into its reference-side computation. Nonzero entropy regularization is also excluded from the MVP so the intervention remains a pure teacher-forced WDL-SFT objective; `calculate_entropy=true` remains allowed as detached diagnostics when its coefficient is zero.

## Freeze, checkpoint, and no-op contract

The same joint actor checkpoint remains canonical. It keeps both `sub_models.0.*` and `sub_models.1.*` namespaces; Dynamic Permutation does not add a second checkpoint writer or permutation RNG state.

With `freeze_model1=false`, gradients flow through the gather/scatter permutation into Model1 and Model2 remains trainable. With `freeze_model1=true`, Model1 remains frozen while Model2 trains normally. The engineering no-op oracle is same-candidate `enabled=false` versus `enabled=true,rho=0`; historical C/fixed-M1 equivalence is a later scientific bridge, not the Delivery oracle.

## Current evidence boundary

As of 2026-08-20, the GON-34 code candidate has CPU plus candidate-bound Slurm engineering evidence:

- focused Dynamic Permutation, actor/config plumbing, checkpoint/no-op, sharded-checkpoint capacity estimation, and hardened Slurm-contract gate: `114 passed` in the final focused offline CPU harness with `CUDA_VISIBLE_DEVICES=""`;
- broader `tests/joint_training` still has pre-existing or environment-dependent failures unrelated to the Dynamic Permutation focused gate.
- candidate-bound Slurm Job 146 completed an 8xL40S FSDP engineering smoke with result `PASS`, covering both `freeze_model1=false` and `freeze_model1=true`, checkpoint save/resume, and same-candidate `rho=0` comparison. Its receipt is explicitly `formal_experiment=false`.

The Slurm wrapper remains part of the engineering admission contract: it fail-closes unless the candidate SHA is an exact lowercase 40-hex value, the staged workspace and output roots canonicalize below the intended node-local `workspace/jobs` and `checkpoints/jobs` prefixes, and the node-local preflight records durable GPU-process, Docker-container, and Slurm-allocation receipts showing no foreign workload before the job container starts. Its eight local ranks use explicit `127.0.0.1` static rendezvous so `--network=none` does not depend on container-hostname resolution. Formal P20/P30/P60 training remains outside Delivery; prepared experiment wrappers require separate exact-dose-and-horizon launch receipts.

The resource-threshold smoke uses the production defaults `row_chunk_size=16` and `audit_rows=4`; invariant auditing remains enabled on every active step. The chunk default is the bounded tradeoff required by the 8xL40S gate: chunk 8 passed the 15% memory bound but failed the 25% step-time bound, while the earlier artificial chunk-64/audit-32 stress setting passed time but failed memory.
