# Stage 2 Model2-Rollout Fused-Loss Fast Validation Status

Last updated: 2026-05-30

## Current State

- Branch: `feature/on-policy-wdl-sft`
- Current milestone: plan contract locked after user clarification; matched-beta
  scripts and dry-run checks are corrected; both matched-beta runtime runs
  completed and final metrics are verified.
- Acceptance tier: local-only PASS. Meituan/AFO is `NOT ACCEPTED` for this plan until explicitly implemented and reviewed.
- Training launch state: no Stage 2 tmux session, Docker training container, or
  GPU workload remains.
- Runtime state at plan start: staged v1 Stage 1 beta queue and monitor were stopped; no `verl-harness` training container remained; all 8 GPUs were idle.

## Implemented So Far

- Added plan: `docs/joint_training/plans/active/stage2_model2_rollout_fused_loss_fast_validation.md`.
- Added trainer switch:
  - `actor_rollout_ref.model.joint_training_rollout_source=model2`
  - training/validation rollout weight sync uses Model2-only extraction while actor training remains joint.
- Added data shard generator:
  - `recipe/on_policy_wdl_sft/staged_v1/create_stage2_nonoverlap_shard.py`
- Added Stage 2 common wrapper:
  - `recipe/on_policy_wdl_sft/staged_v1/_run_stage2_model2_rollout_common.sh`
- Added matched-beta default wrappers:
  - `recipe/on_policy_wdl_sft/staged_v1/run_s2_from_s1_beta0_beta0.sh`
  - `recipe/on_policy_wdl_sft/staged_v1/run_s2_from_s1_beta01_beta01.sh`
- Added cross-beta diagnostic wrappers:
  - `recipe/on_policy_wdl_sft/staged_v1/run_s2_from_s1_beta01_beta0.sh`
  - `recipe/on_policy_wdl_sft/staged_v1/run_s2_from_s1_beta0_beta01.sh`
- Added local queue:
  - `recipe/on_policy_wdl_sft/staged_v1/run_stage2_fast_validation_queue.sh`
- Added WxPusher-aware queue/monitor:
  - queue script now sends notifications for queue start, individual launch, run completion, queue completion, and failure.
  - `recipe/on_policy_wdl_sft/staged_v1/monitor_stage2_fast_validation_queue_notify.sh`

## Required Inputs

Stage 1 beta `0.0`:

- Run: `ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA0-V1_1779962803`
- Checkpoint: `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA0-V1_1779962803/global_step_85/actor`
- Stage 1 baseline metric: `val-core/HuggingFaceH4/MATH-500/acc/mean@3=0.7325268817204301`

Stage 1 beta `0.1`:

- Run: `ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA01-V1_1779981295`
- Checkpoint: `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA01-V1_1779981295/global_step_150/actor`
- Stage 1 baseline metric: `val-core/HuggingFaceH4/MATH-500/acc/mean@3=0.7573924731182795`

## Completed Gates

- Stage 2 non-overlap shard generated and verified:
  - Path: `/data-1/dataset/EnsembleLLM-data-processed/staged_v1/stage2_after_s1_150steps_seed20260528_75steps.parquet`
  - Manifest: `/data-1/dataset/EnsembleLLM-data-processed/staged_v1/stage2_after_s1_150steps_seed20260528_75steps.manifest.json`
  - Raw source rows: `104916`
  - Eligible rows after Stage 1-compatible prompt filtering: `104673`
  - Offset: `9600` prompts (`150 * 64`)
  - Length: `4800` prompts (`75 * 64`)
  - Overlap with consumed Stage 1 prefix: `0`
  - SHA256: `28cdd1d9f0c5c06d7eab768b264ad175830f72d8c07ad66652aa43de862435eb`
  - First 10 `extra_info.index`: `109719, 111365, 88991, 58015, 24690, 37418, 66497, 110437, 83593, 15772`
  - Last 10 `extra_info.index`: `32774, 41205, 35439, 10950, 110171, 62866, 66622, 62003, 100706, 11795`
- Original shell syntax checks passed:
  - `bash -n recipe/on_policy_wdl_sft/staged_v1/*.sh recipe/on_policy_wdl_sft/staged_v1/meituan/*.sh platform/hope_staged_v1/*.sh`
- Python compile checks passed before launch preparation:
  - `python3 -m py_compile recipe/on_policy_wdl_sft/staged_v1/create_stage2_nonoverlap_shard.py verl/trainer/ppo/ray_trainer.py`
- Wrapper dry-runs passed without triggering merge/training:
  - `STAGE2_DRY_RUN=1 bash recipe/on_policy_wdl_sft/staged_v1/run_s2_from_s1_beta01_beta0.sh`
  - `STAGE2_DRY_RUN=1 bash recipe/on_policy_wdl_sft/staged_v1/run_s2_from_s1_beta0_beta0.sh`
  - Both render `JOINT_TRAINING_ROLLOUT_SOURCE=model2`, `ROLLOUT_SOURCE=model2-only`, `ACTOR_TRAINING_MODEL=joint`, `LOSS_MODE=wdl_sft`, `WDL_SFT_BETA=0.0`, `TOTAL_TRAINING_STEPS=75`, and the Stage 2 shard path.
- Stage 1 best checkpoints merged to HF Model2 dirs and load-checked in `verl-harness`:
  - Beta `0.1` Stage 1 best step `150`: `/data-1/model_weights/staged_v1/ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA01-V1_1779981295/step_150`
  - Beta `0.0` Stage 1 best step `85`: `/data-1/model_weights/staged_v1/ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA0-V1_1779962803/step_85`
  - Both load checks reported `model_type=qwen3`, `architectures=["Qwen3ForCausalLM"]`, `hidden_size=2560`, `num_hidden_layers=36`, `vocab_size=151936`, `tokenizer_class=Qwen2TokenizerFast`, and `has_model_index=true`.
  - Merge command path, used only when the merged HF dir is absent:
    `CUDA_VISIBLE_DEVICES=${MERGE_CUDA_VISIBLE_DEVICES:-0} python3 -u -m verl.model_merger merge --backend fsdp --local_dir "$FSDP_ACTOR_DIR" --target_dir "$MERGED_MODEL2_DIR" --trust-remote-code`.
  - Beta `0.0` source actor checkpoint:
    `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA0-V1_1779962803/global_step_85/actor`
  - Beta `0.1` source actor checkpoint:
    `/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA01-V1_1779981295/global_step_150/actor`
  - Beta `0.0` merged model inventory:
    `added_tokens.json:707; chat_template.jinja:4116; config.json:1536; generation_config.json:117; merges.txt:1671853; model-00001-of-00002.safetensors:4985530640; model-00002-of-00002.safetensors:3837363872; model.safetensors.index.json:32913; special_tokens_map.json:616; tokenizer.json:11422654; tokenizer_config.json:5407; vocab.json:2776833`.
  - Beta `0.0` selected digests:
    `model.safetensors.index.json=c4626a55ecad131134d454c83023c51dabd01a50fd370ef18019827a0c575c61`,
    `config.json=b8a48734f0776473bd8650f7d6f836543ff88f9898410d47defa419f8587e655`,
    `tokenizer_config.json=67e5a0a11cd35f9c00ee52e0af4cdc0baa75fea0cb5fce7d1beb251b4621d15c`.
  - Beta `0.1` merged model inventory:
    `added_tokens.json:707; chat_template.jinja:4116; config.json:1536; generation_config.json:117; merges.txt:1671853; model-00001-of-00002.safetensors:4954110272; model-00002-of-00002.safetensors:3868784240; model.safetensors.index.json:32913; special_tokens_map.json:616; tokenizer.json:11422654; tokenizer_config.json:5407; vocab.json:2776833`.
  - Beta `0.1` selected digests:
    `model.safetensors.index.json=fb3738cbcf442a90b8743887d13dd8bca30c608235bc9548c90a031eda956407`,
    `config.json=b8a48734f0776473bd8650f7d6f836543ff88f9898410d47defa419f8587e655`,
    `tokenizer_config.json=67e5a0a11cd35f9c00ee52e0af4cdc0baa75fea0cb5fce7d1beb251b4621d15c`.
- No existing incomplete Stage 2 checkpoint directory was found for either default run prefix before queue launch.
- First queue launch exposed a config dataclass issue:
  - Error: `omegaconf.errors.ConfigKeyError: Key 'joint_training_rollout_source' not in 'HFModelConfig'`
  - Fix: added `joint_training_rollout_source: str = "joint"` to `verl/workers/config/model.py`.
  - Verification: Docker py_compile passed after the fix.
- Queue relaunched with `ALLOW_RESUME=1` for the 0-step first run directory.
- WxPusher queue-start notification sent successfully:
  - title: `Stage2 WDL-SFT queue started`
  - WxPusher response: `success=true`
- First run runtime evidence:
  - Checkpoint dir: `/data-1/checkpoints/WDL-SFT-STAGED-V1-S2-FROM-S1-BETA01-BETA0_1780048439`
  - Run tmux: `staged_v1_s2_from_s1_beta01_beta0`
  - Docker container: `sweet_mccarthy`
  - Logs confirm `Training from scratch`.
  - Logs confirm `[WDL-SFT VERIFY] rollout source: model2-only; reason=initial; actor_training_model=joint; sync_eval_only=True`.
  - Logs confirm `extracting model2-only weights (sub_model_index=1)`.
  - Logs confirm `vllm_joint.load_weights: is_joint=False, will set _use_model2_only=True`.
  - Logs confirm first post-update Model2-only sync after step 1.
  - Progress reached at least `Training Progress: 1/75`.
- First queue attempt reached checkpoint step `10` before failing:
  - Checkpoint dir: `/data-1/checkpoints/WDL-SFT-STAGED-V1-S2-FROM-S1-BETA01-BETA0_1780048439`
  - Latest checkpoint marker: `10`
  - Best checkpoint after attempt: step `5`, MATH-500 mean@3 `0.7580645161290323`
  - Failure: CUDA OOM in actor entropy diagnostic computation
    (`compute_entropy_from_logits`), while `entropy_coeff=0`.
  - Mitigation: Stage 2 now defaults `CALCULATE_ENTROPY=False` and
    `ROLLOUT_GPU_MEMORY_UTILIZATION=0.35`; this preserves the WDL-SFT loss
    semantics but removes the non-loss entropy diagnostic from required runtime
    metrics.
- Second attempt relaunched `WDL-SFT-STAGED-V1-S2-FROM-S1-BETA01-BETA0`
  after OOM mitigation and reached step `35`, but was stopped manually on
  2026-05-30 after the user clarified the experiment definition:
  - This run used Stage 1 beta `0.1` as Model2 but Stage 2
    `WDL_SFT_BETA=0.0`.
  - It is a cross-beta diagnostic attempt, not a matched-beta acceptance run.
  - It does not count toward runtime PASS for this plan.
  - It confirmed the Model2-only rollout path and fused joint actor path still
    ran after OOM mitigation.
  - Online validation degraded sharply by step `35`; because the beta pairing
    was not the intended experiment, this is recorded only as a warning signal.
- User clarification on 2026-05-30:
  - Default acceptance runs must be beta-matched:
    `s2-from-s1-beta0-beta0` and `s2-from-s1-beta01-beta01`.
  - `s2-from-s1-beta01-beta0` and `s2-from-s1-beta0-beta01` are cross-beta
    diagnostics and require explicit authorization.
  - Actor mini-batching must not split one rollout batch into multiple
    optimizer steps. For `TRAIN_PROMPT_BSZ=64` and `ROLLOUT_N=8`, the required
    actor mini-batch is `512` response samples with `ppo_epochs=1`.
- Script revision after clarification:
  - `recipe/on_policy_wdl_sft/staged_v1/run_stage2_fast_validation_queue.sh`
    now runs matched defaults in order:
    `WDL-SFT-STAGED-V1-S2-FROM-S1-BETA0-BETA0`, then
    `WDL-SFT-STAGED-V1-S2-FROM-S1-BETA01-BETA01`.
  - `recipe/on_policy_wdl_sft/staged_v1/monitor_stage2_fast_validation_queue_notify.sh`
    watches the same two matched default prefixes.
  - `recipe/on_policy_wdl_sft/staged_v1/_run_stage2_model2_rollout_common.sh`
    now defaults to `TRAIN_PROMPT_BSZ=64`, `ROLLOUT_N=8`,
    `TRAIN_PROMPT_MINI_BSZ=512`, `ACTOR_PPO_EPOCHS=1`, and
    `ACTOR_SHUFFLE=false`; dry-run output also prints the exact Hydra override
    names and values for these gates.
- Corrected pre-launch checks passed on 2026-05-30:
  - `bash -n recipe/on_policy_wdl_sft/_common_wdl_sft_is_joint.sh recipe/on_policy_wdl_sft/staged_v1/*.sh`
  - `bash -n recipe/on_policy_wdl_sft/_common_wdl_sft_is_joint.sh recipe/on_policy_wdl_sft/staged_v1/*.sh recipe/on_policy_wdl_sft/staged_v1/meituan/*.sh platform/hope_staged_v1/*.sh`
  - `STAGE2_DRY_RUN=1 bash recipe/on_policy_wdl_sft/staged_v1/run_s2_from_s1_beta0_beta0.sh`
    rendered `WDL_SFT_BETA=0.0`, pinned Stage 1 beta `0.0` step `85`,
    `JOINT_TRAINING_ROLLOUT_SOURCE=model2`, `ROLLOUT_SOURCE=model2-only`,
    `ACTOR_TRAINING_MODEL=joint`, `TRAIN_PROMPT_BSZ=64`, `ROLLOUT_N=8`,
    `TRAIN_PROMPT_MINI_BSZ=512`, `ACTOR_PPO_EPOCHS=1`,
    `ACTOR_SHUFFLE=false`, `ROLLOUT_IS=null`, `ROLLOUT_RS=null`, and
    the Stage 2 non-overlap shard path.
  - `STAGE2_DRY_RUN=1 bash recipe/on_policy_wdl_sft/staged_v1/run_s2_from_s1_beta01_beta01.sh`
    rendered `WDL_SFT_BETA=0.1`, pinned Stage 1 beta `0.1` step `150`,
    and the same Model2-only rollout plus single-mini-batch gates.
  - Docker compile passed:
    `docker run --rm --gpus all --ipc=host --shm-size=64g -v /data-1:/data-1 -v /data-1/verl07/verl:/workspace/verl -w /workspace/verl verl-harness python3 -m py_compile verl/workers/config/model.py verl/trainer/ppo/ray_trainer.py`
  - No Stage 2 tmux session or Docker training container was running.
  - No existing checkpoint directory was found for the corrected default
    prefixes `WDL-SFT-STAGED-V1-S2-FROM-S1-BETA0-BETA0_*` or
    `WDL-SFT-STAGED-V1-S2-FROM-S1-BETA01-BETA01_*`.
  - `/data-1/verl07/run_train.sh` is not present on this host; the queue script
    will use its fallback `docker run ... verl-harness` launch path unless
    `LAUNCHER` is explicitly provided.

## Final Runtime Evidence

- The corrected default queue ran the matched-beta candidates sequentially:
  - First: `WDL-SFT-STAGED-V1-S2-FROM-S1-BETA0-BETA0_1780073162`
  - Second: `WDL-SFT-STAGED-V1-S2-FROM-S1-BETA01-BETA01_1780096269`
- Queue script:
  `recipe/on_policy_wdl_sft/staged_v1/run_stage2_fast_validation_queue.sh`
- Queue log:
  `recipe/on_policy_wdl_sft/staged_v1/run_stage2_fast_validation_queue.log`
- Monitor log:
  `recipe/on_policy_wdl_sft/staged_v1/monitor_stage2_fast_validation_queue_notify.log`
- Post-run state verified at `2026-05-30 16:24:05 CST`:
  - No Stage 2 tmux sessions remain.
  - No `verl-harness` Docker container remains.
  - All 8 GPUs are idle, with only `1 MiB` memory shown per GPU.
  - `/data-1` free space is about `143G` (`95%` used).

### Runtime Results

| Run | Stage 1 source | Stage 2 beta | latest marker | best step | Stage 1 baseline MATH mean@3 | Stage 2 best MATH mean@3 | Delta | Final step 75 MATH mean@3 | Final AIME mean@3 | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `s2-from-s1-beta0-beta0` | beta `0.0`, step `85` | `0.0` | `75` | `35` | `0.7325268817204301` | `0.7479838709677419` | `+1.55 pp` | `0.014112903225806451` | `0.0` | Runtime PASS, method collapse after best |
| `s2-from-s1-beta01-beta01` | beta `0.1`, step `150` | `0.1` | `75` | `20` | `0.7573924731182795` | `0.7668010752688172` | `+0.94 pp` | `0.013440860215053762` | `0.0` | Runtime PASS, method collapse after best |

`s2-from-s1-beta0-beta0` evidence:

- Checkpoint dir:
  `/data-1/checkpoints/WDL-SFT-STAGED-V1-S2-FROM-S1-BETA0-BETA0_1780073162`
- `latest_checkpointed_iteration.txt`: `75`
- `best_checkpoint.json`:
  - metric key: `val-core/HuggingFaceH4/MATH-500/acc/mean@3`
  - best step: `35`
  - metric value: `0.7479838709677419`
- Metrics JSONL:
  `recipe/on_policy_wdl_sft/staged_v1/metrics/OnPolicySFT-Then-WDLSFT-StagedV1/WDL-SFT-STAGED-V1-S2-FROM-S1-BETA0-BETA0_1780073162.jsonl`
  - lines: `75`
  - final `training/global_step`: `75`
  - final `actor/wdl_sft_beta`: `0.0`
  - final `actor/wdl_sft_loss_positive`: `26.37750026157924`
  - final `actor/wdl_sft_loss_negative`: `-24.799610319591704`
  - final `actor/wdl_sft_loss_total`: `26.37750026157924`
  - final `wdl_sft/correct_ratio`: `0.01171875`
  - final `actor/grad_norm`: `427.344970703125`
  - final `response/aborted_ratio`: `0.0`
  - final `response_length/clip_ratio`: `0.435546875`
  - final `jointTraining/answer_extraction_failure_rate`: `0.5574712643678161`
- Logs:
  `recipe/on_policy_wdl_sft/staged_v1/WDL-SFT-STAGED-V1-S2-FROM-S1-BETA0-BETA0_1780073162.log`
  - contains `Training Progress: 100%|...| 75/75`
  - contains `rollout source: model2-only`
  - contains `extracting model2-only weights (sub_model_index=1)`
  - contains `vllm_joint.load_weights: is_joint=False, will set _use_model2_only=True`
  - contains `actor_training_model=joint`
  - contains final checkpoint path `global_step_75`
  - W&B offline path:
    `/data-1/wandb_runs/WDL-SFT-STAGED-V1-S2-FROM-S1-BETA0-BETA0/wandb/offline-run-20260529_164940-j2qmgtqi`
- Required submodel gradient evidence:
  - `jointTraining/model1_grad_norm` and `jointTraining/model2_grad_norm` were
    both positive on `74` metric records.
  - Final values were `18.19676936459008` and `423.8404249827868`.

`s2-from-s1-beta01-beta01` evidence:

- Checkpoint dir:
  `/data-1/checkpoints/WDL-SFT-STAGED-V1-S2-FROM-S1-BETA01-BETA01_1780096269`
- `latest_checkpointed_iteration.txt`: `75`
- `best_checkpoint.json`:
  - metric key: `val-core/HuggingFaceH4/MATH-500/acc/mean@3`
  - best step: `20`
  - metric value: `0.7668010752688172`
- Metrics JSONL:
  `recipe/on_policy_wdl_sft/staged_v1/metrics/OnPolicySFT-Then-WDLSFT-StagedV1/WDL-SFT-STAGED-V1-S2-FROM-S1-BETA01-BETA01_1780096269.jsonl`
  - lines: `75`
  - final `training/global_step`: `75`
  - final `actor/wdl_sft_beta`: `0.1`
  - final `actor/wdl_sft_loss_positive`: `10.921120673418045`
  - final `actor/wdl_sft_loss_negative`: `-6.155470609664917`
  - final `actor/wdl_sft_loss_total`: `10.305573403835297`
  - final `wdl_sft/correct_ratio`: `0.00390625`
  - final `actor/grad_norm`: `71.9848403930664`
  - final `response/aborted_ratio`: `0.0`
  - final `response_length/clip_ratio`: `0.833984375`
  - final `jointTraining/answer_extraction_failure_rate`: `0.9323116219667944`
- Logs:
  `recipe/on_policy_wdl_sft/staged_v1/WDL-SFT-STAGED-V1-S2-FROM-S1-BETA01-BETA01_1780096269.log`
  - contains `Training Progress: 100%|...| 75/75`
  - contains `rollout source: model2-only`
  - contains `extracting model2-only weights (sub_model_index=1)`
  - contains `vllm_joint.load_weights: is_joint=False, will set _use_model2_only=True`
  - contains `actor_training_model=joint`
  - contains final checkpoint path `global_step_75`
  - W&B offline path:
    `/data-1/wandb_runs/WDL-SFT-STAGED-V1-S2-FROM-S1-BETA01-BETA01/wandb/offline-run-20260529_231457-0ywgguty`
- Required submodel gradient evidence:
  - `jointTraining/model1_grad_norm` and `jointTraining/model2_grad_norm` were
    both positive on `61` metric records.
  - Final values were `3.6810674003131427` and `71.52325973347772`.

### Algorithm and Config Evidence

- Dry-run logs for both default wrappers render:
  - `JOINT_TRAINING_ROLLOUT_SOURCE=model2`
  - `ROLLOUT_SOURCE=model2-only`
  - `ACTOR_TRAINING_MODEL=joint`
  - `ROLLOUT_IS=null`
  - `ROLLOUT_RS=null`
  - `TRAIN_PROMPT_BSZ=64`
  - `ROLLOUT_N=8`
  - `TRAIN_PROMPT_MINI_BSZ=512`
  - `ACTOR_PPO_EPOCHS=1`
  - `ACTOR_SHUFFLE=false`
- Queue run logs show the exact Hydra launch overrides:
  - `algorithm.use_kl_in_reward=False`
  - `algorithm.rollout_correction.rollout_is=null`
  - `algorithm.rollout_correction.rollout_rs=null`
  - `actor_rollout_ref.actor.use_kl_loss=False`
  - `actor_rollout_ref.actor.policy_loss.loss_mode=wdl_sft`
  - `actor_rollout_ref.actor.ppo_mini_batch_size=512`
  - `actor_rollout_ref.actor.ppo_epochs=1`
  - `actor_rollout_ref.actor.shuffle=false`
  - `actor_rollout_ref.rollout.n=8`
  - `data.train_batch_size=64`
  - `data.shuffle=False`
  - `data.train_files=/data-1/dataset/EnsembleLLM-data-processed/staged_v1/stage2_after_s1_150steps_seed20260528_75steps.parquet`
  - `+actor_rollout_ref.model.joint_training=True`
  - `+actor_rollout_ref.model.joint_training_rollout_source=model2`
- Matched beta evidence:
  - beta0 run logs render `WDL_SFT_BETA=0.0` and
    `+actor_rollout_ref.actor.policy_loss.wdl_sft_beta=0.0`.
  - beta0.1 run logs render `WDL_SFT_BETA=0.1` and
    `+actor_rollout_ref.actor.policy_loss.wdl_sft_beta=0.1`.
- Prepared joint model configs:
  - `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-step_85/config.json`
  - `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-step_150/config.json`
  - Both show `model_type=qwen_joint`, `architectures=["QwenJointForCausalLM"]`,
    `fusion_lambda=0.5`, and `freeze_model1=false`. No `freeze_model2` field is
    present, so Model2 is not configured as frozen.

### Data Verification

- Verified in Docker image `verl-harness` with:

```bash
docker run --rm --ipc=host --shm-size=64g \
  -v /data-1:/data-1 \
  -v /data-1/verl07/verl:/workspace/verl \
  -w /workspace/verl verl-harness \
  python3 recipe/on_policy_wdl_sft/staged_v1/create_stage2_nonoverlap_shard.py \
    --verify-only \
    --output /data-1/dataset/EnsembleLLM-data-processed/staged_v1/stage2_after_s1_150steps_seed20260528_75steps.parquet
```

- Verifier result:
  - `status=PASS`
  - `row_count=4800`
  - `eligible_row_count=104673`
  - `overlap_count=0`
  - `sha256=28cdd1d9f0c5c06d7eab768b264ad175830f72d8c07ad66652aa43de862435eb`
  - first 10 `extra_info.index`:
    `109719, 111365, 88991, 58015, 24690, 37418, 66497, 110437, 83593, 15772`
  - last 10 `extra_info.index`:
    `32774, 41205, 35439, 10950, 110171, 62866, 66622, 62003, 100706, 11795`

### Notifications and Queue Caveat

- Queue and monitor sent WxPusher notifications with `success=true` for:
  - queue start;
  - individual launch start/confirmation;
  - per-run completion;
  - queue completion.
- Caveat: the original queue/monitor completion condition used
  `latest_checkpointed_iteration.txt >= 75`, so the beta0.1 run-complete and
  queue-complete messages were sent before final validation metrics flushed.
  The run itself continued until about `2026-05-30 16:23 CST`, then metrics line
  `75` landed and the tmux/container exited.
- Mitigation applied after observing the caveat:
  - `recipe/on_policy_wdl_sft/staged_v1/run_stage2_fast_validation_queue.sh`
    now requires both final checkpoint marker and final metrics JSONL fields
    before sending future completion notifications.
  - `recipe/on_policy_wdl_sft/staged_v1/monitor_stage2_fast_validation_queue_notify.sh`
    has the same final-metrics gate.
  - `bash -n` passed for both revised scripts.
- Additional WxPusher messages were sent after true completion was verified:
  - `Stage2 WDL-SFT verified complete`
  - `Stage2 beta0.1 run verified complete`
  - Both returned `success=true`.

### Runtime Warnings

- Both completed logs contain W&B `BrokenPipeError` tracebacks after:
  - final checkpoint save;
  - final validation metrics;
  - W&B run summary and sync path.
- The beta0.1 log also contains a vLLM multiprocessing resource-tracker warning
  during shutdown.
- These are classified as cleanup-time warnings, not runtime failures, because
  final metrics and checkpoint completion are verified and no training process
  remains.
- Method-level warning: both runs improve at their best checkpoint but collapse
  badly by step 75. The default interpretation should use `best_checkpoint.json`
  for the plan's best-vs-best comparison while explicitly reporting final-step
  collapse.

## Notes

- The rollout engine may still use a prepared joint model architecture shell for vLLM compatibility, but `joint_training_rollout_source=model2` must make weight sync extract and load only `sub_models.1` into rollout, setting the rollout to Model2-only mode.
- Runtime PASS is satisfied locally for the two matched default runs. Meituan/AFO
  remains `NOT ACCEPTED`.
