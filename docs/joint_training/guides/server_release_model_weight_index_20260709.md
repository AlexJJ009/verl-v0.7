# Server Release Model Weight Index - 2026-07-09

Purpose: preserve the migration decision index before releasing the server. This file does not contain secrets. Full TSV is committed at `docs/joint_training/reports/data/server_release_model_weight_index_20260709.tsv`; a local copy also exists at `/data-1/model_weights/manifests/server_release_model_weight_index_20260709.tsv`.

## HF Connectivity Smoke

- Public smoke repo: `AlexGeek/verl-public-upload-smoke-20260709`.
- Direct/no-proxy: failed. `curl https://huggingface.co` timed out; direct `upload_file` failed with `Network is unreachable` at HF preupload.
- Proxy: succeeded. `proxy.txt` and `proxy_lfs_16m.bin` uploaded; 16MiB LFS commit `afc55cba262b2928e3e8ed7a0a229db54ea74674`.
- Operational conclusion: public uploads still require proxy on this server unless network routing changes. Use `HF_HUB_DISABLE_XET=1`; direct is not currently viable.

## Priority Semantics

- `P1_PUBLIC_UPLOAD`: worth uploading as public copy, likely local-only or newly generated.
- `P1_PUBLIC_COPY_VERIFY`: important result already has upload proof; verify remote and create public copy if existing repo is private.
- `P2_PUBLIC_COPY_VERIFY`: useful historical/math provenance; upload after P1 if traffic allows.
- `P2_VERIFY_ONLY`: historical artifact already uploaded/verified; public copy optional.
- `P3_SKIP_OR_LOW`: smoke, upstream public base model, or intermediate; skip unless explicitly needed.

## Summary

| Priority | Count | Size GiB |
|---|---:|---:|
| `P1_PUBLIC_UPLOAD` | 67 | 743.56 |
| `P1_PUBLIC_COPY_VERIFY` | 12 | 125.70 |
| `P2_PUBLIC_COPY_VERIFY` | 21 | 283.82 |
| `P2_VERIFY_ONLY` | 23 | 243.10 |
| `P2_REVIEW` | 9 | 32.92 |
| `P3_SKIP_OR_LOW` | 3 | 49.35 |

## Top Candidates

| Priority | Size GiB | Artifact | Local path | Existing HF repo |
|---|---:|---|---|---|
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P60-BETA0-BETA0-V1-ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA0-V1-step_60` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-CODE-S2-KODCODE-INSTRUCT2507-CTX8K-P60-BETA0-BETA0-V1-ONPOLICY-SFT-Qwen3-4B-INSTRUCT2507-CODE-KODCODE-CTX8K-S1-BETA0-V1-step_60` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-CODE-S2-RETENTION-BETA0-BETA0-ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V2-RETENTION-R2-step_70` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-CODE-S2-RETENTION-BETA0-BETA0-ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA0-V2-RETENTION-R2-step_70` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-CODE-S2-RETENTION-BETA01-BETA01-ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V2-RETENTION-R2-step_70` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-CODE-S2-RETENTION-BETA01-BETA01-ONPOLICY-SFT-Qwen3-4B-CODE-KODCODE-S1-BETA01-V2-RETENTION-R2-step_70` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-kod_i2507_p40_b01_s2` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-kod_i2507_p40_b01_s2` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-kod_i2507_p40_b01_s2_lambda06` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-kod_i2507_p40_b01_s2_lambda06` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-kod_i2507_p40_b01_s2_lambda07` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-kod_i2507_p40_b01_s2_lambda07` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-kod_i2507_p40_b01_s2_lambda08` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-kod_i2507_p40_b01_s2_lambda08` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-kod_i2507_p40_b01_s2_lambda09` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-kod_i2507_p40_b01_s2_lambda09` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-kod_i2507_p40_b01_subkl_both_on_lambda08` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-kod_i2507_p40_b01_subkl_both_on_lambda08` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-kod_i2507_p40_b01_subkl_model1_lambda08` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-kod_i2507_p40_b01_subkl_model1_lambda08` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-kod_i2507_p40_b01_subkl_model2_lambda08` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-kod_i2507_p40_b01_subkl_model2_lambda08` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-kod_i2507_p40_b0_s2` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-kod_i2507_p40_b0_s2` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-kod_i2507_p60_b01_s2` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-kod_i2507_p60_b01_s2` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-kod_i2507_p60_b0_s2` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-kod_i2507_p60_b0_s2` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-model2-from-s1-p60-beta0-step60` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-model2-from-s1-p60-beta0-step60` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-model2-from-s1-p60-beta01-step60` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-model2-from-s1-p60-beta01-step60` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-subkl_both_off_p40_b01` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-subkl_both_off_p40_b01` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-subkl_both_on_kl_p40_b01` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-subkl_both_on_kl_p40_b01` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-subkl_both_on_low_var_kl_p40_b01` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-subkl_both_on_low_var_kl_p40_b01` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-subkl_both_on_mse_p40_b01` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-subkl_both_on_mse_p40_b01` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-subkl_model1_kl_p40_b01` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-subkl_model1_kl_p40_b01` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-subkl_model1_low_var_kl_p40_b01` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-subkl_model1_low_var_kl_p40_b01` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-subkl_model1_mse_p40_b01` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-subkl_model1_mse_p40_b01` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-subkl_model1_p40_b01` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-subkl_model1_p40_b01` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-subkl_model2_kl_p40_b01` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-subkl_model2_kl_p40_b01` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-subkl_model2_low_var_kl_p40_b01` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-subkl_model2_low_var_kl_p40_b01` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `QwenJoint-4B-Stage2-subkl_model2_mse_p40_b01` | `/data-1/.cache/huggingface/QwenJoint-4B-Stage2-subkl_model2_mse_p40_b01` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `s2-p60-beta0-step20-best-joint` | `/data-1/model_weights/staged_v1/plateau_handoff_p60/s2-p60-beta0-step20-best-joint` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `s2-p60-beta0-step40-final-joint` | `/data-1/model_weights/staged_v1/plateau_handoff_p60/s2-p60-beta0-step40-final-joint` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `s2-p60-beta01-step35-best-joint` | `/data-1/model_weights/staged_v1/plateau_handoff_p60/s2-p60-beta01-step35-best-joint` | `` |
| `P1_PUBLIC_UPLOAD` | 16.45 | `s2-p60-beta01-step40-final-joint` | `/data-1/model_weights/staged_v1/plateau_handoff_p60/s2-p60-beta01-step40-final-joint` | `` |
| `P1_PUBLIC_UPLOAD` | 8.23 | `actor_step115` | `/data-1/model_weights/code_task/offline_eval_lcb_v5_fair/kodcode_beta01_best_step115/actor_step115` | `` |
| `P1_PUBLIC_UPLOAD` | 8.23 | `actor_step115` | `/data-1/model_weights/code_task/offline_eval/kodcode_i2507_ctx8k_beta01_step115/actor_step115` | `` |
| `P1_PUBLIC_UPLOAD` | 8.23 | `actor_step115` | `/data-1/model_weights/code_task/offline_eval/deepcoder_i2507_r8k_beta01_step115/actor_step115` | `` |
| `P1_PUBLIC_UPLOAD` | 8.23 | `actor_step120` | `/data-1/model_weights/code_task/offline_eval/deepcoder_i2507_r8k_beta0_step120/actor_step120` | `` |
| `P1_PUBLIC_UPLOAD` | 8.23 | `actor_step145` | `/data-1/model_weights/code_task/offline_eval/kodcode_i2507_ctx8k_beta0_step145/actor_step145` | `` |
| `P1_PUBLIC_UPLOAD` | 8.23 | `actor_step150` | `/data-1/model_weights/code_task/offline_eval_lcb_v5_fair/kodcode_beta01_latest_step150/actor_step150` | `` |
| `P1_PUBLIC_UPLOAD` | 8.23 | `actor_step150` | `/data-1/model_weights/code_task/offline_eval/deepcoder_i2507_r8k_beta0_step150/actor_step150` | `` |
| `P1_PUBLIC_UPLOAD` | 8.23 | `actor_step150` | `/data-1/model_weights/code_task/offline_eval/kodcode_i2507_ctx8k_beta01_step150/actor_step150` | `` |
| `P1_PUBLIC_UPLOAD` | 8.23 | `actor_step150` | `/data-1/model_weights/code_task/offline_eval/kodcode_i2507_ctx8k_beta0_step150/actor_step150` | `` |
