# hope_on_policy_wdl_sft

Unified Meituan hope_dir template for On-Policy WDL-SFT reruns.

Use one template for both joint 1X and single-model 2X runs. Copy this
directory, fill the docker image and `EXPERIMENT`, then submit.

```bash
cp -r /data-1/verl07/verl/platform/hope_on_policy_wdl_sft ~/hope_2a_base
# edit ~/hope_2a_base/run.hope:
#   afo.app.name           = verl-2a-base
#   afo.docker.image.name  = <your verl v2 image URL>
#   afo.app.env.EXPERIMENT = 2a-base
cd ~/hope_2a_base && hope submit run.hope
```

## Experiment Names

| EXPERIMENT | Family | Init / model | Loss | β | lr | Rerun? | Notes |
|---|---|---|---|---|---|---|---|
| `1a` | joint | Base + SFT-stage-1 | `wdl_sft_is` | 0.0 | 5e-7 | yes | Writes `WDL-SFT-Qwen3-4B-MATH-1A-LABELFIX_*` |
| `1b` | joint | Base + SFT-stage-1 | `wdl_sft_is` | 0.1 | 5e-7 | yes | Writes `WDL-SFT-Qwen3-4B-MATH-1B-LABELFIX_*` |
| `1c` | joint | Base + SFT-stage-1 | `wdl_sft_is` | 0.0 | 1e-6 | yes | Writes `WDL-SFT-Qwen3-4B-MATH-1C-LABELFIX_*` |
| `2a-base` | single | Qwen3-4B-Base | `wdl_sft_is` | 0.0 | 5e-7 | yes | Writes `WDL-SFT-Qwen3-4B-MATH-2A-BASE-LABELFIX_*` |
| `2a-sft` | single | Qwen3-4B-Base-SFT-stage-1 | `wdl_sft_is` | 0.0 | 5e-7 | yes | Writes `WDL-SFT-Qwen3-4B-MATH-2A-SFT-LABELFIX_*` |
| `2b-base` | single | Qwen3-4B-Base | `wdl_sft_is` | 0.1 | 5e-7 | yes | Writes `WDL-SFT-Qwen3-4B-MATH-2B-BASE-LABELFIX_*` |
| `2b-sft` | single | Qwen3-4B-Base-SFT-stage-1 | `wdl_sft_is` | 0.1 | 5e-7 | yes | Writes `WDL-SFT-Qwen3-4B-MATH-2B-SFT-LABELFIX_*` |
| `2c-base` | single | Qwen3-4B-Base | `wdl_sft_is` | 0.0 | 1e-6 | yes | Writes `WDL-SFT-Qwen3-4B-MATH-2C-BASE-LABELFIX_*` |
| `2c-sft` | single | Qwen3-4B-Base-SFT-stage-1 | `wdl_sft_is` | 0.0 | 1e-6 | yes | Writes `WDL-SFT-Qwen3-4B-MATH-2C-SFT-LABELFIX_*` |
| `2z-base` | single | Qwen3-4B-Base | `minirl` | - | 5e-7 | no, optional | Baseline unaffected by label bug |
| `2z-sft` | single | Qwen3-4B-Base-SFT-stage-1 | `minirl` | - | 5e-7 | no, optional | Baseline unaffected by label bug |
| `2g-base` | single | Qwen3-4B-Base | `vanilla` | - | 5e-7 | no, optional | Baseline unaffected by label bug |
| `2g-sft` | single | Qwen3-4B-Base-SFT-stage-1 | `vanilla` | - | 5e-7 | no, optional | Baseline unaffected by label bug |

## Optional Smoke Mode

Add this to `[others]` for a 10-step sanity pass:

```ini
afo.app.env.SMOKE = 1
```

## Path Assumptions

Default repo path:

```text
$LGX/verl08/verl-v0.7-feature-on-policy-wdl-sft
```

Override with:

```ini
afo.app.env.REPO_SUBPATH = some/other/repo/path
```

Expected data/model locations are defined in:

- Joint 1X: `recipe/on_policy_wdl_sft/meituan/env.sh`
- Single 2X: `recipe/on_policy_wdl_sft/ablation_single_model/meituan/env.sh`
