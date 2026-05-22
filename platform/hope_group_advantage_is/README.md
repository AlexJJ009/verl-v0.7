# hope_group_advantage_is

Template `hope_dir` for Meituan AFO WDL group-advantage IS runs.

```bash
cp -r /data-1/verl07/verl/platform/hope_group_advantage_is ~/hope_group_adv_is
cd ~/hope_group_adv_is
# edit run.hope:
#   afo.app.name
#   afo.docker.image.name
#   afo.app.env.EXPERIMENT = 1a-group-adv-is
hope submit run.hope
```

Smoke mode:

```text
afo.app.env.SMOKE = 1
```

Layer flow:

```text
platform/hope_group_advantage_is/jupyter.sh
-> recipe/on_policy_wdl_sft/group_advantage_is/meituan/jupyter.sh
-> recipe/on_policy_wdl_sft/group_advantage_is/run_1a_group_adv_is.sh
-> recipe/on_policy_wdl_sft/group_advantage_is/_common_group_adv_is.sh
```

Defaults: `ROLLOUT_IS=null`, KL disabled, `LOSS_AGG_MODE=seq-mean-token-sum`,
`NORM_ADV_BY_STD_IN_GRPO=false`, all-correct fallback coefficient `1.0`.
