# Math WDL causal-P60 AFO entry

Set `EXPERIMENT` to required `arm-c` or `arm-d0`, or to the fixed-Model1
controls `fixed-m1-cold-start` / `fixed-m1-stage1`. The direct-Model2 `arm-d`
is optional after the equivalence probe and additionally requires
`RUN_OPTIONAL_D=1`. The worker entry reads the
dolphinfs-mounted repository, applies path overrides from
`recipe/on_policy_wdl_sft/math_task/meituan/env.sh`, and fails before training
when the selected Cold Start / Stage1 Model2 source, provenance, train shard,
or Math-7 validation files are absent.
