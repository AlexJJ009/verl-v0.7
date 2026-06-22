# Code Task Meituan/AFO Entry

Supported `EXPERIMENT` values:

- `s1-code-smoke-beta-0`
- `s1-code-pilot-beta-0`
- `s1-code-onpolicy-sft-beta-0`
- `s1-code-onpolicy-sft-beta-01`
- `s1-code-onpolicy-sft-beta-0-retention`
- `s1-code-onpolicy-sft-beta-01-retention`
- `s1-code-deepcoder-beta-0-retention`
- `s1-code-deepcoder-beta-01-retention`
- `s2-code-smoke-beta0-beta0`
- `s2-code-pilot-beta0-beta0`
- `s2-code-retention-beta0-beta0`
- `s2-code-retention-beta01-beta01`

Dry-run dispatch:

```bash
DRY_RUN=1 EXPERIMENT=s1-code-smoke-beta-0 bash platform/hope_code_task/jupyter.sh
```

Pilot runs require `SANDBOX_FUSION_URL` and explicit user approval. DolphinFS
paths live only in `recipe/on_policy_wdl_sft/code_task/meituan/env.sh`; per-run
wrappers remain default-local and environment-overridable.
