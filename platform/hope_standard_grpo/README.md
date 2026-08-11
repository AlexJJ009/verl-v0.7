# Qwen3-1.7B Standard GRPO AFO entry

This directory is a credential-free template. Copy it to a submission-specific
`hope_dir`, replace every `REPLACE_*` value in `run.hope`, and submit that copy.
Do not commit real account, queue, image, or dolphinfs endpoints.

Supported `EXPERIMENT` values:

- `math-stage1-grpo`
- `math-cold-start-grpo`
- `code-stage1-grpo`
- `code-cold-start-grpo`

The immutable input manifest is produced during staging and must bind the Git
commit, recipe gitlink, image identity, model directory, dataset/receipt, and
evaluator dependencies. AFO workers force `WANDB_MODE=offline`; syncing is a
separate release-gated operation.
