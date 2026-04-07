# Commit Conventions

## Commit Message Format

Use conventional commits with a scope:

```
<type>(<scope>): <description>
```

Common types: `feat`, `fix`, `docs`, `test`, `chore`, `refactor`
Common scopes: `joint_training`, `trainer`, `checkpoint`, `vllm_rollout`, `recipe`

Examples:
```
feat(joint_training): add per-submodel gradient norm metrics
fix(checkpoint): harden fsdp shard saves against disk pressure
docs(tests): refresh joint training recipe coverage
```

## Git Submodule Workflow

The `recipe/` directory is a git submodule. Always push `recipe/` first, then update the parent repo pointer.

```bash
cd recipe
git add <files>
git commit -m "..."
git push origin feature/on-policy-wdl-sft

cd ..
git add recipe
git commit -m "chore(recipe): update submodule pointer"
git push origin feature/on-policy-wdl-sft
```

After updating the parent repo on another server, refresh the submodule state as well.

## Branching

- Active branch: `feature/on-policy-wdl-sft`
- Parent branch: `feature/joint-training` (Stage 1 & 2 complete)
- This is a personal fork. No pull requests are needed — push directly to the branch.
- Always commit to `feature/on-policy-wdl-sft`, not `main`.
