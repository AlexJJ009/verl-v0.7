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
git push origin feature/joint-training

cd ..
git add recipe
git commit -m "chore(recipe): update submodule pointer"
git push origin feature/joint-training
```

After updating the parent repo on another server, refresh the submodule state as well.

## Branching

- Active branch: `feature/joint-training`
- This is a personal fork. No pull requests are needed — push directly to the branch.
- Always commit to `feature/joint-training`, not `main`.
