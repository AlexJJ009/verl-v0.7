# Milestone 3 Independent Review

- Reviewer: fresh independent Codex subagent `Wegener`
- Reviewed recipe commit: `e7d559c`
- Reviewed superproject commit: `89756ca4`
- Original verdict: `REJECT`

## Blocking Findings

1. The container path lacked `jq`, while the queue still depended on it.
2. An inactive historical checkpoint could trigger `run_started`.
3. Docker/GPU ownership attribution was insufficient to prove resource release.
4. A first training step incorrectly satisfied the complete-validation deadline.

## Replacement Evidence

The replacement implementation removes `jq` from Stage123 queue/monitor paths,
requires active run evidence for `run_started`, requires complete validation metrics
to satisfy the validation deadline, assigns a stable Docker container name, resolves
the container init PID through `docker inspect`, records container descendants and
their intersecting GPU PIDs, and refuses `resources_released=true` without proven
container ownership.

The next independent review must inspect the replacement commits and rerun the fast,
full, deadline, notification, and end-to-end gates. This document records the rejected
review; it is not an acceptance statement.
