---
name: test
description: >
  Run joint-training tests. Usage: /test [target]
  Examples: /test, /test reward, /test regression, /test all
---

# test

Run joint-training tests in the verl project.

## Usage

- `/test` — run all joint-training tests
- `/test <subdir>` — run tests in `tests/joint_training/<subdir>/` (e.g., `reward`, `regression`, `feat`)
- `/test all` — run joint-training tests plus related framework tests
- `/test <file>` — run a specific test file path

## Execution

```bash
conda activate verl07
```

Then based on the argument:

- No argument or empty: `pytest tests/joint_training/ -v`
- A subdirectory name (reward, regression, feat): `pytest tests/joint_training/<subdir>/ -v`
- `all`: `pytest tests/joint_training/ tests/workers/actor/test_special_dp_actor.py -v`
- A file path: `pytest <file> -v`

Report the result summary (passed/failed/errors) to the user. If any test fails, show the failure output.
