#!/bin/bash
# Pre-commit hook: run joint-training tests before allowing git commit.
# Reads the tool input from stdin (JSON with tool_input.command).

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // ""' 2>/dev/null)

# Only intercept git commit commands
if ! echo "$COMMAND" | grep -qE '^git commit'; then
  exit 0
fi

echo "Pre-commit: running joint-training tests..." >&2

cd /data-1/verl07/verl

# This project keeps its Python environment inside the `verl-harness` Docker
# image. Do not run pytest with the host Python environment.
if ! docker image inspect verl-harness >/dev/null 2>&1; then
  echo "Docker image verl-harness not found; run: bash /data-1/verl07/build.sh" >&2
  TEST_EXIT=127
  TEST_OUTPUT=""
else
  TEST_OUTPUT=$(
    docker run --rm \
      --gpus "${GPUS:-all}" \
      --ipc=host \
      -v /data-1/verl07/verl:/workspace/verl \
      -v /data-1:/data-1 \
      verl-harness \
      bash -lc 'cd /workspace/verl && pytest tests/joint_training/ -q --tb=short' \
      2>&1
  )
  TEST_EXIT=$?
fi

if [ $TEST_EXIT -ne 0 ]; then
  echo "$TEST_OUTPUT" >&2
  echo "" >&2
  echo "Pre-commit: tests FAILED. Fix failures before committing." >&2
  jq -n '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: "Joint-training tests failed. Fix the failures before committing."
    }
  }'
  exit 0
fi

echo "Pre-commit: all tests passed." >&2
exit 0
