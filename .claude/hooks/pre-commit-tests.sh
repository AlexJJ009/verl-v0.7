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

# Activate conda and run tests
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate verl07 2>/dev/null

cd /data-1/verl07/verl
TEST_OUTPUT=$(pytest tests/joint_training/ -q --tb=short 2>&1)
TEST_EXIT=$?

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
