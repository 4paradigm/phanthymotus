#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
export PYTHONPATH="$TASK_ROOT/perception"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONDONTWRITEBYTECODE=1
exec "${GOAL_TEST_PYTHON:-python3}" -m pytest -q -p no:cacheprovider \
  "$TASK_ROOT/perception/tests/test_goal_verifier.py" \
  "$TASK_ROOT/perception/tests/test_goal_verifier_integration.py" \
  "$TASK_ROOT/perception/tests/test_goal_verifier_build.py" --tb=short "$@"
