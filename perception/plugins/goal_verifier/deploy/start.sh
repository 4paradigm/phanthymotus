#!/usr/bin/env bash
set -euo pipefail
: "${GOAL_CONFIG:?Set absolute path of your own validated source configuration}"
TASK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
export PYTHONPATH="$TASK_ROOT/perception"
export PYTHONDONTWRITEBYTECODE=1
exec "${GOAL_PYTHON:-python3}" -m plugins.goal_verifier.server \
  --config "$GOAL_CONFIG" --host 127.0.0.1 --port "${GOAL_PORT:-15785}"
