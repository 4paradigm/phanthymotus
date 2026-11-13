#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
export PYTHONPATH="$ROOT/perception${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONDONTWRITEBYTECODE=1
exec "${ROI_TEST_PYTHON:-python3}" -m pytest -q -p no:cacheprovider \
  "$ROOT/perception/tests/test_tabletop_roi_goal_verifier.py" \
  "$ROOT/perception/tests/test_tabletop_roi_transport.py" \
  "$ROOT/perception/tests/test_tabletop_roi_field.py" "$@"
