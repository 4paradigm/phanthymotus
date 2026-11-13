#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
exec python3 "$ROOT/bin/runner.py" stop --root "$ROOT"
