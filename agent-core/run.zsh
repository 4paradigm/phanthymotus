#!/usr/bin/env zsh
# run.zsh — Start agent core
set -e

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

export KMP_DUPLICATE_LIB_OK=TRUE
exec uv run python src/start.py "$@"
