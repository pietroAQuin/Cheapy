#!/usr/bin/env bash
# Run the cost-only benchmark bar chart over export/ (or data/) and write
# results/total_cost.png. All flags are passed straight through to
# research/legacy/benchmark.py — see `./run_benchmark.sh --help`.
set -euo pipefail

cd "$(dirname "$0")"

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3 || command -v python)"
fi
if [ -z "$PY" ]; then
  echo "no python found: create the venv (python3 -m venv .venv) and install requirements-dev.txt" >&2
  exit 1
fi

exec "$PY" research/legacy/benchmark.py "$@"
