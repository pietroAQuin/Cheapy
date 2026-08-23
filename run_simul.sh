#!/usr/bin/env bash
# Run the routing simulation over every chunk in data/ and write data/routing_simul.csv.
# All flags are passed straight through to src/cheapy/cli.py — see `./run_simul.sh --help`.
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

exec "$PY" src/cheapy/cli.py "$@"
