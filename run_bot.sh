#!/usr/bin/env bash
# Convenience launcher for the packaged original engine.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m bot "$@"
