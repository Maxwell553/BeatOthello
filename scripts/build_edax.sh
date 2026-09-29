#!/usr/bin/env bash
# Build Edax (pro-strength Othello engine) for this machine.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EDAX="$ROOT/third_party/edax-reversi"
mkdir -p "$EDAX/bin" "$EDAX/data"
if [[ ! -f "$EDAX/data/eval.dat" ]]; then
  echo "Missing $EDAX/data/eval.dat"
  echo "Download eval.7z from https://github.com/abulmo/edax-reversi/releases/tag/v4.4"
  exit 1
fi
cd "$EDAX/src"
make build OS=osx COMP=clang ARCH=native CC=clang
echo "Built $EDAX/bin/mEdax-native"
