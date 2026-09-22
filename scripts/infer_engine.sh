#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UNICACHE_KIVI_ROOT="${UNICACHE_KIVI_ROOT:-$ROOT/third_party/KIVI}"
exec "${PYTHON:-python}" "$ROOT/infer.py" "$@" --backend engine
