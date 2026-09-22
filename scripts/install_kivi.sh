#!/usr/bin/env bash
# Build the pinned external KIVI dependency, without touching an existing checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${UNICACHE_KIVI_ROOT:-$ROOT/third_party/KIVI}"
COMMIT=876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6
if [[ -e "$DEST" ]]; then
  echo "Destination already exists: $DEST. Use a fresh UNICACHE_KIVI_ROOT directory." >&2
  exit 1
fi
git clone https://github.com/jy-yuan/KIVI.git "$DEST"
git -C "$DEST" checkout --detach "$COMMIT"
git -C "$DEST" apply --check "$ROOT/patches/kivi_long_query_outer_dim.patch"
git -C "$DEST" apply "$ROOT/patches/kivi_long_query_outer_dim.patch"
"${PYTHON:-python}" -m pip install --no-build-isolation --no-deps "$DEST/quant"
"${PYTHON:-python}" -c 'import kivi_gemv; print("KIVI extension imports successfully")'
printf '\nSet this before engine editing inference:\nexport UNICACHE_KIVI_ROOT="%s"\n' "$DEST"
