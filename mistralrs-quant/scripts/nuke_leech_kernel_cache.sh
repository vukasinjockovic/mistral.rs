#!/usr/bin/env bash
# Full reset of the leech / leech_q24 CUDA compilation cache.
#
# cudaforge content-hashes only `.cu` files, not `.cuh` or `.h`, and its cache
# doesn't invalidate when the set of `.cu` files changes. When you:
#   - edit a `.cuh` / `.h` header included by a kernel,
#   - add or remove a `.cu` file,
#   - change anything compile-time-baked (constants, table values, ...),
# cargo will happily reuse the stale `.o` and link a kernel that doesn't
# match the source. The symptom is usually an "undefined symbol" linker
# error or, worse, a benchmark that silently runs the previous kernel.
#
# This script does the full nuke:
#   1. Delete `.cudaforge_cache.json` in every mistralrs-quant build dir.
#   2. Delete every `leech_*.o` in every mistralrs-quant OUT_DIR.
#   3. Delete cached test binaries that link the kernel statically.
#   4. Bust each `.cu` file's content hash by appending a timestamp comment
#      (replacing any prior bust marker so the file stays clean over time).
#
# After running this, do:
#   cargo build --features cuda --release -p mistralrs-quant
#
# Usage:
#   ./mistralrs-quant/scripts/nuke_leech_kernel_cache.sh
#   ./mistralrs-quant/scripts/nuke_leech_kernel_cache.sh --quiet
set -euo pipefail

QUIET=0
for arg in "$@"; do
  case "$arg" in
    -q|--quiet) QUIET=1 ;;
    *) echo "unknown arg: $arg"; exit 2 ;;
  esac
done

log() {
  if [ "$QUIET" -eq 0 ]; then
    echo "$@"
  fi
}

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

if [ ! -d "mistralrs-quant/kernels/leech" ] && [ ! -d "mistralrs-quant/kernels/leech_q24" ]; then
  echo "error: run this from the mistral.rs repo root" >&2
  exit 1
fi

# 1) Cudaforge cache files
log "[1/4] removing cudaforge cache files..."
n=0
while IFS= read -r -d '' f; do
  rm -f "$f"
  log "    rm $f"
  n=$((n + 1))
done < <(find target -type f -name ".cudaforge_cache.json" -path "*mistralrs-quant-*/out/*" -print0 2>/dev/null)
log "    ($n removed)"

# 2) Stale leech .o object files
log "[2/4] removing stale leech_*.o object files..."
n=0
while IFS= read -r -d '' f; do
  rm -f "$f"
  n=$((n + 1))
done < <(find target -type f -name "leech_*.o" -path "*mistralrs-quant-*/out/*" -print0 2>/dev/null)
log "    ($n removed)"

# 3) Stale test binaries that link the kernel archive
log "[3/4] removing stale leech_*_cuda-* test binaries..."
n=0
while IFS= read -r -d '' f; do
  rm -f "$f"
  n=$((n + 1))
done < <(find target -type f \( -name "leech_q24_*_cuda-*" -o -name "leech_*_cuda-*" \) -path "*target/*/deps/*" -print0 2>/dev/null)
log "    ($n removed)"

# 4) Bust .cu file content hashes
log "[4/4] busting .cu content hashes..."
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
n=0
for f in mistralrs-quant/kernels/leech/*.cu mistralrs-quant/kernels/leech_q24/*.cu; do
  if [ -f "$f" ]; then
    # Drop any previous bust marker so the file doesn't accumulate them.
    if grep -qE '^// __nuke_cache: ' "$f" 2>/dev/null; then
      sed -i '/^\/\/ __nuke_cache: /d' "$f"
    fi
    printf '\n// __nuke_cache: %s\n' "$TIMESTAMP" >> "$f"
    n=$((n + 1))
  fi
done
log "    ($n .cu files busted with marker __nuke_cache: $TIMESTAMP)"

log "done. now run:"
log "    cargo build --features cuda --release -p mistralrs-quant"
