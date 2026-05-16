#!/usr/bin/env bash
# Full reset of the leech / leech_q24 CUDA compilation cache.
#
# cudaforge content-hashes only `.cu` files, not `.cuh` or `.h`, and its cache
# doesn't fully invalidate when the set of `.cu` files changes. When you:
#   - edit a `.cuh` / `.h` header included by a kernel,
#   - change anything compile-time-baked (constants, table values, ...),
# cargo will happily reuse the stale `.o` and link a kernel that doesn't
# match the source. The symptom is usually an "undefined symbol" linker
# error or, worse, a benchmark that silently runs the previous kernel.
#
# This script does the full nuke:
#   1. Delete `.cudaforge_cache.json` in every mistralrs-quant build dir.
#   2. Delete every `leech_*.o` in every mistralrs-quant OUT_DIR.
#   3. Delete cached test binaries that link the kernel statically.
#   4. Write an UNTRACKED sentinel `_bust_<timestamp>.cu` into each kernels/
#      subdir, replacing any prior sentinel. cudaforge sees a new file in
#      its source_glob → hashes everything from scratch. Because the
#      sentinel filename is .gitignored, the working tree stays clean and
#      `git pull` continues to fast-forward.
#
# After running this, do:
#   cargo build --features cuda --release -p mistralrs-quant
#
# Usage:
#   ./mistralrs-quant/scripts/nuke_leech_kernel_cache.sh
#   ./mistralrs-quant/scripts/nuke_leech_kernel_cache.sh --quiet
#   ./mistralrs-quant/scripts/nuke_leech_kernel_cache.sh --revert
#       Remove sentinels (use when you're done iterating, or if a previous
#       version of this script appended an in-place `// __nuke_cache:` line
#       to a tracked .cu — that legacy form is also stripped here).
set -euo pipefail

QUIET=0
REVERT=0
for arg in "$@"; do
  case "$arg" in
    -q|--quiet) QUIET=1 ;;
    --revert) REVERT=1 ;;
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

LEECH_DIRS=(
  "mistralrs-quant/kernels/leech"
  "mistralrs-quant/kernels/leech_q24"
)

any_dir_exists=0
for d in "${LEECH_DIRS[@]}"; do
  [ -d "$d" ] && any_dir_exists=1
done
if [ "$any_dir_exists" -eq 0 ]; then
  echo "error: run this from the mistral.rs repo root" >&2
  exit 1
fi

# ── --revert: clean up sentinels and legacy in-place bust markers ────
if [ "$REVERT" -eq 1 ]; then
  log "[revert] removing sentinel files..."
  n=0
  for d in "${LEECH_DIRS[@]}"; do
    [ -d "$d" ] || continue
    while IFS= read -r -d '' f; do
      rm -f "$f"
      n=$((n + 1))
    done < <(find "$d" -maxdepth 1 -type f -name "_bust_*.cu" -print0 2>/dev/null)
  done
  log "    ($n sentinel files removed)"

  # Legacy: strip any `// __nuke_cache:` trailing comments from earlier
  # script versions that mutated tracked .cu files in place.
  log "[revert] stripping legacy in-place bust markers..."
  m=0
  for d in "${LEECH_DIRS[@]}"; do
    [ -d "$d" ] || continue
    for f in "$d"/*.cu; do
      [ -f "$f" ] || continue
      if grep -qE '^// __nuke_cache: ' "$f" 2>/dev/null; then
        sed -i '/^\/\/ __nuke_cache: /d' "$f"
        # Drop trailing blank lines left behind.
        sed -i -e :a -e '/^$/{$d;N;ba' -e '}' "$f"
        m=$((m + 1))
      fi
    done
  done
  log "    ($m tracked .cu files cleaned)"
  log "done. tree should now be clean — verify with: git status"
  exit 0
fi

# ── 1) cudaforge cache files ─────────────────────────────────────────
log "[1/4] removing cudaforge cache files..."
n=0
while IFS= read -r -d '' f; do
  rm -f "$f"
  log "    rm $f"
  n=$((n + 1))
done < <(find target -type f -name ".cudaforge_cache.json" -path "*mistralrs-quant-*/out/*" -print0 2>/dev/null)
log "    ($n removed)"

# ── 2) stale leech .o objects ────────────────────────────────────────
log "[2/4] removing stale leech_*.o object files..."
n=0
while IFS= read -r -d '' f; do
  rm -f "$f"
  n=$((n + 1))
done < <(find target -type f -name "leech_*.o" -path "*mistralrs-quant-*/out/*" -print0 2>/dev/null)
log "    ($n removed)"

# ── 3) stale test binaries that statically link the kernel ──────────
log "[3/4] removing stale leech_*_cuda-* test binaries..."
n=0
while IFS= read -r -d '' f; do
  rm -f "$f"
  n=$((n + 1))
done < <(find target -type f \( -name "leech_q24_*_cuda-*" -o -name "leech_*_cuda-*" \) -path "*target/*/deps/*" -print0 2>/dev/null)
log "    ($n removed)"

# ── 4) write a gitignored sentinel into each kernels dir ─────────────
# cudaforge's source_glob is "kernels/*/*.cu". A new file in the glob
# changes the set cudaforge has to handle, forcing a re-evaluation.
# Filename pattern `_bust_*.cu` is .gitignored in each kernels subdir.
log "[4/4] writing gitignored bust sentinels..."
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
n=0
for d in "${LEECH_DIRS[@]}"; do
  [ -d "$d" ] || continue
  # Remove any prior sentinels from this directory.
  find "$d" -maxdepth 1 -type f -name "_bust_*.cu" -delete 2>/dev/null || true

  sentinel="$d/_bust_${TIMESTAMP}.cu"
  cat > "$sentinel" <<EOF
// AUTO-GENERATED by nuke_leech_kernel_cache.sh — DO NOT TRACK.
// Empty translation unit; its presence in cudaforge's source_glob is
// what forces a fresh rehash. Cleaned up by --revert.
namespace leech_q24 { inline void _bust_${TIMESTAMP//[^0-9A-Za-z]/_}() {} }
EOF
  log "    wrote $sentinel"
  n=$((n + 1))
done
log "    ($n sentinels written)"

log "done. now run:"
log "    cargo build --features cuda --release -p mistralrs-quant"
log "when finished iterating, clean up with:"
log "    ./mistralrs-quant/scripts/nuke_leech_kernel_cache.sh --revert"
