#!/usr/bin/env bash
# Relocation check for a built profile directory (issue #659).
# Usage: scripts/tests/relocatable_bins.sh [target/release]
# Needs no GPU. Run after `cargo build --release --features cuda` and, for the
# test-binary check, `cargo test --release --features cuda --no-run`.

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REL="$(cd "${1:-$ROOT/my-version/target/release}" && pwd)"
BINS="rcsr rcsr-gpu-bench rcsr-bench cpu_bench rcsr_serve"
fail() { echo "FAIL: $*" >&2; exit 1; }

for b in $BINS; do
  [[ -x "$REL/$b" ]] || continue
  dyn="$(readelf -d "$REL/$b")"
  grep -E 'RUNPATH|RPATH' <<<"$dyn" | grep -q '/build/' && fail "$b: RUNPATH points into a build dir"
  grep -q librcsrcuda <<<"$dyn" && fail "$b: NEEDED librcsrcuda"
  ldd "$REL/$b" | grep -q 'not found' && fail "$b: unresolved library"
  ldd "$REL/$b" | grep -qF "$REL" && fail "$b: resolves a library inside $REL"
  echo "ok layout $b"
done

dir="$(mktemp -d)"
trap 'rm -f "$dir/rcsr"; rmdir "$dir"' EXIT
cp "$REL/rcsr" "$dir/rcsr"
env -i PATH="$PATH" "$dir/rcsr" --version >/dev/null || fail "copied rcsr --version failed"
echo "ok copied rcsr runs outside the target dir"

n=0
for src in "$ROOT"/my-version/tests/*.rs; do
  name="$(basename "$src" .rs)"
  for t in "$REL"/deps/"$name"-*; do
    [[ -f "$t" && -x "$t" && "$t" != *.d ]] || continue
    ! cmp -s "$t" "$REL/$name" || continue
    "$t" --list >/dev/null 2>&1 || fail "$(basename "$t") --list failed"
    n=$((n + 1))
  done
done
echo "ok $n test binaries start"
