#!/usr/bin/env bash
# Relocation check for a built profile directory (issue #659).
# Usage: scripts/tests/relocatable_bins.sh [target/release]
# Needs no GPU. Run after `cargo build --release --features cuda` and, for the
# test-binary check, `cargo test --release --features cuda --no-run`.

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REL="$(cd "${1:-$ROOT/my-version/target/release}" && pwd)" || exit 1
BINS="rcsr rcsr-gpu-bench rcsr-bench cpu_bench rcsr_serve"
fail() { echo "FAIL: $*" >&2; exit 1; }
need() { [[ -n "$1" && -f "$1" ]] || fail "missing path: '$1'"; }

check_elf() {
  local t="$1" label="$2" d
  need "$t"
  d="$(readelf -d "$t")" || fail "$label: readelf failed"
  [[ -n "$d" ]] || fail "$label: readelf printed nothing"
  grep -E 'RUNPATH|RPATH' <<<"$d" | grep -q '/build/' && fail "$label: RUNPATH points into a build dir"
  grep -q librcsrcuda <<<"$d" && fail "$label: NEEDED librcsrcuda"
  return 0
}

need "$REL/rcsr"
for b in $BINS; do
  [[ -x "$REL/$b" ]] || continue
  check_elf "$REL/$b" "$b"
  libs="$(ldd "$REL/$b")" || fail "$b: ldd failed"
  grep -q 'not found' <<<"$libs" && fail "$b: unresolved library"
  grep -qF "$REL" <<<"$libs" && fail "$b: resolves a library inside $REL"
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
    check_elf "$t" "$(basename "$t")"
    "$t" --list >/dev/null 2>&1 || fail "$(basename "$t") --list failed"
    n=$((n + 1))
  done
done
(( n > 0 )) || fail "no test binaries found in $REL/deps"
echo "ok $n test binaries start"
