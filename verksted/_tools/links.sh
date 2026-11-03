#!/usr/bin/env bash
# Usage: links.sh [--strict]
# Builds verksted-check in the build slot, then runs `links` and `walk` on verksted/.
# --strict adds the index, orphan tool and root route checks. No test lock.
# Writes: the target/ directory of the crate.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
strict=()
case "${1:-}" in
  '') ;;
  -h|--help) usage; exit 0 ;;
  --strict) strict=(--strict) ;;
  *) usage >&2; exit 2 ;;
esac
[ $# -le 1 ] || { usage >&2; exit 2; }
crate=$VERKSTED/_tools/verksted-check
build=(bash -c "cd '$crate' && CARGO_BUILD_JOBS=2 timeout 600 cargo build --release")
if [ -x "$B" ]; then "$B" "${build[@]}"; else "${build[@]}"; fi
rc=$?
if [ "$rc" -eq 0 ]; then
  bin=$crate/target/release/verksted-check
  "$bin" links "$VERKSTED"; a=$?
  "$bin" walk "${strict[@]}" "$VERKSTED"; b=$?
  [ "$a" -eq 0 ] && [ "$b" -eq 0 ] || rc=1
fi
rc_done "$rc"
exit "$rc"
