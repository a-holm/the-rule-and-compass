#!/usr/bin/env bash
# Usage: sanitize.sh <worktree> <id> <tag> (--test <file> | --lib) [--timeout S] [--api-errors no|explicit|all] [-- <test filters and flags>]
# Builds the CUDA test binary in the build slot with `cargo test --release --features cuda --no-run --message-format=json`,
# takes its path from the compiler-artifact JSON line (profile.test true; target.name = <file>, or target.kind holding lib),
# and runs it under the CUDA 13.0 compute-sanitizer memcheck through the test lock with the Xid check:
# env RCSR_UNDER_SANITIZER=memcheck RCSR_REQUIRE_DEVICE=1 $SANITIZER --tool memcheck --leak-check no --error-exitcode 99 [--report-api-errors <v>] <binary> <filters>.
# RCSR_UNDER_SANITIZER lets a test pick a smaller fixture (tests/common/sanitizer.rs). --api-errors no stops deliberate driver errors counting as sanitizer errors.
# Run the whole planned pass with sanitize-all.sh.
# Search-level tests must set dispatch: DispatchMode::Device (CLI --dispatch device), or every stage runs on the CPU.
# Writes: the logs <id>-<tag>-build and <id>-<tag> (see rc-run.sh).
# Exit: 0 when the log holds `ERROR SUMMARY: 0 errors` and the run exits 0; 1 otherwise; 3 when $SANITIZER is missing or not a CUDA 13 path; 70 Xid rose; 2 usage.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RC_RUN=$(dirname "${BASH_SOURCE[0]}")/rc-run.sh

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac
[ $# -ge 4 ] || { usage >&2; exit 2; }
wt=$1 id=$2 tag=$3
shift 3
target=
timeout_s=580
api_errors=
while [ $# -gt 0 ]; do
  case "$1" in
    --test)
      [ $# -ge 2 ] && [ -z "$target" ] || { usage >&2; exit 2; }
      target=test:$2
      shift 2
      ;;
    --lib)
      [ -z "$target" ] || { usage >&2; exit 2; }
      target=lib
      shift
      ;;
    --api-errors)
      [ $# -ge 2 ] || { usage >&2; exit 2; }
      api_errors=$2
      shift 2
      ;;
    --timeout)
      [ $# -ge 2 ] || { usage >&2; exit 2; }
      timeout_s=$2
      shift 2
      ;;
    --) shift; break ;;
    *) usage >&2; exit 2 ;;
  esac
done
[ -n "$target" ] || { usage >&2; exit 2; }
case "$timeout_s" in
  ''|*[!0-9]*) echo "sanitize: --timeout must be whole seconds" >&2; exit 2 ;;
esac
case "$api_errors" in
  ""|no|explicit|all) ;;
  *) echo "sanitize: --api-errors must be no, explicit or all" >&2; exit 2 ;;
esac
if [ "$timeout_s" -gt "$LOCK_HOLD_MAX" ]; then
  echo "sanitize: --timeout $timeout_s exceeds the lock hold cap $LOCK_HOLD_MAX" >&2
  exit 2
fi

san_real=$(realpath "$SANITIZER" 2>/dev/null || true)
case "$san_real" in
  */cuda-13*/*) ;;
  *) echo "sanitize: $SANITIZER is not a CUDA 13 compute-sanitizer (needs /cuda-13 in its real path; /usr/bin/compute-sanitizer is CUDA 12 and lacks the injection library)" >&2; exit 3 ;;
esac
[ -x "$san_real" ] && [ -f "$san_real" ] || { echo "sanitize: $SANITIZER is not executable" >&2; exit 3; }

build_args=(cargo test --release --features cuda --no-run --message-format=json)
case "$target" in
  test:*) build_args+=(--test "${target#test:}") ;;
  lib) build_args+=(--lib) ;;
esac
"$RC_RUN" build "$wt" "$id" "$tag-build" --cuda -- "${build_args[@]}" > /dev/null
brc=$?
build_log=$(log_path "$id" "$tag-build")
if [ "$brc" -ne 0 ]; then
  echo "sanitize: build failed (rc=$brc); see $build_log" >&2
  exit 1
fi

case "$target" in
  test:*) name=${target#test:}
    filter='select(.reason=="compiler-artifact" and .profile.test==true and .target.name==$n and .executable != null) | .executable' ;;
  lib) name=
    filter='select(.reason=="compiler-artifact" and .profile.test==true and (.target.kind | index("lib")) and .executable != null) | .executable' ;;
esac
bins=$(grep '^{' "$build_log" | jq -r --arg n "$name" "$filter" 2>/dev/null | sort -u)
count=$(printf '%s' "$bins" | grep -c .)
if [ "$count" -ne 1 ]; then
  echo "sanitize: expected exactly one test binary for ${target#test:}, found $count: $(printf '%s' "$bins" | tr '\n' ' ')" >&2
  exit 1
fi

"$RC_RUN" run "$wt" "$id" "$tag" --cuda --timeout "$timeout_s" -- \
  env RCSR_UNDER_SANITIZER=memcheck RCSR_REQUIRE_DEVICE=1 "$SANITIZER" --tool memcheck --leak-check no --error-exitcode 99 ${api_errors:+--report-api-errors "$api_errors"} "$bins" "$@"
rc=$?
log=$(log_path "$id" "$tag")
[ "$rc" -ne 70 ] || exit 70
grep -q 'running 0 tests' "$log" && echo "WARNING: tests run: 0"
if [ "$rc" -eq 0 ] && grep -q 'ERROR SUMMARY: 0 errors' "$log"; then
  exit 0
fi
echo "sanitize: failed (rc=$rc); see $log" >&2
exit 1
