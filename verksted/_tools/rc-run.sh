#!/usr/bin/env bash
# Usage: rc-run.sh build|test|run|cpu <worktree> <id> <tag> [--cuda] [--timeout S] -- <command...>
# build: through the build slot ($B), default timeout 1500 s (cargo build, cargo test --no-run).
# test:  through the test lock ($L) with --cuda, else through the build slot (CPU-only tests); default 580 s.
# run:   through the test lock ($L) for rcsr, rcsr-bench and any GPU run; default 580 s.
# cpu:   any CPU-only command through the build slot; default 580 s.
# The command runs in <worktree>/my-version with CARGO_BUILD_JOBS=2 (and RCSR_CUDA_ARCH=sm_86 with --cuda)
# under an outer `timeout -k $KILL_AFTER` (30 s, override RCSR_KILL_AFTER). Lock modes refuse --timeout above LOCK_HOLD_MAX (600 s).
# Lock modes run the command through lock-run.sh: its own session; when it owns the flock on $LOCKFILE, every process sharing that flock after the command ended is
# TERMed, then KILLed, by pid, so a crashed tool cannot leave the lock held.
# Lock modes add the line `xid before=<a> after=<b>`; the last log line is `RC_DONE rc=<n> at=<UTC>`.
# Writes: the log $(log_path <id> <tag>) (overwritten on rerun).
# Exit: the command's exit code; 70 when the Xid count rose; 71 when a process still held the lock after the KILL; 75 when the slot or lock gave up; 2 usage or refused input.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

refuse() {
  echo "rc-run: $1" >&2
  exit 2
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac
[ $# -ge 4 ] || { usage >&2; exit 2; }
mode=$1 wt=$2 id=$3 tag=$4
shift 4
cuda=0
timeout_s=
while [ $# -gt 0 ]; do
  case "$1" in
    --cuda) cuda=1; shift ;;
    --timeout)
      [ $# -ge 2 ] || refuse "--timeout needs a value"
      timeout_s=$2
      shift 2
      ;;
    --) shift; break ;;
    *) refuse "unexpected argument '$1' before --" ;;
  esac
done
[ $# -gt 0 ] || refuse "no command after --"

case "$mode" in
  build) runner=$B; lock=0; default_s=1500 ;;
  test) if [ "$cuda" -eq 1 ]; then runner=$L; lock=1; else runner=$B; lock=0; fi; default_s=580 ;;
  run) runner=$L; lock=1; default_s=580 ;;
  cpu) runner=$B; lock=0; default_s=580 ;;
  *) refuse "unknown mode '$mode' (build|test|run|cpu)" ;;
esac
[ -f "$wt/my-version/Cargo.toml" ] || refuse "$wt/my-version/Cargo.toml missing"
[ -z "${CARGO_TARGET_DIR:-}" ] || refuse "CARGO_TARGET_DIR is set; each worktree uses its own target/"
timeout_s=${timeout_s:-$default_s}
case "$timeout_s" in
  ''|*[!0-9]*) refuse "--timeout must be whole seconds" ;;
esac
if [ "$lock" -eq 1 ] && [ "$timeout_s" -gt "$LOCK_HOLD_MAX" ]; then
  refuse "--timeout $timeout_s exceeds the lock hold cap $LOCK_HOLD_MAX"
fi
need_gates

wt=$(cd "$wt" && pwd)
env_prefix="CARGO_BUILD_JOBS=2"
[ "$cuda" -eq 1 ] && env_prefix="$env_prefix RCSR_CUDA_ARCH=$CUDA_ARCH"
reap=
if [ "$lock" -eq 1 ]; then
  reap=$(printf 'bash %q %q ' "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lock-run.sh" "$LOCKFILE")
fi
inner=$(printf 'cd %q && %s %stimeout -k %s %s' "$wt/my-version" "$env_prefix" "$reap" "$KILL_AFTER" "$timeout_s")
for arg in "$@"; do
  inner="$inner $(printf '%q' "$arg")"
done

log=$(log_path "$id" "$tag")
: > "$log"
echo "$log"
x0=0
[ "$lock" -eq 1 ] && x0=$(xid_count)
"$runner" bash -c "$inner" >> "$log" 2>&1
rc=$?
if [ "$lock" -eq 1 ]; then
  x1=$(xid_count)
  echo "xid before=$x0 after=$x1" >> "$log"
  if [ "$x1" -gt "$x0" ]; then
    echo "STOP: NVRM Xid count rose $x0 -> $x1" >> "$log"
    rc=70
  fi
fi
line=$(rc_done "$rc")
echo "$line" >> "$log"
echo "$line"
exit "$rc"
