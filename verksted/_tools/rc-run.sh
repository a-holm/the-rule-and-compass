#!/usr/bin/env bash
# Usage: rc-run.sh build|test|run|cpu <worktree> <id> <tag> [--cuda] [--timeout S] -- <command...>
# build: through the build slot ($B), default timeout 1500 s (cargo build, cargo test --no-run).
# test:  through the test lock ($L) with --cuda, else through the build slot (CPU-only tests); default 580 s.
# run:   through the test lock ($L) for rcsr, rcsr-bench and any GPU run; default 580 s.
# cpu:   any CPU-only command through the build slot; default 580 s.
# Writes: the log $(log_path <id> <tag>) (overwritten on rerun), the stamp, one lock file $GATES/.rc-run-<hash>.lock per job.
# The command runs in <worktree>/<crate dir> with CARGO_BUILD_JOBS=2 (and RCSR_CUDA_ARCH=$CUDA_ARCH with --cuda)
# pinned with `taskset -c` to PIN_CORES_BUILD of $VLOCAL/machine.md (PIN_CORES_BUSY while $OC/desktop-busy.sh exits 0; no default stands in
# for a missing key) under an outer `timeout -k $KILL_AFTER` (30 s, override RCSR_KILL_AFTER). Lock modes refuse --timeout above LOCK_HOLD_MAX (600 s).
# A command whose program is a file in the deps directory of a target directory is refused: tests start through cargo test, never as a binary.
# Lock modes run the command through lock-run.sh: its own session; when it owns the flock on $LOCKFILE, every process sharing that flock after the command ended is
# TERMed, then KILLed, by pid, so a crashed tool cannot leave the lock held.
# Lock modes add the line `xid new=<n> since=<start>` (new NVRM Xid lines since the start of the hold); the first log line is
# `RC_HEAD sha=<HEAD of the worktree, none outside git> dirty=<0|1>` and the last is `RC_DONE rc=<n> at=<UTC>`.
# One job per <worktree>, <id> and <tag> at a time: a second identical job exits 3 naming the pid of the first and leaves its log alone.
# A successful build or test job, when <crate dir>/target/release/rcsr exists, writes the feature set of the job (`cuda` with --cuda, else `cpu`)
# to <crate dir>/target/release/.rcsr-features. `run` exits 3 when that stamp names the other feature set than the job (no --cuda is cpu) and prints the
# build command that fixes it; without a stamp it runs.
# Exit: the command's exit code; 3 a duplicate job or a feature stamp mismatch; 70 when a new Xid line appeared or the Xid check failed; 71 when a process still held the lock after the KILL; 75 when the slot or lock gave up; 2 usage or refused input.
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
C=$(crate_dir "$wt") || refuse "no single engine crate under $wt"
[ -z "${CARGO_TARGET_DIR:-}" ] || refuse "CARGO_TARGET_DIR is set; each worktree uses its own target/"
timeout_s=${timeout_s:-$default_s}
case "$timeout_s" in
  ''|*[!0-9]*) refuse "--timeout must be whole seconds" ;;
esac
if [ "$lock" -eq 1 ] && [ "$timeout_s" -gt "$LOCK_HOLD_MAX" ]; then
  refuse "--timeout $timeout_s exceeds the lock hold cap $LOCK_HOLD_MAX"
fi
prog=
for a in "$@"; do
  case "$a" in env | [A-Za-z_]*=*) continue ;; esac
  prog=$a
  break
done
case "$prog" in
  */target/*/deps/*) refuse "$prog is a test binary: start tests with cargo test through rc-run.sh, never as a binary" ;;
esac
need_gates

pin_key=PIN_CORES_BUILD
if [ -x "$OC/desktop-busy.sh" ] && "$OC/desktop-busy.sh"; then pin_key=PIN_CORES_BUSY; fi
pin=$(machine_key "$pin_key") || exit 2
case "$pin" in
  *[!0-9,-]* | '' | *[,-] | [,-]*) refuse "key $pin_key is not a core list: $pin" ;;
esac

wt=$(cd "$wt" && pwd)
features=cpu
[ "$cuda" -eq 0 ] || features=cuda
stamp=$C/target/release/.rcsr-features
if [ "$mode" = run ] && [ -f "$stamp" ]; then
  have=$(cat "$stamp")
  if [ "$have" != "$features" ]; then
    fix="rc-run.sh build $wt $id $tag-build"
    [ "$cuda" -eq 0 ] || fix="$fix --cuda"
    fix="$fix -- cargo build --release"
    [ "$cuda" -eq 0 ] || fix="$fix --features cuda"
    echo "rc-run: the release rcsr binary of $wt was built for the $have feature set, this job needs $features; build it first: $fix --bin rcsr" >&2
    exit 3
  fi
fi

key=$(printf '%s\n%s\n%s\n' "$wt" "$id" "$tag" | sha1sum | cut -c1-20)
dupfile=$GATES/.rc-run-$key.lock
exec {dupfd}>> "$dupfile"
if ! flock -n "$dupfd"; then
  echo "rc-run: the job $id $tag of $wt already runs (pid $(cat "$dupfile" 2> /dev/null))" >&2
  exit 3
fi
: > "$dupfile"
echo $$ > "$dupfile"

env_prefix="CARGO_BUILD_JOBS=2"
[ "$cuda" -eq 1 ] && env_prefix="$env_prefix RCSR_CUDA_ARCH=$CUDA_ARCH"
reap=
if [ "$lock" -eq 1 ]; then
  reap=$(printf 'bash %q %q ' "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lock-run.sh" "$LOCKFILE")
fi
inner=$(printf 'cd %q && %s %staskset -c %s timeout -k %s %s' "$C" "$env_prefix" "$reap" "$pin" "$KILL_AFTER" "$timeout_s")
for arg in "$@"; do
  inner="$inner $(printf '%q' "$arg")"
done

log=$(log_path "$id" "$tag")
: > "$log"
echo "$log"
sha=$(git -C "$wt" rev-parse HEAD 2> /dev/null)
[ -z "$(git -C "$wt" rev-parse --show-prefix 2> /dev/null)" ] || sha=
if [ -n "$sha" ]; then
  dirty=0
  [ -z "$(git -C "$wt" status --porcelain 2> /dev/null)" ] || dirty=1
else
  sha=none dirty=1
fi
echo "RC_HEAD sha=$sha dirty=$dirty" >> "$log"
xs=
[ "$lock" -eq 1 ] && xs=$(xid_start)
"$runner" bash -c "$inner" >> "$log" 2>&1 {dupfd}>&-
rc=$?
if [ "$lock" -eq 1 ]; then
  if xn=$(xid_new_since "$xs" 2>> "$log"); then
    echo "xid new=$xn since=$xs" >> "$log"
    if [ "$xn" -gt 0 ]; then
      echo "STOP: $xn new NVRM Xid line(s) since $xs" >> "$log"
      rc=70
    fi
  else
    echo "STOP: Xid check failed (start=${xs:-missing}); failing closed" >> "$log"
    rc=70
  fi
fi
if [ "$rc" -eq 0 ] && { [ "$mode" = build ] || [ "$mode" = test ]; } && [ -f "$C/target/release/rcsr" ]; then
  echo "$features" > "$stamp"
fi
line=$(rc_done "$rc")
echo "$line" >> "$log"
echo "$line"
exit "$rc"
