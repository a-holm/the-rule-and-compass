#!/usr/bin/env bash
# Usage: gate.sh <worktree> <tag>
# The full gate on an integration tip: CPU suite, CUDA suite, tiny benchmark (27/27).
# The CPU suite and the benchmark are one lock hold each. The CUDA suite is built once in the build slot
# (cargo test --features cuda --no-run --message-format=json), then every test target (lib, each bin, each
# --test file, --doc) runs in its own lock hold under `timeout LOCK_HOLD_MAX-20`, so one target that times out or
# fails never stops the others and no hold exceeds the cap. A target above the limit is a failure: size it, the
# limit is never raised.
# Exit 0 only when every suite passes, no CUDA target failed or timed out, and the benchmark recovers 27/27.
# Exit 70 at once when the Xid count rose during a CUDA target (the remaining targets are not run).
# Writes: $D/<tag>-{cpu,cuda,bench}.log, $D/<tag>-cuda-{build,summary}.log, $D/<tag>-cuda-<target>.log
# The summary has one line per target: <target> rc=<n> xid=<a>-><b> wait=<s> run=<s> | <test result line> [| timeout].
# wait is the time from asking for the lock to the hold start (the hold writes its own start time, RCSR_NOW_CMD, default `date +%s`); run is the time from the hold start to its end.
# Every hold (CPU suite, each CUDA target, benchmark) runs through lock-run.sh: a process the command leaves holding the lock is ended by pid.
# $D/<tag>-cuda.log is the build failure (if any) plus every target log under a `=== <target> ===` header.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac
if [ $# -ne 2 ]; then
  usage >&2
  exit 2
fi
W=$1
T=$2
D=${RCSR_GATE_DIR:-$GATES}
if [ ! -d "$D" ]; then
  echo "gate dir $D missing; link it to the log directory or set RCSR_GATES" >&2
  exit 1
fi
LOCK_RUN=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lock-run.sh
NOW_CMD=${RCSR_NOW_CMD:-date +%s}
now() { bash -c "$NOW_CMD"; }
# hold <command>: one test-lock hold; lock-run.sh ends every process the command leaves holding the lock
hold() {
  "$L" bash "$LOCK_RUN" "$LOCKFILE" bash -c "$1"
}
step() {
  local name=$1 cmd=$2
  hold "cd '$W/my-version' && CARGO_BUILD_JOBS=2 $cmd" > "$D/$T-$name.log" 2>&1
}

cuda_env="CARGO_BUILD_JOBS=2 RCSR_CUDA_ARCH=$CUDA_ARCH RCSR_REQUIRE_DEVICE=1"
cuda_cargo="cargo test --release --features cuda"
cuda_hold=$((LOCK_HOLD_MAX - 20))
cuda_targets=0
cuda_failed=0
cuda_timeouts=0

hold_times() { # <call> <hold start> <end>: wait for the lock and run time of the hold, both in seconds
  if [ -z "$2" ]; then
    echo "wait=$(($3 - $1)) run=none"
  else
    echo "wait=$(($2 - $1)) run=$(($3 - $2))"
  fi
}

# cuda_suite: sets g (0, 1 or 70) and the counters above.
cuda_suite() {
  local log=$D/$T-cuda.log summary=$D/$T-cuda-summary.log build=$D/$T-cuda-build.log
  local targets name args tlog rc xb xa res line t0 t1 ts start
  : > "$log"
  : > "$summary"
  "$B" bash -c "cd '$W/my-version' && $cuda_env timeout -k $KILL_AFTER 5400 $cuda_cargo --no-run --message-format=json" > "$build" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    cat "$build" >> "$log"
    echo "cuda build failed rc=$rc" | tee -a "$summary" >> "$log"
    g=1
    return
  fi
  targets=$(grep -a '^{' "$build" | jq -r '
    select(.reason == "compiler-artifact" and .profile.test == true and .executable != null)
    | if (.target.kind | index("lib")) then "lib\t--lib"
      elif (.target.kind | index("bin")) then "bin-" + .target.name + "\t--bin " + .target.name
      elif (.target.kind | index("test")) then .target.name + "\t--test " + .target.name
      else empty end' 2> /dev/null | sort -u)
  if [ -z "$targets" ]; then
    cat "$build" >> "$log"
    echo "cuda build lists no test target" | tee -a "$summary" >> "$log"
    g=1
    return
  fi
  targets=$(printf '%s\ndoc\t--doc\n' "$targets")
  g=0
  while IFS=$'\t' read -r name args; do
    cuda_targets=$((cuda_targets + 1))
    tlog=$D/$T-cuda-$name.log
    xb=$(xid_count)
    start=$D/$T-cuda-$name.start
    rm -f "$start"
    t0=$(now)
    hold "$NOW_CMD > '$start' && cd '$W/my-version' && $cuda_env timeout -k $KILL_AFTER $cuda_hold $cuda_cargo --no-fail-fast $args" > "$tlog" 2>&1
    rc=$?
    t1=$(now)
    ts=$(cat "$start" 2> /dev/null)
    rm -f "$start"
    xa=$(xid_count)
    res=$(grep -a '^test result:' "$tlog" | tail -n 1)
    line="$name rc=$rc xid=$xb->$xa $(hold_times "$t0" "$ts" "$t1") | ${res:-no test result line}"
    if [ "$rc" -eq 124 ]; then
      line="$line | timeout"
      cuda_timeouts=$((cuda_timeouts + 1))
    fi
    [ "$rc" -eq 0 ] || cuda_failed=$((cuda_failed + 1))
    echo "$line" >> "$summary"
    { echo "=== $name ==="; cat "$tlog"; } >> "$log"
    if [ "$xa" -gt "$xb" ]; then
      echo "Xid count rose during $name; remaining CUDA targets not run" | tee -a "$summary" >> "$log"
      g=70
      return
    fi
  done <<< "$targets"
  [ "$cuda_failed" -eq 0 ] || g=1
}

step cpu "timeout -k $KILL_AFTER 5400 cargo test --release --no-fail-fast"; c=$?
g=0
cuda_suite
step bench "timeout -k $KILL_AFTER 5400 cargo build --release && timeout -k $KILL_AFTER 1800 ./target/release/rcsr-bench --tiny"; b=$?
bad=0
for k in cpu cuda; do
  f=$D/$T-$k.log
  ok=$(grep -ac 'test result: ok' "$f")
  failed=$(grep -aEc 'test result: FAILED|^error(\[|:)' "$f")
  echo "$k: ok=$ok failed=$failed"
  [ "$failed" -eq 0 ] || bad=1
done
echo "cuda: targets=$cuda_targets failed=$cuda_failed timeouts=$cuda_timeouts"
rec=$(grep -a -E '^recovered [0-9]+/[0-9]+' "$D/$T-bench.log" | tail -n 1)
echo "bench: ${rec:-no recovered line}"
echo "exit cpu=$c cuda=$g bench=$b"
echo "tested: $(git -C "$W" rev-parse HEAD)"
case "$rec" in *"recovered 27/27"*) ;; *) bad=1 ;; esac
[ "$g" -ne 70 ] || exit 70
[ "$c" -eq 0 ] && [ "$g" -eq 0 ] && [ "$b" -eq 0 ] && [ "$bad" -eq 0 ]
