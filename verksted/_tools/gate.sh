#!/usr/bin/env bash
# Usage: gate.sh <worktree> <tag>
# The full gate on an integration tip: CPU suite, CUDA suite, tiny benchmark (27/27).
# The benchmark is one lock hold. Each suite is built once in the build slot (cargo test [--features cuda] --no-run
# --message-format=json), then every test target (lib, each bin, each --test file, --doc) runs in its own lock hold under
# `timeout LOCK_HOLD_MAX-20`, so one target that times out or fails never stops the others and no hold exceeds the cap.
# A target above the limit is a failure: size it, the limit is never raised.
# Exit 0 only when every suite passes, the verksted-check tests and `links --external` pass (log $D/<tag>-verksted.log), no target failed or timed out, and the benchmark recovers 27/27.
# Exit 70 at once when a new Xid line appeared (or the Xid check failed) during a CUDA target (the remaining targets are not run).
# Writes: $D/<tag>-{cpu,cuda,bench}.log, $D/<tag>-verksted.log, $D/<tag>-{cpu,cuda}-{build,summary}.log, $D/<tag>-{cpu,cuda}-<target>.log
# The summary has one line per target: <target> rc=<n> [xid=new:<n>] wait=<s> run=<s> | <test result line> [| timeout]. The xid field is CUDA only.
# wait is the time from asking for the lock to the hold start (the hold writes its own start time, RCSR_NOW_CMD, default `date +%s`); run is the time from the hold start to its end.
# Every hold (each CPU and CUDA target, benchmark) runs through lock-run.sh: a process the command leaves holding the lock is ended by pid.
# $D/<tag>-{cpu,cuda}.log is the build failure (if any) plus every target log under a `=== <target> ===` header.
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
C=$(crate_dir "$W") || { echo "gate: no single engine crate under $W" >&2; exit 2; }
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
  hold "cd '$C' && CARGO_BUILD_JOBS=2 $cmd" > "$D/$T-$name.log" 2>&1
}

cpu_env="CARGO_BUILD_JOBS=2"
cpu_cargo="cargo test --release"
cuda_env="CARGO_BUILD_JOBS=2 RCSR_CUDA_ARCH=$CUDA_ARCH RCSR_REQUIRE_DEVICE=1"
cuda_cargo="cargo test --release --features cuda"
hold_cap=$((LOCK_HOLD_MAX - 20))
n_targets=0
n_failed=0
n_timeouts=0

hold_times() { # <call> <hold start> <end>: wait for the lock and run time of the hold, both in seconds
  if [ -z "$2" ]; then
    echo "wait=$(($3 - $1)) run=none"
  else
    echo "wait=$(($2 - $1)) run=$(($3 - $2))"
  fi
}

# list_targets <build log>: "<name><TAB><cargo args>" per test target of a --no-run --message-format=json build, plus doc
list_targets() {
  grep -a '^{' "$1" | jq -r '
    select(.reason == "compiler-artifact" and .profile.test == true and .executable != null)
    | if (.target.kind | index("lib")) then "lib\t--lib"
      elif (.target.kind | index("bin")) then "bin-" + .target.name + "\t--bin " + .target.name
      elif (.target.kind | index("test")) then .target.name + "\t--test " + .target.name
      else empty end' 2> /dev/null | sort -u
}

# suite <prefix> <env> <cargo cmd> <xid: 0|1>: builds once in the build slot, then runs every target in its own hold.
# Sets s (0, 1 or 70) and n_targets, n_failed, n_timeouts. With xid=1 a new Xid line since the start of a target stops the run (70).
suite() {
  local prefix=$1 env=$2 cargo=$3 xid=$4
  local log=$D/$T-$prefix.log summary=$D/$T-$prefix-summary.log build=$D/$T-$prefix-build.log
  local targets name args tlog rc xb xa res line t0 t1 ts start
  n_targets=0
  n_failed=0
  n_timeouts=0
  : > "$log"
  : > "$summary"
  "$B" bash -c "cd '$C' && $env timeout -k $KILL_AFTER 5400 $cargo --no-run --message-format=json" > "$build" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    cat "$build" >> "$log"
    echo "$prefix build failed rc=$rc" | tee -a "$summary" >> "$log"
    s=1
    return
  fi
  targets=$(list_targets "$build")
  if [ -z "$targets" ]; then
    cat "$build" >> "$log"
    echo "$prefix build lists no test target" | tee -a "$summary" >> "$log"
    s=1
    return
  fi
  targets=$(printf '%s\ndoc\t--doc\n' "$targets")
  s=0
  while IFS=$'\t' read -r name args; do
    n_targets=$((n_targets + 1))
    tlog=$D/$T-$prefix-$name.log
    xs=
    [ "$xid" -eq 0 ] || xs=$(xid_start)
    start=$D/$T-$prefix-$name.start
    rm -f "$start"
    t0=$(now)
    hold "$NOW_CMD > '$start' && cd '$C' && $env timeout -k $KILL_AFTER $hold_cap $cargo --no-fail-fast $args" > "$tlog" 2>&1
    rc=$?
    t1=$(now)
    ts=$(cat "$start" 2> /dev/null)
    rm -f "$start"
    xn=0
    if [ "$xid" -eq 1 ]; then xn=$(xid_new_since "$xs") || xn=check-failed; fi
    res=$(grep -a '^test result:' "$tlog" | tail -n 1)
    if [ "$xid" -eq 1 ]; then
      line="$name rc=$rc xid=new:$xn $(hold_times "$t0" "$ts" "$t1") | ${res:-no test result line}"
    else
      line="$name rc=$rc $(hold_times "$t0" "$ts" "$t1") | ${res:-no test result line}"
    fi
    if [ "$rc" -eq 124 ]; then
      line="$line | timeout"
      n_timeouts=$((n_timeouts + 1))
    fi
    [ "$rc" -eq 0 ] || n_failed=$((n_failed + 1))
    echo "$line" >> "$summary"
    { echo "=== $name ==="; cat "$tlog"; } >> "$log"
    if [ "$xn" != 0 ]; then
      echo "New Xid line(s) or failed Xid check ($xn) during $name; remaining CUDA targets not run" | tee -a "$summary" >> "$log"
      s=70
      return
    fi
  done <<< "$targets"
  [ "$n_failed" -eq 0 ] || s=1
}

suite cpu "$cpu_env" "$cpu_cargo" 0
c=$s
cpu_targets=$n_targets cpu_failed=$n_failed cpu_timeouts=$n_timeouts
suite cuda "$cuda_env" "$cuda_cargo" 1
g=$s
cuda_targets=$n_targets cuda_failed=$n_failed cuda_timeouts=$n_timeouts
step bench "timeout -k $KILL_AFTER 5400 cargo build --release && timeout -k $KILL_AFTER 1800 ./target/release/rcsr-bench --tiny"; b=$?
"$B" bash -c "cd '$W/verksted/_tools/verksted-check' && CARGO_BUILD_JOBS=2 timeout -k $KILL_AFTER 3000 cargo test --release && CARGO_BUILD_JOBS=2 timeout -k $KILL_AFTER 1800 cargo run --release -- links --external '$W/verksted'" > "$D/$T-verksted.log" 2>&1; v=$?
bad=0
for k in cpu cuda; do
  f=$D/$T-$k.log
  ok=$(grep -ac 'test result: ok' "$f")
  failed=$(grep -aEc 'test result: FAILED|^error(\[|:)' "$f")
  echo "$k: ok=$ok failed=$failed"
  [ "$failed" -eq 0 ] || bad=1
done
echo "cpu: targets=$cpu_targets failed=$cpu_failed timeouts=$cpu_timeouts"
echo "cuda: targets=$cuda_targets failed=$cuda_failed timeouts=$cuda_timeouts"
rec=$(grep -a -E '^recovered [0-9]+/[0-9]+' "$D/$T-bench.log" | tail -n 1)
echo "bench: ${rec:-no recovered line}"
echo "exit cpu=$c cuda=$g bench=$b verksted=$v"
echo "tested: $(git -C "$W" rev-parse HEAD)"
case "$rec" in *"recovered 27/27"*) ;; *) bad=1 ;; esac
[ "$g" -ne 70 ] || exit 70
[ "$c" -eq 0 ] && [ "$g" -eq 0 ] && [ "$b" -eq 0 ] && [ "$v" -eq 0 ] && [ "$bad" -eq 0 ]
