#!/usr/bin/env bash
# Usage: gate.sh <worktree> <tag>
# The full gate on an integration tip: CPU suite, CUDA suite, tiny benchmark (27/27).
# Each step is its own lock hold with an outer timeout. Logs: $G/<tag>-{cpu,cuda,bench}.log.
# Exit 0 only when every suite passes and the benchmark recovers 27/27.
set -u
if [ $# -ne 2 ]; then
  echo "usage: gate.sh <worktree> <tag>" >&2
  exit 2
fi
W=$1
T=$2
L=${RCSR_LOCK_SCRIPT:-/mnt/lager/tmp/opencode/one-full-suite.sh}
D=${RCSR_GATE_DIR:-/mnt/lager/tmp/eureqa-gates}
mkdir -p "$D"
step() {
  local name=$1 cmd=$2
  "$L" bash -c "cd '$W/my-version' && CARGO_BUILD_JOBS=2 $cmd" > "$D/$T-$name.log" 2>&1
}
step cpu "timeout 5400 cargo test --release"; c=$?
step cuda "RCSR_CUDA_ARCH=sm_86 timeout 5400 cargo test --release --features cuda"; g=$?
step bench "timeout 5400 cargo build --release && timeout 1800 ./target/release/rcsr-bench --tiny"; b=$?
bad=0
for k in cpu cuda; do
  f=$D/$T-$k.log
  ok=$(grep -c 'test result: ok' "$f")
  failed=$(grep -cE 'test result: FAILED|^error(\[|:)' "$f")
  echo "$k: ok=$ok failed=$failed"
  [ "$failed" -eq 0 ] || bad=1
done
rec=$(grep -E 'recovered [0-9]+/[0-9]+' "$D/$T-bench.log" | tail -n 1)
echo "bench: ${rec:-no recovered line}"
echo "exit cpu=$c cuda=$g bench=$b"
echo "tested: $(git -C "$W" rev-parse HEAD)"
case "$rec" in *"recovered 27/27"*) ;; *) bad=1 ;; esac
[ "$c" -eq 0 ] && [ "$g" -eq 0 ] && [ "$b" -eq 0 ] && [ "$bad" -eq 0 ]
