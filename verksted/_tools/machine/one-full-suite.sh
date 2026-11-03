#!/usr/bin/env bash
# One full-suite build/test at a time, machine-wide, at the lowest CPU and I/O
# priority (SCHED_IDLE) so the desktop always wins.
# Usage: one-full-suite.sh bash -c 'cd <worktree>/my-version && cargo test --release'
# While the pause file exists, new runs wait before queueing for the lock.
set -euo pipefail
LOCK="${RCSR_TEST_LOCK:-/mnt/lager/tmp/opencode/full-suite.lock}"
PAUSE="${RCSR_TEST_PAUSE:-/mnt/lager/tmp/opencode/PAUSE}"
mkdir -p "$(dirname "$LOCK")"
if [ -e "$PAUSE" ]; then
  echo "[one-full-suite] paused while $PAUSE exists; waiting" >&2
  while [ -e "$PAUSE" ]; do sleep 30; done
fi
exec 9>"$LOCK"
if ! flock -w 7200 9; then
  echo "[one-full-suite] gave up waiting for $LOCK" >&2
  exit 75
fi
echo "[one-full-suite] lock held: $*" >&2
echo 19 > /proc/self/autogroup 2>/dev/null || true
# While the desktop streams video, keep cores 0 and 1 (and their hyperthreads) free.
if /mnt/lager/tmp/opencode/desktop-busy.sh; then cpus="${RCSR_CPUS_DISCORD:-2,3,6,7}"; else cpus="${RCSR_CPUS:-0-7}"; fi
export RUST_TEST_THREADS="${RUST_TEST_THREADS:-4}"
exec taskset -c "$cpus" chrt -i 0 nice -n 19 ionice -c 3 "$@"
