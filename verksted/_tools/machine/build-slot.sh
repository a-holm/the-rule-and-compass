#!/usr/bin/env bash
# Up to RCSR_BUILD_SLOTS concurrent builds machine-wide (default 3), at the lowest
# CPU and I/O priority, and only while enough memory is free. Builds use the CPU
# only; tests and benchmarks use one-full-suite.sh, one at a time.
set -euo pipefail
DIR="${RCSR_BUILD_SLOT_DIR:-/mnt/lager/tmp/opencode/build-slots}"
PAUSE="${RCSR_TEST_PAUSE:-/mnt/lager/tmp/opencode/PAUSE}"
MIN_KB="${RCSR_BUILD_MIN_AVAIL_KB:-8388608}"
mkdir -p "$DIR"
if [ -e "$PAUSE" ]; then
  echo "[build-slot] paused while $PAUSE exists; waiting" >&2
  while [ -e "$PAUSE" ]; do sleep 30; done
fi
deadline=$(( $(date +%s) + 7200 ))
while :; do
  avail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  # The desktop has priority: while it streams video, allow fewer concurrent builds.
  if /mnt/lager/tmp/opencode/desktop-busy.sh; then N="${RCSR_BUILD_SLOTS_DISCORD:-1}"; else N="${RCSR_BUILD_SLOTS:-3}"; fi
  if [ "$avail" -ge "$MIN_KB" ]; then
    for i in $(seq 1 "$N"); do
      exec 8>"$DIR/slot$i.lock"
      if flock -n 8; then
        echo "[build-slot] slot $i/$N held: $*" >&2
        echo 19 > /proc/self/autogroup 2>/dev/null || true
        # While the desktop streams video, keep cores 0 and 1 (and their hyperthreads) free.
        if /mnt/lager/tmp/opencode/desktop-busy.sh; then cpus="${RCSR_CPUS_DISCORD:-2,3,6,7}"; else cpus="${RCSR_CPUS:-0-7}"; fi
        export RUST_TEST_THREADS="${RUST_TEST_THREADS:-4}"
        exec taskset -c "$cpus" chrt -i 0 nice -n 19 ionice -c 3 "$@"
      fi
      exec 8>&-
    done
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then echo "[build-slot] gave up waiting" >&2; exit 75; fi
  sleep 5
done
