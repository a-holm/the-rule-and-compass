#!/usr/bin/env bash
# Usage: lock-run.sh <lockfile> <command...>
# Runs <command> in its own session, waits for it, then ends every process of its tree that still holds the test lock.
# It runs inside a lock hold: the lock script (one-full-suite.sh) took the flock on an fd that this process inherited.
# <lockfile> is the path the lock script locks (rc-run.sh passes its LOCKFILE). This process must itself own an flock on that
# file through an fd it inherited; an flock on any other file (the build slot, another lock file) does not count. When it
# owns none, nothing is signalled and the command runs as usual.
# It also acts only when it is the sole holder at start: every process that holds the flock after the command ended was then
# started by the command. A lock hold that an outer process shares with others (a test run under the lock that itself calls
# rc-run.sh) is never reaped.
# A holder is a process with an fd whose fdinfo shows that same flock. The kernel prints a lock only on the open file
# description that owns it, so every holder shares the description lock-run.sh inherited: it was inherited from this
# hold. A process that only has the lock file open (a waiter, or a holder of another lock file) shows no flock and is
# never touched.
# Order: TERM to the command's process group and to each holder by pid; RCSR_REAP_TERM_WAITS (default 50) counted waits of
# 0.1 s; KILL to the group and to each holder still left; RCSR_REAP_KILL_WAITS (default 50) counted waits.
# Never uses pkill or killall.
# Writes: nothing but messages on stderr.
# Exit: the command's exit code; 71 when a process still holds the lock after the KILL; 2 usage.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac
[ $# -ge 2 ] || { usage >&2; exit 2; }
lock=$1
shift
term_waits=${RCSR_REAP_TERM_WAITS:-50}
kill_waits=${RCSR_REAP_KILL_WAITS:-50}

# owns_flock <pid> <fd>: true when that fd's open file description owns an flock
owns_flock() {
  local line
  {
    while IFS= read -r line; do
      case $line in lock:*FLOCK*) return 0 ;; esac
    done < "/proc/$1/fdinfo/$2"
  } 2> /dev/null
  return 1
}

own_lock_fd() {
  local fd
  for fd in /proc/$$/fd/*; do
    [ "$fd" -ef "$lock" ] && owns_flock "$$" "${fd##*/}" && { echo "$fd"; return 0; }
  done
  return 1
}

holders() {
  local d p fd
  for d in /proc/[0-9]*/fd; do
    p=${d#/proc/}
    p=${p%/fd}
    [ "$p" = "$$" ] || [ "$p" = "$BASHPID" ] && continue
    for fd in "$d"/*; do
      [ "$fd" -ef "$lock" ] || continue
      owns_flock "$p" "${fd##*/}" && { echo "$p"; break; }
    done
  done
}

wait_free() { # <count>: returns 0 once nothing holds the lock, 1 after <count> waits
  local i
  for ((i = 0; i < $1; i++)); do
    [ -z "$(holders)" ] && return 0
    sleep 0.1
  done
  [ -z "$(holders)" ]
}

lockfd=$(own_lock_fd) || lockfd=
if [ -z "$lockfd" ]; then
  echo "lock-run: no flock held by this process; nothing will be signalled" >&2
  exec "$@"
fi

shared=$(holders)
if [ -n "$shared" ]; then
  echo "lock-run: other processes already share this flock ($(echo $shared)); the hold is not this process alone, nothing will be signalled" >&2
  exec "$@"
fi
setsid "$@" <&0 &
pgid=$!
wait "$pgid"
rc=$?

left=$(holders)
if [ -n "$left" ]; then
  echo "lock-run: ended pids still holding the lock: $(echo $left)" >&2
  kill -TERM -- "-$pgid" 2> /dev/null
  for p in $left; do kill -TERM "$p" 2> /dev/null; done
  if ! wait_free "$term_waits"; then
    echo "lock-run: KILL after $term_waits waits: $(echo $(holders))" >&2
    kill -KILL -- "-$pgid" 2> /dev/null
    for p in $(holders); do kill -KILL "$p" 2> /dev/null; done
    wait_free "$kill_waits" || true
  fi
  left=$(holders)
  if [ -n "$left" ]; then
    echo "lock-run: pids still hold the lock after KILL: $(echo $left)" >&2
    exit 71
  fi
fi
exit "$rc"
