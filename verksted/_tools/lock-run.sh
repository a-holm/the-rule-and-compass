#!/usr/bin/env bash
# Usage: lock-run.sh <lockfile> <command...>
# Runs <command> in its own session, waits for it, then ends every process of its tree that still holds the lock <lockfile>.
# Runs inside the lock hold (rc-run.sh puts it between the lock script and the command), so a process that holds the lock
# after the command ended belongs to this tree: the lock is exclusive. A holder is a process with an fd on <lockfile> whose
# open file description owns the flock; a process that only has the file open (a waiter) is never touched.
# Order: TERM to the command's process group and to each holder by pid; RCSR_REAP_TERM_WAITS (default 50) counted waits of
# 0.1 s; KILL to the group and to each holder still left; RCSR_REAP_KILL_WAITS (default 50) counted waits.
# Never uses pkill or killall. Writes: nothing but messages on stderr.
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

holders() {
  local d p fd n line
  for d in /proc/[0-9]*/fd; do
    p=${d#/proc/}
    p=${p%/fd}
    [ "$p" = "$$" ] || [ "$p" = "$BASHPID" ] && continue
    for fd in "$d"/*; do
      [ "$fd" -ef "$lock" ] || continue
      n=${fd##*/}
      {
        while IFS= read -r line; do
          case $line in lock:*FLOCK*) echo "$p"; break 2 ;; esac
        done < "/proc/$p/fdinfo/$n"
      } 2> /dev/null
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
