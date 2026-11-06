#!/usr/bin/env bash
# Usage: lock-run.sh <lockfile> <command...>
# Runs <command>, then ends the processes it left holding the test lock, by pid. Never pkill or killall, no group signals.
# Acts only when this process owns an flock on <lockfile> and is its sole holder at start (the lock script exec'd into it).
# A holder has an fd whose link text is the lock path and whose /proc fdinfo shows this process's own lock line (locker pid
# and dev:ino), so it shares this hold's open file description. When the own line is gone or changes (the file was replaced,
# the flock was released), nothing more is signalled. A waiter or a holder of another file or inode is never touched.
# TERM, INT and HUP sent to lock-run are forwarded to the command; the reap still runs. Order: TERM to each holder,
# RCSR_REAP_TERM_WAITS (50) counted waits of 0.1 s, then KILL, RCSR_REAP_KILL_WAITS (50) counted waits.
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
lock_dir=$(cd "$(dirname "$lock")" 2> /dev/null && pwd -P) || lock_dir=$(dirname "$lock")
lock_real=$lock_dir/$(basename "$lock")

LL=
lockline() { # <pid> <fd>: sets LL to the flock line of that fd without its list index; returns 1 when there is none
  local line
  LL=
  {
    while IFS= read -r line; do
      case $line in lock:*FLOCK*) LL=${line#*: }; return 0 ;; esac
    done < "/proc/$1/fdinfo/$2"
  } 2> /dev/null
  return 1
}

FDS=()
own_fd=
own_line=
fd_matches() { # FDS = /proc/<pid>/fd/<n> links whose text is the lock path (no stat of any target)
  FDS=()
  mapfile -t FDS < <(eval "find /proc/[0-9]*/fd -maxdepth 1 -lname \"\$lock_real\" 2> /dev/null ${own_fd:+$own_fd>&-}")
}

HOLD=()
LOST=0
scan() { # HOLD = pids sharing this hold's lock line; LOST=1 when our own line is gone or changed
  local f p n own_seen=0
  HOLD=()
  LOST=0
  fd_matches
  for f in "${FDS[@]}"; do
    p=${f#/proc/}
    p=${p%%/*}
    n=${f##*/}
    if [ "$p" = "$$" ]; then
      [ "$n" = "$own_fd" ] && lockline "$p" "$n" && [ "$LL" = "$own_line" ] && own_seen=1
      continue
    fi
    [ "$p" = "$BASHPID" ] && continue
    lockline "$p" "$n" && [ "$LL" = "$own_line" ] && HOLD+=("$p")
  done
  [ "$own_seen" -eq 1 ] || { LOST=1; HOLD=(); }
}

fd_matches
for f in "${FDS[@]}"; do
  p=${f#/proc/}
  p=${p%%/*}
  [ "$p" = "$$" ] || continue
  lockline "$p" "${f##*/}" && { own_fd=${f##*/}; own_line=$LL; break; }
done
if [ -z "$own_fd" ]; then
  echo "lock-run: no flock held by this process; nothing will be signalled" >&2
  exec "$@"
fi
scan
if [ "$LOST" -eq 1 ] || [ ${#HOLD[@]} -gt 0 ]; then
  echo "lock-run: this process is not the sole holder of the flock (${HOLD[*]:-own line unreadable}); nothing will be signalled" >&2
  exec "$@"
fi

cpid=
forward() { [ -z "$cpid" ] || kill "-$1" "$cpid" 2> /dev/null; }
trap 'forward TERM' TERM
trap 'forward INT' INT
trap 'forward HUP' HUP
setsid "$@" <&0 &
cpid=$!
wait "$cpid"
rc=$?
while kill -0 "$cpid" 2> /dev/null; do
  wait "$cpid"
  rc=$?
done

lost_exit() {
  echo "lock-run: ownership of the flock is gone (the lock file was replaced or the flock was released); nothing more is signalled" >&2
  exit "$rc"
}
signal_all() { local p; for p in "${HOLD[@]}"; do kill "-$1" "$p" 2> /dev/null; done; }
wait_free() { # <count>: returns 0 once no process holds the lock, 1 after <count> waits
  local i
  for ((i = 0; i < $1; i++)); do
    scan
    [ "$LOST" -eq 0 ] || lost_exit
    [ ${#HOLD[@]} -eq 0 ] && return 0
    sleep 0.1
  done
  scan
  [ "$LOST" -eq 0 ] || lost_exit
  [ ${#HOLD[@]} -eq 0 ]
}

scan
[ "$LOST" -eq 0 ] || lost_exit
if [ ${#HOLD[@]} -gt 0 ]; then
  echo "lock-run: ended pids still holding the lock: ${HOLD[*]}" >&2
  signal_all TERM
  if ! wait_free "$term_waits"; then
    echo "lock-run: KILL after $term_waits waits: ${HOLD[*]}" >&2
    signal_all KILL
    wait_free "$kill_waits" || true
  fi
  if [ ${#HOLD[@]} -gt 0 ]; then
    echo "lock-run: pids still hold the lock after KILL: ${HOLD[*]}" >&2
    exit 71
  fi
fi
exit "$rc"
