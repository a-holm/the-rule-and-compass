#!/usr/bin/env bash
# Sourced by verksted/_tools/tests/test-*.sh and by the stub scripts they write; not run.
# Bounded FIFO helpers: no open and no read waits for a partner that never comes.
# A FIFO holds data only while some process has it open. The test body keeps one read-write fd per FIFO (fifo_make), so a
# line written by a stub before the reader opens its end is not lost.
# Reads and writes open the FIFO read-write, which never blocks on a missing partner; reads end after FIFO_WAIT seconds (default 30).

# fifo_make <path>...: mkfifo each path and keep it open read-write in this shell.
fifo_make() {
  local p fd
  for p in "$@"; do
    rm -f "$p"
    mkfifo "$p" || return 1
    exec {fd}<> "$p" || return 1
  done
}

# fifo_line <fifo> [seconds]: reads one line into REPLY. Returns 0, or non-zero (above 128 on timeout) when no line came.
fifo_line() {
  local fifo=$1 secs=${2:-${FIFO_WAIT:-30}} fd rc
  REPLY=
  exec {fd}<> "$fifo" || return 1
  read -t "$secs" -r -u "$fd" REPLY
  rc=$?
  exec {fd}<&-
  return "$rc"
}

# fifo_put <fifo> <line>: writes one line without waiting for a reader.
fifo_put() { printf '%s\n' "$2" 1<> "$1"; }

# fifo_read <fifo> <name> [seconds]: fifo_line for a test body; a missing line is the check failure "name".
fifo_read() {
  fifo_line "$1" "${3:-}" && return 0
  echo "FAIL $2: no line from $1 within ${3:-${FIFO_WAIT:-30}}s"
  fails=$((fails + 1))
  return 1
}

# wall_clock_guard "$@": call first in a test file. Runs the file again under timeout -k 10 $TEST_WALL_SECONDS (default 300)
# in its own process group; exit 124 and a line on stderr when the limit is hit.
wall_clock_guard() {
  [ -z "${TEST_WALL_GUARD:-}" ] || return 0
  local self=${BASH_SOURCE[1]} rc=0
  TEST_WALL_GUARD=1 timeout -k 10 "${TEST_WALL_SECONDS:-300}" bash "$self" "$@" || rc=$?
  [ "$rc" -ne 124 ] || echo "$(basename "$self"): wall-clock limit exceeded" >&2
  exit "$rc"
}
