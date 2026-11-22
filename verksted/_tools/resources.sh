#!/usr/bin/env bash
# Usage: resources.sh
# Read-only resource check before a worktree is created or a build starts (rules 64 and 65). Reads the keys
# BUILD_VOLUME, DISK_MIN_FREE and BUILD_SLOTS from $VLOCAL/machine.md (lines `- KEY: value`; `_shared/local.md`), then prints:
#   disk:   free space on BUILD_VOLUME against DISK_MIN_FREE (a number with an optional unit B, KB, MB, GB, TB, KiB, MiB, GiB, TiB)
#   builds: build-slot holders against BUILD_SLOTS
#   cargo:  cargo processes of this user (none expected before a build starts)
# Each line starts with ok or FAIL and holds the measured value. No default stands in for a missing key.
# RCSR_FREE_CMD (a command that prints free bytes of the volume given as its argument) and RCSR_PS_CMD (prints one command name per line)
# replace df and ps; tests use them.
# Writes: nothing.
# Exit: 0 every check ok; 1 a check failed; 2 usage, machine.md missing, or a key missing or unreadable (the message names it).
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

case ${1:-} in
  -h | --help) usage; exit 0 ;;
  '') ;;
  *) usage >&2; exit 2 ;;
esac

min=$(machine_key DISK_MIN_FREE) || exit 2
vol=$(machine_key BUILD_VOLUME) || exit 2
slots=$(machine_key BUILD_SLOTS) || exit 2
case $slots in '' | *[!0-9]*) echo "resources.sh: key BUILD_SLOTS is not a number: $slots" >&2; exit 2 ;; esac
min_bytes=$(printf '%s\n' "$min" | awk '
  match($0, /^[0-9]+(\.[0-9]+)?/) {
    n = substr($0, 1, RLENGTH); u = substr($0, RLENGTH + 1); gsub(/^ +| +$/, "", u)
    split("B 1 KB 1000 MB 1000000 GB 1000000000 TB 1000000000000 KiB 1024 MiB 1048576 GiB 1073741824 TiB 1099511627776", t, " ")
    for (i = 1; i < 18; i += 2) if (t[i] == (u == "" ? "B" : u)) { printf "%.0f\n", n * t[i + 1]; exit }
  }')
[ -n "$min_bytes" ] || { echo "resources.sh: key DISK_MIN_FREE is not a size: $min" >&2; exit 2; }

rc=0
if [ -n "${RCSR_FREE_CMD:-}" ]; then
  free=$(bash -c "$RCSR_FREE_CMD \"\$1\"" _ "$vol")
else
  free=$(df --output=avail -B1 "$vol" 2> /dev/null | tail -n 1 | tr -d ' ')
fi
case $free in
  '' | *[!0-9]*) echo "FAIL disk: free space of $vol unreadable"; rc=1 ;;
  *)
    if [ "$free" -ge "$min_bytes" ]; then
      echo "ok disk: free=$free bytes on $vol, DISK_MIN_FREE=$min_bytes bytes"
    else
      echo "FAIL disk: free=$free bytes on $vol, below DISK_MIN_FREE=$min_bytes bytes"
      rc=1
    fi ;;
esac

held=$(find /proc/[0-9]*/fd -maxdepth 1 -lname "$SLOTDIR/slot*.lock" -printf '%h\n' 2> /dev/null | cut -d/ -f3 | sort -un | wc -l | tr -d ' ')
if [ "$held" -le "$slots" ]; then
  echo "ok builds: running=$held, BUILD_SLOTS=$slots"
else
  echo "FAIL builds: running=$held, above BUILD_SLOTS=$slots"
  rc=1
fi

if [ -n "${RCSR_PS_CMD:-}" ]; then
  names=$(bash -c "$RCSR_PS_CMD")
else
  names=$(ps -u "$(id -u)" -o comm= 2> /dev/null)
fi
cargos=$(printf '%s\n' "$names" | grep -cx cargo)
if [ "$cargos" -eq 0 ]; then
  echo "ok cargo: running=0"
else
  echo "FAIL cargo: running=$cargos, expected 0 before a build starts"
  rc=1
fi
exit "$rc"
