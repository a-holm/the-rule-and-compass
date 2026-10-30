#!/usr/bin/env bash
# Run a command and dump thread backtraces when it stalls (issue #353).
#
# Usage: scripts/stall-watch.sh [-i SECONDS] [-m MINUTES] [-l LOG] -- <command...>
#
#   -i  sampling interval, default 60 s
#   -m  minutes without progress before a dump, default 10
#   -l  log prefix, default stall-watch-<timestamp>; stacks go to <LOG>.stacks.txt
#
# The command runs with RCSR_ATTACHABLE=1. Every sample reads CPU ticks and
# /proc/<pid>/io of the command and each descendant whose executable lives
# under a target/ directory. A pid with an unchanged sample for MINUTES gets one
# stacks.sh dump.
# The watchdog never kills anything and exits with the command's status.

set -u

INTERVAL=60
MINUTES=10
LOG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i) INTERVAL="$2"; shift 2 ;;
        -m) MINUTES="$2"; shift 2 ;;
        -l) LOG="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "usage: $0 [-i SECONDS] [-m MINUTES] [-l LOG] -- <command...>" >&2; exit 2 ;;
    esac
done
if [[ $# -eq 0 ]]; then
    echo "usage: $0 [-i SECONDS] [-m MINUTES] [-l LOG] -- <command...>" >&2
    exit 2
fi
LOG="${LOG:-stall-watch-$(date +%Y%m%d-%H%M%S)}"
STACKS_OUT="$LOG.stacks.txt"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIMIT=$((MINUTES * 60))

descendants() {
    local p c
    for c in $(pgrep -P "$1" 2>/dev/null); do
        echo "$c"
        descendants "$c"
    done
}

signature() {
    local pid="$1" stat io
    stat="$(cut -d' ' -f14,15 "/proc/$pid/stat" 2>/dev/null)" || return 1
    io="$(grep -E '^(rchar|wchar)' "/proc/$pid/io" 2>/dev/null | tr '\n' ' ')"
    echo "$stat $io"
}

RCSR_ATTACHABLE=1 "$@" &
CMD=$!

declare -A last_sig last_change dumped
while kill -0 "$CMD" 2>/dev/null; do
    sleep "$INTERVAL" &
    wait $! 2>/dev/null
    now=$(date +%s)
    for pid in "$CMD" $(descendants "$CMD"); do
        exe="$(readlink "/proc/$pid/exe" 2>/dev/null)" || continue
        [[ "$exe" == */target/* ]] || continue
        sig="$(signature "$pid")" || continue
        if [[ "${last_sig[$pid]:-}" != "$sig" ]]; then
            last_sig[$pid]="$sig"
            last_change[$pid]=$now
            unset 'dumped[$pid]'
        elif [[ -z "${dumped[$pid]:-}" && $((now - last_change[$pid])) -ge $LIMIT ]]; then
            dumped[$pid]=1
            {
                echo "=== pid $pid ($exe) no progress for ${MINUTES} min, $(date -Is)"
                "$HERE/stacks.sh" "$pid"
                echo "=== stacks.sh exit $?"
            } >>"$STACKS_OUT" 2>&1
            echo "stall-watch: stacks for pid $pid in $STACKS_OUT" >&2
        fi
    done
done
wait "$CMD"
