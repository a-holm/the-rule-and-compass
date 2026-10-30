#!/usr/bin/env bash
# Dump all thread backtraces of a pid (issue #353).
#
# Usage: scripts/stacks.sh <pid> [out-file]
#
# Exit code says which path worked:
#   0  gdb backtraces
#   3  eu-stack backtraces
#   4  /proc fallback (comm, wchan, stat per thread); both tracers failed
#   2  usage error
#
# Under Yama ptrace_scope=1 the target must be a descendant of the caller or
# run with RCSR_ATTACHABLE=1 (see docs/issue-workflow.md).

set -u

PID="${1:-}"
OUT="${2:-/dev/stdout}"
TRACE_TIMEOUT="${STACKS_TIMEOUT:-60}"

if [[ ! "$PID" =~ ^[0-9]+$ || ! -d "/proc/$PID" ]]; then
    echo "usage: $0 <pid> [out-file]" >&2
    exit 2
fi

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

has_frames() { grep -q '^#0 ' "$1"; }

if command -v gdb >/dev/null 2>&1; then
    timeout "$TRACE_TIMEOUT" gdb -batch -p "$PID" -ex 'set pagination off' \
        -ex 'thread apply all bt' >"$tmp" 2>&1
    if has_frames "$tmp"; then
        cat "$tmp" >"$OUT"
        exit 0
    fi
fi

if command -v eu-stack >/dev/null 2>&1; then
    timeout "$TRACE_TIMEOUT" eu-stack -p "$PID" >"$tmp" 2>&1
    if has_frames "$tmp"; then
        cat "$tmp" >"$OUT"
        exit 3
    fi
fi

{
    echo "# no tracer worked for pid $PID; /proc table"
    printf 'tid\tcomm\twchan\tstat\n'
    for t in /proc/"$PID"/task/*; do
        printf '%s\t%s\t%s\t%s\n' "${t##*/}" "$(cat "$t/comm" 2>/dev/null)" \
            "$(cat "$t/wchan" 2>/dev/null)" "$(cat "$t/stat" 2>/dev/null)"
    done
} >"$OUT"
exit 4
