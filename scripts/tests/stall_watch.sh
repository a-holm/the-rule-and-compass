#!/usr/bin/env bash
# Self-test for scripts/stall-watch.sh (issue #353), about 2 minutes.
#
# Needs the helper built first (through the machine lock):
#   cd my-version && cargo build --release --example hung
# Run it outside cargo: scripts/tests/stall_watch.sh

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HUNG="${HUNG_BIN:-$ROOT/my-version/target/release/examples/hung}"
[[ -x "$HUNG" ]] || { echo "missing $HUNG" >&2; exit 2; }

dir="$(mktemp -d)"
trap 'c="$(pgrep -P "$wpid" 2>/dev/null)"; kill "$wpid" $c 2>/dev/null; rm -rf "$dir"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

"$ROOT/scripts/stall-watch.sh" -i 10 -m 1 -l "$dir/run" -- "$HUNG" 2>"$dir/watch.err" &
wpid=$!

for _ in $(seq 1 75); do
    [[ -s "$dir/run.stacks.txt" ]] && break
    sleep 2
done
[[ -s "$dir/run.stacks.txt" ]] || fail "no stacks file within M + 1 minutes"
grep -q 'recv' "$dir/run.stacks.txt" || fail "backtrace does not name recv"
grep -q 'stacks.sh exit 0' "$dir/run.stacks.txt" || fail "tracer path did not exit 0"
kill -0 "$wpid" 2>/dev/null || fail "watchdog exited"
pgrep -P "$wpid" >/dev/null || fail "watched command was killed"
echo "PASS: stacks written, command still alive"
