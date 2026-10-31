#!/usr/bin/env bash
# Usage: check.sh <output-dir> '<check command>'
# Runs the command with bash -c and writes <output-dir>/check.md: line 1 is `pass` or
# `fail: exit <code>: <last output line>`, then the command and the exit code.
set -u
if [ $# -ne 2 ]; then
  echo "usage: check.sh <output-dir> '<check command>'" >&2
  exit 2
fi
dir=$1
cmd=$2
mkdir -p "$dir"
out=$(bash -c "$cmd" 2>&1)
code=$?
if [ "$code" -eq 0 ]; then
  head="pass"
else
  last=$(printf '%s\n' "$out" | grep . | tail -n 1 | cut -c1-200)
  head="fail: exit $code: ${last:-no output}"
fi
{
  printf '%s\n\n- command: `%s`\n- exit: %s\n' "$head" "$cmd" "$code"
} > "$dir/check.md"
printf '%s\n' "$head"
exit "$code"
