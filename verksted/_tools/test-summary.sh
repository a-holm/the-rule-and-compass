#!/usr/bin/env bash
# Usage: test-summary.sh <log>...
# Prints one block per log:
#   == <log>
#   tests: passed=<n> failed=<n> ignored=<n> results=<ok>/<total>
#   warnings: <n>
#   errors: <n>
#   failing: <names, comma-separated> | none
#   sanitizer: <every ERROR SUMMARY line, joined> | none
#   xid: <the last "xid before" line> | none
#   rc: <n from the last RC_DONE line> | none
#   verdict: ok | fail
# Fail when failed > 0, errors > 0, an ERROR SUMMARY has a non-zero count, xid after > before,
# or rc is present and not 0. A red proof reads failed>0 with errors: 0.
# Writes: nothing.
# Exit: 0 when every block says ok; 1 otherwise; 2 usage or a missing log.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

case "${1:-}" in
  -h | --help) usage; exit 0 ;;
  '') usage >&2; exit 2 ;;
esac
for f in "$@"; do
  if [ ! -f "$f" ]; then
    echo "test-summary.sh: no such log: $f" >&2
    exit 2
  fi
done

summarize() {
  awk -v name="$1" '
    function num(s, key,    m) {
      if (match(s, "[0-9]+ " key)) { m = substr(s, RSTART, RLENGTH); sub(/ .*/, "", m); return m + 0 }
      return 0
    }
    /^test result:/ {
      total++
      if ($0 ~ /^test result: ok/) okc++
      passed += num($0, "passed"); failed += num($0, "failed"); ignored += num($0, "ignored")
    }
    /^warning: / && !/^warning: .* generated [0-9]+ warnings?/ { warnings++ }
    /^error(\[E[0-9]+\])?:/ && !/^error: test failed/ { errors++ }
    /^test .* \.\.\. FAILED$/ {
      n = $0; sub(/^test /, "", n); sub(/ \.\.\. FAILED$/, "", n)
      failing = failing (failing == "" ? "" : ",") n
    }
    /ERROR SUMMARY/ {
      s = $0; sub(/^[ =]*/, "", s)
      san = san (san == "" ? "" : "; ") s
      if (match($0, /ERROR SUMMARY: *[0-9]+/)) {
        c = substr($0, RSTART, RLENGTH); sub(/[^0-9]*/, "", c)
        if (c + 0 > 0) bad = 1
      }
    }
    /^xid before=[0-9]+ after=[0-9]+/ {
      xid = $0
      b = $2; a = $3; sub(/before=/, "", b); sub(/after=/, "", a)
      xbad = (a + 0 > b + 0)
    }
    /^RC_DONE rc=[0-9]+/ { rc = $2; sub(/rc=/, "", rc) }
    END {
      verdict = "ok"
      if (failed > 0 || errors > 0 || bad || xbad || (rc != "" && rc != 0)) verdict = "fail"
      print "== " name
      printf "tests: passed=%d failed=%d ignored=%d results=%d/%d\n", passed, failed, ignored, okc, total
      printf "warnings: %d\n", warnings
      printf "errors: %d\n", errors
      print "failing: " (failing == "" ? "none" : failing)
      print "sanitizer: " (san == "" ? "none" : san)
      print "xid: " (xid == "" ? "none" : xid)
      print "rc: " (rc == "" ? "none" : rc)
      print "verdict: " verdict
    }
  ' "$1"
}

status=0
first=1
for f in "$@"; do
  [ "$first" -eq 1 ] || echo
  first=0
  out=$(summarize "$f")
  printf '%s\n' "$out"
  case "$out" in *$'\nverdict: ok') ;; *) status=1 ;; esac
done
exit $status
