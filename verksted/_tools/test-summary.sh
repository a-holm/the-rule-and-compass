#!/usr/bin/env bash
# Usage: test-summary.sh [--expect <name>]... <log>...
# Writes: nothing.
# Prints one block per log:
#   == <log>
#   tests: passed=<n> failed=<n> ignored=<n> results=<ok>/<total> ran=<passed+failed>
#   warnings: <n>
#   errors: <n>
#   failing: <names, comma-separated> | none
#   sanitizer: <every ERROR SUMMARY line, joined> | none
#   xid: <the last "xid new" line> | none
#   rc: <n from the last RC_DONE line> | none
#   not run: <name>   (one line per --expect name without its own `test <name> ... ok` line; a path prefix `mod::` is ignored)
#   verdict: ok | empty | fail
# Fail when failed > 0, errors > 0, an ERROR SUMMARY has a non-zero count, xid new > 0,
# rc is present and not 0, or an --expect name did not run. Empty when the log holds `test result:` lines
# and ran is 0 (every test ignored or filtered out): rule 70. A red proof reads failed>0 with errors: 0.
# --expect applies to every log and may repeat; each takes one name.
# Exit: 0 when every block says ok; 1 otherwise; 2 usage or a missing log.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

expect=
while [ $# -gt 0 ]; do
  case "$1" in
    -h | --help) usage; exit 0 ;;
    --expect)
      [ $# -ge 2 ] || { usage >&2; exit 2; }
      expect="$expect$2"$'\n'
      shift 2 ;;
    *) break ;;
  esac
done
[ $# -gt 0 ] || { usage >&2; exit 2; }
for f in "$@"; do
  if [ ! -f "$f" ]; then
    echo "test-summary.sh: no such log: $f" >&2
    exit 2
  fi
done

summarize() {
  awk -v name="$1" -v expect="$expect" '
    function num(s, key,    m) {
      if (match(s, "[0-9]+ " key)) { m = substr(s, RSTART, RLENGTH); sub(/ .*/, "", m); return m + 0 }
      return 0
    }
    /^test result:/ {
      total++
      if ($0 ~ /^test result: ok/) okc++
      passed += num($0, "passed"); failed += num($0, "failed"); ignored += num($0, "ignored")
    }
    /^test .* \.\.\. ok$/ {
      n = $0; sub(/^test /, "", n); sub(/ \.\.\. ok$/, "", n)
      oks[n] = 1
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
    /^xid new=[0-9]+ since=/ {
      xid = $0
      n = $2; sub(/new=/, "", n)
      xbad = (n + 0 > 0)
    }
    /^RC_DONE rc=[0-9]+/ { rc = $2; sub(/rc=/, "", rc) }
    function ran_ok(want,    k, tail) {
      if (want in oks) return 1
      tail = "::" want
      for (k in oks) if (length(k) > length(tail) && substr(k, length(k) - length(tail) + 1) == tail) return 1
      return 0
    }
    END {
      ran = passed + failed
      missing = ""
      nexp = split(expect, want, "\n")
      for (i = 1; i <= nexp; i++) if (want[i] != "" && !ran_ok(want[i])) missing = missing "not run: " want[i] "\n"
      verdict = "ok"
      if (total > 0 && ran == 0) verdict = "empty"
      if (failed > 0 || errors > 0 || bad || xbad || (rc != "" && rc != 0) || missing != "") verdict = "fail"
      print "== " name
      printf "tests: passed=%d failed=%d ignored=%d results=%d/%d ran=%d\n", passed, failed, ignored, okc, total, ran
      printf "warnings: %d\n", warnings
      printf "errors: %d\n", errors
      print "failing: " (failing == "" ? "none" : failing)
      print "sanitizer: " (san == "" ? "none" : san)
      print "xid: " (xid == "" ? "none" : xid)
      print "rc: " (rc == "" ? "none" : rc)
      printf "%s", missing
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
