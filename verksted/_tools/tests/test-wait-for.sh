#!/usr/bin/env bash
# Usage: test-wait-for.sh
# Hermetic test of wait-for.sh: log match, timeout, file, count, custom pattern, usage errors, summary on cargo logs.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
W=$HERE/../wait-for.sh
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
export RCSR_POLL=0.2
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}
elapsed_ms() { echo $(( (${EPOCHREALTIME/./} - $1) / 1000 )); }

# default pattern: background writer appends RC_DONE after 1 s
: > "$T/a.log"
(sleep 1; echo 'RC_DONE rc=0 at=x' >> "$T/a.log") &
start=${EPOCHREALTIME/./}
out=$(bash "$W" --log "$T/a.log"); rc=$?
ms=$(elapsed_ms "$start")
check log-rc "$rc" 0
check log-line "$out" 'RC_DONE rc=0 at=x'
[ "$ms" -lt 3000 ] || { echo "FAIL log-fast: $ms ms"; fails=$((fails + 1)); }
[ "$ms" -ge 900 ] || { echo "FAIL log-waited: $ms ms"; fails=$((fails + 1)); }
wait

# timeout without a match, also for a missing log
: > "$T/b.log"
start=${EPOCHREALTIME/./}
bash "$W" --log "$T/b.log" --timeout 1 > /dev/null; rc=$?
ms=$(elapsed_ms "$start")
check timeout-rc "$rc" 124
[ "$ms" -lt 3000 ] || { echo "FAIL timeout-fast: $ms ms"; fails=$((fails + 1)); }
bash "$W" --log "$T/missing.log" --timeout 1 > /dev/null; check timeout-missing-log "$?" 124
bash "$W" --file "$T/nofile" --timeout 1 > /dev/null; check timeout-file "$?" 124
bash "$W" --count 1 --dir "$T/nodir" --timeout 1 > /dev/null; check timeout-dir "$?" 124
bash "$W" --log "$T/b.log" --timeout 0 > /dev/null; check timeout-zero "$?" 124

# a shell that waits on nothing else cannot be held by another process name
check no-pgrep "$(grep -vE '^\s*#' "$W" | grep -cE 'pgrep|pkill|ps ')" 0

# custom ERE
printf 'x\nALLDONE 42\n' > "$T/c.log"
out=$(bash "$W" --log "$T/c.log" --pattern 'ALLDONE [0-9]+' --timeout 2); rc=$?
check pattern-rc "$rc" 0
check pattern-line "$out" 'ALLDONE 42'
bash "$W" --log "$T/c.log" --pattern '^RC_DONE' --timeout 1 > /dev/null; check pattern-nomatch "$?" 124

# --file waits for a non-empty file
: > "$T/d.done"
(sleep 1; echo ok > "$T/d.done") &
start=${EPOCHREALTIME/./}
bash "$W" --file "$T/d.done" --timeout 5; rc=$?
ms=$(elapsed_ms "$start")
check file-rc "$rc" 0
[ "$ms" -ge 900 ] || { echo "FAIL file-waited-for-content: $ms ms"; fails=$((fails + 1)); }
wait

# --count returns once the third file appears
mkdir "$T/dir"
touch "$T/dir/f1.out" "$T/dir/skip.txt"
(sleep 0.5; touch "$T/dir/f2.out"; sleep 0.7; touch "$T/dir/f3.out") &
start=${EPOCHREALTIME/./}
bash "$W" --count 3 --dir "$T/dir" --glob '*.out' --timeout 5; rc=$?
ms=$(elapsed_ms "$start")
check count-rc "$rc" 0
[ "$ms" -ge 1100 ] || { echo "FAIL count-waited-for-third: $ms ms"; fails=$((fails + 1)); }
wait

# cargo log: matching line, then the summary block
cat > "$T/e.log" <<'LOG'
running 2 tests
test a ... ok
test b ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
RC_DONE rc=0 at=2026-11-03T00:00:00Z
LOG
out=$(bash "$W" --log "$T/e.log" --timeout 2); rc=$?
check summary-rc "$rc" 0
case $out in
  *'RC_DONE rc=0'*'tests: passed=2 failed=0 ignored=0 results=1/1'*'verdict: ok'*) ;;
  *) echo "FAIL summary-block: $out"; fails=$((fails + 1)) ;;
esac
# no summary for a plain log
out=$(bash "$W" --log "$T/c.log" --pattern 'ALLDONE' --timeout 2)
check no-summary "$out" 'ALLDONE 42'

# usage errors
bash "$W" > /dev/null 2>&1; check usage-none "$?" 2
bash "$W" --log a --file b > /dev/null 2>&1; check usage-two-modes "$?" 2
bash "$W" --log a --timeout x > /dev/null 2>&1; check usage-timeout "$?" 2
bash "$W" --count 2 > /dev/null 2>&1; check usage-count-no-dir "$?" 2
bash "$W" --bogus > /dev/null 2>&1; check usage-unknown "$?" 2
bash "$W" --log > /dev/null 2>&1; check usage-missing-value "$?" 2
bash "$W" -h > /dev/null 2>&1; check help "$?" 0

if [ "$fails" -ne 0 ]; then
  echo "test-wait-for: $fails failed"
  exit 1
fi
echo "test-wait-for: ok"
