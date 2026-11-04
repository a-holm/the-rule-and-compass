#!/usr/bin/env bash
# Usage: test-wait-for.sh
# Hermetic test of wait-for.sh: log match, timeout, file, count, custom pattern, usage errors, summary on cargo logs.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/lib/fakes.sh"
W=$HERE/../wait-for.sh
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
export VCLOCK=$T/clock
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}

# wait-for runs on a virtual clock: every sleep advances it by whole seconds and counts as one poll
mkdir -p "$T/fake"; fake_clock "$T/fake"
w() { PATH="$T/fake:$PATH" RCSR_POLL=1 bash "$W" "$@"; }

# default pattern: the writer appends RC_DONE at virtual second 3
vclock_reset
: > "$T/a.log"
vclock_at 3 "echo 'RC_DONE rc=0 at=x' >> '$T/a.log'"
out=$(w --log "$T/a.log"); rc=$?
check log-rc "$rc" 0
check log-line "$out" 'RC_DONE rc=0 at=x'
check log-polls "$(polls)" 3

# timeout without a match, also for a missing log: one poll per second of the limit
vclock_reset
: > "$T/b.log"
w --log "$T/b.log" --timeout 3 > /dev/null; check timeout-rc "$?" 124
check timeout-polls "$(polls)" 3
vclock_reset; w --log "$T/missing.log" --timeout 1 > /dev/null; check timeout-missing-log "$?" 124
check timeout-missing-log-polls "$(polls)" 1
vclock_reset; w --file "$T/nofile" --timeout 1 > /dev/null; check timeout-file "$?" 124
check timeout-file-polls "$(polls)" 1
vclock_reset; w --count 1 --dir "$T/nodir" --timeout 1 > /dev/null; check timeout-dir "$?" 124
check timeout-dir-polls "$(polls)" 1
vclock_reset; w --log "$T/b.log" --timeout 0 > /dev/null; check timeout-zero "$?" 124
check timeout-zero-polls "$(polls)" 0

# a shell that waits on nothing else cannot be held by another process name
check no-pgrep "$(grep -vE '^\s*#' "$W" | grep -cE 'pgrep|pkill|ps ')" 0

# custom ERE
printf 'x\nALLDONE 42\n' > "$T/c.log"
vclock_reset
out=$(w --log "$T/c.log" --pattern 'ALLDONE [0-9]+' --timeout 2); rc=$?
check pattern-rc "$rc" 0
check pattern-line "$out" 'ALLDONE 42'
check pattern-polls "$(polls)" 0
vclock_reset; w --log "$T/c.log" --pattern '^RC_DONE' --timeout 1 > /dev/null; check pattern-nomatch "$?" 124
check pattern-nomatch-polls "$(polls)" 1

# --file waits for a non-empty file
vclock_reset
: > "$T/d.done"
vclock_at 2 "echo ok > '$T/d.done'"
w --file "$T/d.done" --timeout 5; rc=$?
check file-rc "$rc" 0
check file-polls "$(polls)" 2

# --count returns once the third file appears
vclock_reset
mkdir "$T/dir"
touch "$T/dir/f1.out" "$T/dir/skip.txt"
vclock_at 1 "touch '$T/dir/f2.out'"
vclock_at 2 "touch '$T/dir/f3.out'"
w --count 3 --dir "$T/dir" --glob '*.out' --timeout 5; rc=$?
check count-rc "$rc" 0
check count-polls "$(polls)" 2

# cargo log: matching line, then the summary block
cat > "$T/e.log" <<'LOG'
running 2 tests
test a ... ok
test b ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
RC_DONE rc=0 at=2026-11-03T00:00:00Z
LOG
vclock_reset
out=$(w --log "$T/e.log" --timeout 2); rc=$?
check summary-rc "$rc" 0
case $out in
  *'RC_DONE rc=0'*'tests: passed=2 failed=0 ignored=0 results=1/1'*'verdict: ok'*) ;;
  *) echo "FAIL summary-block: $out"; fails=$((fails + 1)) ;;
esac
# no summary for a plain log
vclock_reset
out=$(w --log "$T/c.log" --pattern "ALLDONE" --timeout 2)
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
