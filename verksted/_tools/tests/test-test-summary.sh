#!/usr/bin/env bash
# Usage: test-test-summary.sh
# Hermetic test of test-summary.sh on fixture logs: green, failing tests, compile error, sanitizer, xid, rc, warnings.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
S=$HERE/../test-summary.sh
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}
line() { bash "$S" "$2" | grep "^$1:"; }

cat > "$T/green.log" <<'LOG'
warning: unused variable: `x`
warning: `rcsr` (lib) generated 1 warning
running 3 tests
test a ... ok
test b ... ok
test c ... ignored
test result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out
running 1 test
test d ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
LOG
bash "$S" "$T/green.log" > "$T/green.out"; rc=$?
check green-rc "$rc" 0
check green-tests "$(grep '^tests:' "$T/green.out")" 'tests: passed=3 failed=0 ignored=1 results=2/2'
check green-warnings "$(grep '^warnings:' "$T/green.out")" 'warnings: 1'
check green-errors "$(grep '^errors:' "$T/green.out")" 'errors: 0'
check green-failing "$(grep '^failing:' "$T/green.out")" 'failing: none'
check green-san "$(grep '^sanitizer:' "$T/green.out")" 'sanitizer: none'
check green-xid "$(grep '^xid:' "$T/green.out")" 'xid: none'
check green-rc-line "$(grep '^rc:' "$T/green.out")" 'rc: none'
check green-head "$(head -n 1 "$T/green.out")" "== $T/green.log"
check green-verdict "$(grep '^verdict:' "$T/green.out")" 'verdict: ok'

cat > "$T/red.log" <<'LOG'
running 3 tests
test one ... ok
test mod::two ... FAILED
test three ... FAILED
test result: FAILED. 1 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out
error: test failed, to rerun pass `--lib`
LOG
bash "$S" "$T/red.log" > "$T/red.out"; rc=$?
check red-rc "$rc" 1
check red-tests "$(grep '^tests:' "$T/red.out")" 'tests: passed=1 failed=2 ignored=0 results=0/1'
check red-failing "$(grep '^failing:' "$T/red.out")" 'failing: mod::two,three'
check red-errors "$(grep '^errors:' "$T/red.out")" 'errors: 0'
check red-verdict "$(grep '^verdict:' "$T/red.out")" 'verdict: fail'

cat > "$T/compile.log" <<'LOG'
error[E0308]: mismatched types
warning: `rcsr` (lib test) generated 2 warnings (run `cargo fix`)
LOG
bash "$S" "$T/compile.log" > "$T/compile.out"; rc=$?
check compile-rc "$rc" 1
check compile-errors "$(grep '^errors:' "$T/compile.out")" 'errors: 1'
check compile-warnings "$(grep '^warnings:' "$T/compile.out")" 'warnings: 0'

printf '========= ERROR SUMMARY: 0 errors\n' > "$T/san0.log"
bash "$S" "$T/san0.log" > "$T/san0.out"; rc=$?
check san0-rc "$rc" 0
check san0-line "$(grep '^sanitizer:' "$T/san0.out")" 'sanitizer: ERROR SUMMARY: 0 errors'
printf '========= ERROR SUMMARY: 3 errors\n' > "$T/san3.log"
bash "$S" "$T/san3.log" > "$T/san3.out"; rc=$?
check san3-rc "$rc" 1
check san3-line "$(grep '^sanitizer:' "$T/san3.out")" 'sanitizer: ERROR SUMMARY: 3 errors'
check san3-verdict "$(grep '^verdict:' "$T/san3.out")" 'verdict: fail'
printf 'ERROR SUMMARY: 0 errors\nERROR SUMMARY: 0 errors\n' > "$T/san2.log"
check san-joined "$(line sanitizer "$T/san2.log")" 'sanitizer: ERROR SUMMARY: 0 errors; ERROR SUMMARY: 0 errors'

printf 'xid before=2 after=2\nxid before=2 after=3\n' > "$T/xid.log"
bash "$S" "$T/xid.log" > "$T/xid.out"; rc=$?
check xid-rc "$rc" 1
check xid-line "$(grep '^xid:' "$T/xid.out")" 'xid: xid before=2 after=3'
printf 'xid before=2 after=2\n' > "$T/xidok.log"
bash "$S" "$T/xidok.log" > /dev/null; check xid-equal-rc "$?" 0

printf 'RC_DONE rc=1 at=x\n' > "$T/rc1.log"
bash "$S" "$T/rc1.log" > "$T/rc1.out"; rc=$?
check rc1-rc "$rc" 1
check rc1-line "$(grep '^rc:' "$T/rc1.out")" 'rc: 1'
printf 'RC_DONE rc=0 at=x\n' > "$T/rc0.log"
bash "$S" "$T/rc0.log" > /dev/null; check rc0-rc "$?" 0

# several logs: one fail makes exit 1, blocks are separated
bash "$S" "$T/green.log" "$T/red.log" > "$T/multi.out"; rc=$?
check multi-rc "$rc" 1
check multi-blocks "$(grep -c '^== ' "$T/multi.out")" 2

# usage and missing log
bash "$S" > /dev/null 2>&1; check usage-none "$?" 2
bash "$S" "$T/nope.log" > /dev/null 2>&1; check usage-missing "$?" 2
bash "$S" -h > /dev/null 2>&1; check help "$?" 0

# no process matching in either script
check no-pgrep "$(cat "$HERE/../wait-for.sh" "$HERE/../test-summary.sh" | grep -vE '^\s*#' | grep -cE 'pgrep|pkill')" 0

if [ "$fails" -ne 0 ]; then
  echo "test-test-summary: $fails failed"
  exit 1
fi
echo "test-test-summary: ok"
