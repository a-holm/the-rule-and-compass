#!/usr/bin/env bash
# Usage: test-rc-run.sh
# Hermetic test of rc-run.sh with a stub slot, lock and Xid command: modes, quoting, refusals, Xid and exit codes.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/lib/fakes.sh"
RC=$HERE/../rc-run.sh
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fails=0
check() {
  local name=$1 got=$2 want=$3
  if [ "$got" != "$want" ]; then
    echo "FAIL $name: got '$got' want '$want'"
    fails=$((fails + 1))
  fi
}
hasnt() {
  local name=$1 file=$2 pattern=$3
  if grep -qE -- "$pattern" "$file"; then
    echo "FAIL $name: '$pattern' found in $file"
    fails=$((fails + 1))
  fi
}
has() {
  local name=$1 file=$2 pattern=$3
  if ! grep -qE -- "$pattern" "$file"; then
    echo "FAIL $name: '$pattern' not in $file"
    fails=$((fails + 1))
  fi
}

mkdir -p "$T/gates" "$T/wt/my-version" "$T/bin" "$T/oc"
fake_timeout "$T/bin"
touch "$T/wt/my-version/Cargo.toml"
echo 4 > "$T/xid"
for kind in slot lock; do
  cat > "$T/$kind.sh" <<STUB
#!/usr/bin/env bash
echo "$kind \$*" >> "$T/calls"
[ -z "\${FAKE_XID_BUMP:-}" ] || echo 5 > "$T/xid"
exec "\$@"
STUB
  chmod +x "$T/$kind.sh"
done
cat > "$T/bin/show-args" <<STUB
#!/usr/bin/env bash
printf '[%s]\n' "\$@"
echo "arch=\${RCSR_CUDA_ARCH:-none} jobs=\${CARGO_BUILD_JOBS:-none} pwd=\$PWD"
exit "\${FAKE_RC:-0}"
STUB
chmod +x "$T/bin/show-args"

rc() {
  env -u CARGO_TARGET_DIR RCSR_GATES="$T/gates" RCSR_SLOT_SCRIPT="$T/slot.sh" RCSR_LOCK_SCRIPT="$T/lock.sh" \
    RCSR_XID_CMD="cat '$T/xid'" RCSR_MACHINE_DIR="$T/oc" RCSR_REAP_TERM_WAITS=2 RCSR_REAP_KILL_WAITS=50 PATH="$T/bin:$PATH" FAKE_TIMEOUT_LOG="$T/timeouts" "$@"
}
log=$T/gates/issue9-t.log

# build uses the slot, no xid line, last line is RC_DONE
: > "$T/calls"
out=$(rc bash "$RC" build "$T/wt" 9 t -- show-args a); r=$?
check build-rc "$r" "0"
check build-calls "$(cut -d' ' -f1 "$T/calls")" "slot"
check build-stdout-first "$(echo "$out" | head -n 1)" "$log"
check build-stdout-last "$(echo "$out" | tail -n 1 | cut -d' ' -f1-2)" "RC_DONE rc=0"
check build-no-xid "$(grep -c '^xid ' "$log")" "0"
check build-last-line "$(tail -n 1 "$log" | cut -d' ' -f1-2)" "RC_DONE rc=0"
check build-limit "$(head -n 1 "$T/timeouts")" "1500 show-args a"
has build-jobs "$log" 'jobs=2'
has build-noarch "$log" 'arch=none'
has build-cwd "$log" "pwd=$T/wt/my-version"
has build-timeout-default "$T/calls" 'timeout 1500 '

# test --cuda and run use the lock and write the xid line; test without --cuda uses the slot
for m in "test" "run"; do
  : > "$T/calls"
  rc bash "$RC" $m "$T/wt" 9 t --cuda -- show-args a > /dev/null; r=$?
  check "lock-$m-rc" "$r" "0"
  check "lock-$m-calls" "$(cut -d' ' -f1 "$T/calls")" "lock"
  has "lock-$m-xid" "$log" '^xid before=4 after=4$'
  has "lock-$m-timeout" "$T/calls" 'timeout 580 '
done
has test-cuda-arch "$log" 'arch=sm_86'
: > "$T/calls"
rc bash "$RC" test "$T/wt" 9 t -- show-args > /dev/null
check test-cpu-slot "$(cut -d' ' -f1 "$T/calls")" "slot"
: > "$T/calls"
rc bash "$RC" cpu "$T/wt" 9 t -- show-args > /dev/null
check cpu-slot "$(cut -d' ' -f1 "$T/calls")" "slot"

# command exit code passes through
out=$(FAKE_RC=3 rc bash "$RC" run "$T/wt" 9 t -- show-args); r=$?
check exit3-rc "$r" "3"
check exit3-done "$(echo "$out" | tail -n 1 | cut -d' ' -f1-2)" "RC_DONE rc=3"
check exit3-log "$(tail -n 1 "$log" | cut -d' ' -f1-2)" "RC_DONE rc=3"

# a timeout gives 124 and still ends with RC_DONE
cat > "$T/bin/hang" <<'STUB'
#!/usr/bin/env bash
echo "hang ran" >&2
exit 99
STUB
chmod +x "$T/bin/hang"
rc bash "$RC" run "$T/wt" 9 t --timeout 1 -- hang > /dev/null; r=$?
check timeout-rc "$r" "124"
check timeout-limit "$(tail -n 1 "$T/timeouts")" "1 hang"
check timeout-last "$(tail -n 1 "$log" | cut -d' ' -f1-2)" "RC_DONE rc=124"

# slot or lock give-up (75) passes through
printf '#!/usr/bin/env bash\necho gave up >&2\nexit 75\n' > "$T/lock.sh"
rc bash "$RC" run "$T/wt" 9 t -- show-args > /dev/null; r=$?
check giveup-rc "$r" "75"
check giveup-last "$(tail -n 1 "$log" | cut -d' ' -f1-2)" "RC_DONE rc=75"
cat > "$T/lock.sh" <<STUB
#!/usr/bin/env bash
echo "lock \$*" >> "$T/calls"
[ -z "\${FAKE_XID_BUMP:-}" ] || echo 5 > "$T/xid"
exec "\$@"
STUB

# a rising Xid count gives 70
echo 4 > "$T/xid"
FAKE_XID_BUMP=1 rc bash "$RC" run "$T/wt" 9 t -- show-args > /dev/null; r=$?
check xid-rc "$r" "70"
has xid-line "$log" '^xid before=4 after=5$'
has xid-stop "$log" '^STOP: NVRM Xid count rose 4 -> 5$'
check xid-last "$(tail -n 1 "$log" | cut -d' ' -f1-2)" "RC_DONE rc=70"
echo 4 > "$T/xid"

# arguments with a space and a quote arrive unchanged
rc bash "$RC" cpu "$T/wt" 9 t -- show-args "a b" "it's \"x\"" '$HOME' > /dev/null
has quote-1 "$log" '^\[a b\]$'
has quote-2 "$log" "^\[it's \"x\"\]\$"
has quote-3 "$log" '^\[\$HOME\]$'

# refusals exit 2 and leave no log
rm -f "$log"
rc bash "$RC" run "$T/wt" 9 t --timeout 900 -- show-args > /dev/null 2>&1; r=$?
check refuse-timeout-lock "$r" "2"
rc bash "$RC" test "$T/wt" 9 t --cuda --timeout 601 -- show-args > /dev/null 2>&1; r=$?
check refuse-timeout-test-cuda "$r" "2"
rc bash "$RC" build "$T/wt" 9 t --timeout 900 -- show-args > /dev/null 2>&1; r=$?
check build-long-timeout-ok "$r" "0"
rc bash "$RC" cpu "$T/wt" 9 t -- show-args > /dev/null
rm -f "$log"
CARGO_TARGET_DIR=/x RCSR_GATES="$T/gates" RCSR_SLOT_SCRIPT="$T/slot.sh" bash "$RC" cpu "$T/wt" 9 t -- show-args > /dev/null 2>&1; r=$?
check refuse-target-dir "$r" "2"
rc bash "$RC" nope "$T/wt" 9 t -- show-args > /dev/null 2>&1; r=$?
check refuse-mode "$r" "2"
rc bash "$RC" cpu "$T/none" 9 t -- show-args > /dev/null 2>&1; r=$?
check refuse-no-cargo-toml "$r" "2"
rc bash "$RC" cpu "$T/wt" 9 t show-args > /dev/null 2>&1; r=$?
check refuse-no-separator "$r" "2"
rc bash "$RC" cpu "$T/wt" 9 t -- > /dev/null 2>&1; r=$?
check refuse-no-command "$r" "2"
rc bash "$RC" cpu "$T/wt" 9 t --timeout abc -- show-args > /dev/null 2>&1; r=$?
check refuse-timeout-nan "$r" "2"
check refuse-no-log "$([ -e "$log" ] && echo yes || echo no)" "no"

# a command that ends but leaves children behind: every process that holds the test lock after it is ended by pid,
# a child that ignores SIGTERM and one in its own session included; a process that only has the lock file open stays
cat > "$T/lock.sh" <<STUB
#!/usr/bin/env bash
echo "lock \$*" >> "$T/calls"
exec 9> "$T/oc/full-suite.lock"
flock -n 9 || exit 75
exec "\$@"
STUB
cat > "$T/bin/leave-children" <<'STUB'
#!/usr/bin/env bash
trap '' TERM
tail -f /dev/null &
echo $! > "$FAKE_DIR/pid-same-session"
setsid tail -f /dev/null &
echo $! > "$FAKE_DIR/pid-other-session"
exit 3
STUB
chmod +x "$T/bin/leave-children"
lockreal=$(readlink -f "$T/oc/full-suite.lock")
lock_holders() { find /proc/[0-9]*/fd -lname "$lockreal" 2> /dev/null | cut -d/ -f3 | grep -vx "$bystander"; }
alive() { [ -r "/proc/$1/stat" ] && [ "$(cut -d' ' -f3 "/proc/$1/stat")" != Z ]; }
tail -f /dev/null 8> "$T/oc/full-suite.lock" &
bystander=$!
trap 'kill -KILL "$bystander" $(cat "$T/pid-same-session" "$T/pid-other-session" 2> /dev/null) 2> /dev/null; rm -rf "$T"' EXIT
FAKE_DIR=$T rc bash "$RC" run "$T/wt" 9 t --cuda -- leave-children > /dev/null 2>&1; r=$?
check orphans-rc "$r" "3"
check orphans-no-holder "$(lock_holders | tr '\n' ' ')" ""
check orphans-same-session-gone "$(alive "$(cat "$T/pid-same-session")" && echo alive || echo gone)" "gone"
check orphans-other-session-gone "$(alive "$(cat "$T/pid-other-session")" && echo alive || echo gone)" "gone"
check orphans-waiter-kept "$(alive "$bystander" && echo alive || echo gone)" "alive"
has orphans-logged "$log" '^lock-run: ended pids still holding the lock: '
check orphans-last "$(tail -n 1 "$log" | cut -d' ' -f1-2)" "RC_DONE rc=3"
kill "$bystander"
FAKE_DIR=$T rc bash "$RC" run "$T/wt" 9 t --cuda -- show-args > /dev/null 2>&1; r=$?
check clean-run-rc "$r" "0"
hasnt clean-run-quiet "$log" 'lock-run:'

# -h prints the header
check help-rc "$(bash "$RC" -h > "$T/h"; echo $?)" "0"
has help-usage "$T/h" '^Usage: rc-run.sh build\|test\|run\|cpu'

if [ "$fails" -ne 0 ]; then
  echo "test-rc-run: $fails failed"
  exit 1
fi
echo "test-rc-run: ok"
