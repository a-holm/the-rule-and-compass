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
has build-timeout-default "$T/calls" 'timeout -k 30 1500 '

# test --cuda and run use the lock and write the xid line; test without --cuda uses the slot
for m in "test" "run"; do
  : > "$T/calls"
  rc bash "$RC" $m "$T/wt" 9 t --cuda -- show-args a > /dev/null; r=$?
  check "lock-$m-rc" "$r" "0"
  check "lock-$m-calls" "$(cut -d' ' -f1 "$T/calls")" "lock"
  has "lock-$m-xid" "$log" '^xid before=4 after=4$'
  has "lock-$m-timeout" "$T/calls" 'timeout -k 30 580 '
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
exec 9> "\${FAKE_LOCK:-$T/oc/full-suite.lock}"
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
trap 'kill -KILL "$bystander" $(cat "$T"/pid-* 2> /dev/null) 2> /dev/null; rm -rf "$T"' EXIT
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

# guard: lock-run signals only processes that share the flock of its own hold. Every path here is the temp dir; $T/oc/full-suite.lock
# stands for the machine lock that rc-run.sh used to scan whatever lock the lock script took.
holder_pid() { ( exec 7> "$1"; flock -n 7 && exec tail -f /dev/null > /dev/null 2>&1 ) & echo $!; }
other=$(holder_pid "$T/oc/full-suite.lock")
trap 'kill -KILL "$bystander" "$other" $(cat "$T"/pid-* 2> /dev/null) 2> /dev/null; rm -rf "$T"' EXIT
# a hold of another lock file: the holder of the standing lock file survives and nothing is signalled
FAKE_LOCK=$T/second.lock FAKE_DIR=$T rc bash "$RC" run "$T/wt" 9 t --cuda -- leave-children > /dev/null 2>&1; r=$?
check guard-other-lock-rc "$r" "3"
check guard-other-lock-holder-kept "$(alive "$other" && echo alive || echo gone)" "alive"
check guard-other-lock-children-left "$(alive "$(cat "$T/pid-same-session")" && echo alive || echo gone)" "alive"
has guard-other-lock-logged "$log" '^lock-run: no flock held by this process; nothing will be signalled$'
kill -KILL $(cat "$T/pid-same-session" "$T/pid-other-session") 2> /dev/null
# a lock script that takes no flock: nothing is signalled, the children are left alone and the log says so
sed -i '/^flock -n 9/d' "$T/lock.sh"
# the caller holds an unrelated flock too (as the build slot does for cargo): it is not this hold and is never signalled
( exec 8> "$T/slot.lock"; flock -n 8 || exit 1; FAKE_DIR=$T rc bash "$RC" run "$T/wt" 9 t --cuda -- leave-children > /dev/null 2>&1 ); r=$?
check guard-no-flock-rc "$r" "3"
check guard-no-flock-holder-kept "$(alive "$other" && echo alive || echo gone)" "alive"
check guard-no-flock-child-kept "$(alive "$(cat "$T/pid-same-session")" && echo alive || echo gone)" "alive"
has guard-no-flock-logged "$log" '^lock-run: no flock held by this process; nothing will be signalled$'
hasnt guard-no-flock-no-signal "$log" 'ended pids'
kill -KILL "$other" $(cat "$T/pid-same-session" "$T/pid-other-session") 2> /dev/null

# nested: the caller already shares the flock on the lock file with a sibling (a test that runs under the lock and calls
# rc-run.sh with a stub lock script). lock-run is not the sole holder at start: nothing is signalled, the caller and its sibling live
cat > "$T/nested.sh" <<'STUB'
#!/usr/bin/env bash
exec 9> "$T/oc/full-suite.lock"
flock -n 9 || exit 1
tail -f /dev/null > /dev/null 2>&1 &
echo $! > "$T/pid-sibling"
rc=0
FAKE_DIR=$T "$@" > /dev/null 2>&1 || rc=$?
echo "$rc" > "$T/nested-rc"
STUB
export T
printf '#!/usr/bin/env bash\nexec "$@"\n' > "$T/lock.sh"
rc bash "$T/nested.sh" bash "$RC" run "$T/wt" 9 t --cuda -- leave-children
check nested-rc "$(cat "$T/nested-rc" 2> /dev/null)" "3"
check nested-sibling-kept "$(alive "$(cat "$T/pid-sibling")" && echo alive || echo gone)" "alive"
check nested-child-kept "$(alive "$(cat "$T/pid-same-session")" && echo alive || echo gone)" "alive"
has nested-logged "$log" '^lock-run: this process is not the sole holder of the flock '
kill -KILL "$(cat "$T/pid-sibling")" $(cat "$T/pid-same-session" "$T/pid-other-session") 2> /dev/null

# --- review findings: ownership, group signals, TERM to lock-run, timeout -k. Every path is in the temp dir.
flock_stub() { # a lock script that takes the flock on $T/oc/full-suite.lock, as one-full-suite.sh does
  cat > "$T/lock.sh" <<STUB
#!/usr/bin/env bash
echo "lock \$*" >> "$T/calls"
exec 9> "$T/oc/full-suite.lock"
flock -n 9 || exit 75
exec "\$@"
STUB
}
mkdir -p "$T/fifo"
# b_holds <fifo>: in the background, take the flock on the lock file path (a new hold B), say ready, then block
cat > "$T/bin/b-holds" <<'STUB'
#!/usr/bin/env bash
# b-holds <ready fifo>: B takes its own flock on the lock path, reports ready or fail, and blocks holding it
exec 7> "$FAKE_DIR/oc/full-suite.lock"
if flock -n 7; then echo ready > "$1"; exec tail -f /dev/null > /dev/null 2>&1; fi
echo fail > "$1"
STUB
chmod +x "$T/bin/b-holds"
cleanup_pids() { kill -KILL $(cat "$T"/pid-* 2> /dev/null) 2> /dev/null; rm -f "$T"/pid-*; }

# finding 1a: the lock file is replaced (rm, then a new hold B creates a new inode): B is never signalled
cat > "$T/bin/replace-lock" <<'STUB'
#!/usr/bin/env bash
trap '' TERM
tail -f /dev/null > /dev/null 2>&1 &
echo $! > "$FAKE_DIR/pid-held"
rm -f "$FAKE_DIR/oc/full-suite.lock"
rm -f "$FAKE_DIR/fifo/b"; mkfifo "$FAKE_DIR/fifo/b"
b-holds "$FAKE_DIR/fifo/b" &
echo $! > "$FAKE_DIR/pid-b"
read -r _ < "$FAKE_DIR/fifo/b"
exit 3
STUB
chmod +x "$T/bin/replace-lock"
flock_stub
FAKE_DIR=$T rc bash "$RC" run "$T/wt" 9 t --cuda -- replace-lock > /dev/null 2>&1; r=$?
check lost-replaced-rc "$r" "3"
check lost-replaced-b-kept "$(alive "$(cat "$T/pid-b")" && echo alive || echo gone)" "alive"
check lost-replaced-held-kept "$(alive "$(cat "$T/pid-held")" && echo alive || echo gone)" "alive"
has lost-replaced-logged "$log" '^lock-run: ownership of the flock is gone'
cleanup_pids

# finding 1b: a descendant releases the flock (flock -u) and a new hold B locks the same inode: B is never signalled
cat > "$T/bin/release-lock" <<'STUB'
#!/usr/bin/env bash
trap '' TERM
flock -u 9
tail -f /dev/null > /dev/null 2>&1 &
echo $! > "$FAKE_DIR/pid-held"
rm -f "$FAKE_DIR/fifo/b"; mkfifo "$FAKE_DIR/fifo/b"
b-holds "$FAKE_DIR/fifo/b" &
echo $! > "$FAKE_DIR/pid-b"
read -r _ < "$FAKE_DIR/fifo/b"
exit 3
STUB
chmod +x "$T/bin/release-lock"
flock_stub
FAKE_DIR=$T rc bash "$RC" run "$T/wt" 9 t --cuda -- release-lock > /dev/null 2>&1; r=$?
check lost-released-rc "$r" "3"
check lost-released-b-kept "$(alive "$(cat "$T/pid-b")" && echo alive || echo gone)" "alive"
check lost-released-held-kept "$(alive "$(cat "$T/pid-held")" && echo alive || echo gone)" "alive"
has lost-released-logged "$log" '^lock-run: ownership of the flock is gone'
cleanup_pids

# finding 3: no group signals. A process that closed the lock fd and stayed in the session is not signalled; one that kept it is
cat > "$T/bin/close-fd" <<'STUB'
#!/usr/bin/env bash
trap '' TERM
( exec 9>&-; exec tail -f /dev/null > /dev/null 2>&1 ) &
echo $! > "$FAKE_DIR/pid-closed"
tail -f /dev/null > /dev/null 2>&1 &
echo $! > "$FAKE_DIR/pid-held"
exit 3
STUB
chmod +x "$T/bin/close-fd"
flock_stub
FAKE_DIR=$T rc bash "$RC" run "$T/wt" 9 t --cuda -- close-fd > /dev/null 2>&1; r=$?
check group-rc "$r" "3"
check group-closed-kept "$(alive "$(cat "$T/pid-closed")" && echo alive || echo gone)" "alive"
check group-held-gone "$(alive "$(cat "$T/pid-held")" && echo alive || echo gone)" "gone"
hasnt group-no-group-kill "$HERE/../lock-run.sh" 'kill .*-- *"?-'
cleanup_pids

# finding 4: TERM sent to lock-run is forwarded to the command and the reap still runs
cat > "$T/bin/block" <<'STUB'
#!/usr/bin/env bash
trap 'exit 7' TERM
( trap '' TERM; exec tail -f /dev/null > /dev/null 2>&1 ) &
echo $! > "$FAKE_DIR/pid-held"
echo "$$" > "$FAKE_DIR/pid-block"
echo "$PPID" > "$FAKE_DIR/fifo/ready"
wait
STUB
chmod +x "$T/bin/block"
flock_stub
rm -f "$T/fifo/ready"; mkfifo "$T/fifo/ready"
FAKE_DIR=$T rc bash "$RC" run "$T/wt" 9 t --cuda -- block > /dev/null 2>&1 &
bg=$!
read -r lockrun_pid < "$T/fifo/ready"
kill -TERM "$lockrun_pid"
wait "$bg"; r=$?
check term-rc "$r" "7"
check term-held-gone "$(alive "$(cat "$T/pid-held")" && echo alive || echo gone)" "gone"
check term-block-gone "$(alive "$(cat "$T/pid-block")" && echo alive || echo gone)" "gone"
check term-lockrun-gone "$(alive "$lockrun_pid" && echo alive || echo gone)" "gone"
cleanup_pids

# finding 2: the command runs under timeout -k; a stub case reads the log, a real timeout ends a command that ignores TERM
check kill-after-stub "$(tail -n 1 "$T/timeouts.kill")" "30"
cat > "$T/bin/ignore-term" <<'STUB'
#!/usr/bin/env bash
trap '' TERM
tail -f /dev/null > /dev/null 2>&1
STUB
chmod +x "$T/bin/ignore-term"
mkdir -p "$T/real-bin"
ln -sf "$T/bin/ignore-term" "$T/real-bin/ignore-term"
flock_stub
RCSR_KILL_AFTER=0.5 env -u CARGO_TARGET_DIR RCSR_GATES="$T/gates" RCSR_SLOT_SCRIPT="$T/slot.sh" RCSR_LOCK_SCRIPT="$T/lock.sh" \
  RCSR_XID_CMD="cat '$T/xid'" RCSR_MACHINE_DIR="$T/oc" RCSR_REAP_TERM_WAITS=2 PATH="$T/real-bin:$PATH" \
  bash "$RC" run "$T/wt" 9 t --cuda --timeout 1 -- ignore-term > /dev/null 2>&1; r=$?
check real-timeout-kill-rc "$r" "137"

# -h prints the header
check help-rc "$(bash "$RC" -h > "$T/h"; echo $?)" "0"
has help-usage "$T/h" '^Usage: rc-run.sh build\|test\|run\|cpu'

if [ "$fails" -ne 0 ]; then
  echo "test-rc-run: $fails failed"
  exit 1
fi
echo "test-rc-run: ok"
