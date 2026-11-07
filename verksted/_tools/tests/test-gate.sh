#!/usr/bin/env bash
# Usage: test-gate.sh
# Hermetic test of gate.sh with a stub lock, a stub build slot and an Xid command: one lock hold per CUDA target, a timed-out
# or failed target does not stop the others and fails the gate, no hold exceeds the cap, an Xid rise stops the run.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GATE=$HERE/../gate.sh
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}
mkdir -p "$T/gates" "$T/wt/my-version"
git -C "$T/wt" init -q

# The stub lock gets `bash -c <cmd>`. It records <cmd> and answers by the target at the end of a CUDA command, using
# the lines "<target args> <rc> [xid]" of $RCSR_STUB_RC (rc 124 prints nothing, like a killed run; xid raises the count).
cat > "$T/lock.sh" <<'STUB'
#!/usr/bin/env bash
cmd=${!#}
echo "$*" >> "$RCSR_STUB_HOLDS"
echo "$cmd" >> "$RCSR_STUB_CMDS"
case "$cmd" in
  *rcsr-bench*) cat "$RCSR_STUB_BENCH"; exit 0 ;;
  *"--features cuda"*)
    start=$(sed -n "s/.*> '\([^']*\.start\)' &&.*/\1/p" <<< "$cmd")
    now=$(cat "$RCSR_STUB_NOW")
    [ -z "$start" ] || echo $((now + 7)) > "$start"
    echo $((now + 37)) > "$RCSR_STUB_NOW"
    tgt=${cmd##*--no-fail-fast }
    line=$(grep -F -- "$tgt " "$RCSR_STUB_RC" | head -n 1)
    rc=$(echo "$line" | awk '{print $(NF - (/xid$/ ? 1 : 0))}')
    case "$line" in *xid) echo $(( $(cat "$RCSR_STUB_XID") + 1 )) > "$RCSR_STUB_XID" ;; esac
    case "$rc" in
      0) echo "test result: ok. 1 passed; 0 failed" ;;
      101) echo "test result: FAILED. 0 passed; 1 failed" ;;
    esac
    exit "$rc"
    ;;
esac
echo "test result: ok. 1 passed; 0 failed"
STUB
cat > "$T/slot.sh" <<'STUB'
#!/usr/bin/env bash
shift 2
echo "$1" >> "$RCSR_STUB_CMDS"
cat "$RCSR_STUB_BUILD"
STUB
chmod +x "$T/lock.sh" "$T/slot.sh"
art() { # kind name
  printf '{"reason":"compiler-artifact","profile":{"test":true},"executable":"/x/%s","target":{"name":"%s","kind":["%s"]}}\n' "$2" "$2" "$1"
}
{
  art lib rcsr
  art bin rcsr
  art test alpha
  art test beta
  art test gamma
  echo 'warning: noise line'
} > "$T/build.json"

# The real `rcsr-bench --tiny` summary: the 27/27 line first, then one line per noise level with smaller counts.
bench_out() { # summary-count
  cat <<EOB
== summary ==
recovered $1 (100.0%)   median complexity 3
by suite and noise level (Wilson 95% intervals):
  known-truth 0.000  runs    9  recovered 9/9 [0.701, 1.000]  oracle 9/9 [0.701, 1.000]  median excess_cx 0.0  median hit_s 0.00  seconds 0.0
  known-truth 0.010  runs    9  recovered 9/9 [0.701, 1.000]  oracle 9/9 [0.701, 1.000]  median excess_cx 0.0  median hit_s 0.00  seconds 1.0
  known-truth 0.100  runs    9  recovered 9/9 [0.701, 1.000]  oracle 9/9 [0.701, 1.000]  median excess_cx 0.0  median hit_s 0.00  seconds 0.9
EOB
}
bench_out 27/27 > "$T/bench-ok"
bench_out 26/27 > "$T/bench-low"
sed 1,2d "$T/bench-ok" > "$T/bench-nosummary"

run_gate() { # tag rc-file
  echo 0 > "$T/xid"
  echo 1000 > "$T/now"
  : > "$T/cmds"
  : > "$T/holds"
  RCSR_STUB_HOLDS=$T/holds RCSR_STUB_NOW=$T/now RCSR_NOW_CMD="cat $T/now" RCSR_STUB_BENCH=${RCSR_STUB_BENCH:-$T/bench-ok} RCSR_STUB_CMDS=$T/cmds RCSR_STUB_RC=$2 RCSR_STUB_XID=$T/xid RCSR_STUB_BUILD=$T/build.json \
    RCSR_MACHINE_DIR=$T RCSR_LOCK_SCRIPT=$T/lock.sh RCSR_SLOT_SCRIPT=$T/slot.sh RCSR_GATE_DIR=$T/gates RCSR_XID_CMD="cat $T/xid" \
    bash "$GATE" "$T/wt" "$1" > "$T/out-$1" 2>&1
}

LOCKP=$(cd "$T" && pwd -P)/full-suite.lock
# All green.
cat > "$T/rc-ok" <<'EOF'
--lib 0
--bin rcsr 0
--test alpha 0
--test beta 0
--test gamma 0
--doc 0
EOF
run_gate ok "$T/rc-ok"
check ok-exit "$?" 0
check ok-holds "$(grep -c -- '--features cuda --no-fail-fast' "$T/cmds")" 6
check ok-doc-hold "$(grep -c -- '--no-fail-fast --doc$' "$T/cmds")" 1
check ok-build "$(grep -c -- '--no-run --message-format=json' "$T/cmds")" 1
check ok-no-run-in-hold "$(grep -- '--no-fail-fast' "$T/cmds" | grep -c -- '--no-run')" 0
check ok-summary "$(grep -c 'rc=0' "$T/gates/ok-cuda-summary.log")" 6
check ok-line "$(grep -c '^cuda: targets=6 failed=0 timeouts=0$' "$T/out-ok")" 1
check ok-cpu-step "$(grep -c 'cargo test --release --no-fail-fast$' "$T/cmds")" 1
check ok-bench-line "$(grep -c "^bench: recovered 27/27 (100.0%)" "$T/out-ok")" 1
check ok-wait-run "$(grep -c " wait=7 run=30 " "$T/gates/ok-cuda-summary.log")" 6
check ok-no-secs "$(grep -c " secs=" "$T/gates/ok-cuda-summary.log")" 0
# every CPU, CUDA and bench hold runs through lock-run.sh on the lock path of the machine dir
check ok-holds-reaped "$(grep -c "lock-run.sh $LOCKP bash -c" "$T/holds")" 8
check ok-holds-total "$(wc -l < "$T/holds")" 8

# Every CUDA hold stays under the cap.
cap=$(bash -c ". '$HERE/../lib.sh'; echo \$LOCK_HOLD_MAX")
over=0
while IFS= read -r t; do
  [ "$t" -le "$cap" ] || over=$((over + 1))
done < <(grep -- '--features cuda --no-fail-fast' "$T/cmds" | sed -n 's/.* timeout -k 30 \([0-9]*\) cargo.*/\1/p')
check hold-under-cap "$over" 0
check hold-timeouts-found "$(grep -- '--features cuda --no-fail-fast' "$T/cmds" | grep -c ' timeout -k 30 [0-9]* cargo')" 6
# every timeout the gate starts has a kill-after: none is left bare
check no-bare-timeout "$(grep -cE 'timeout [0-9]' "$T/cmds")" 0
check cpu-kill-after "$(grep -c 'timeout -k 30 5400 cargo test --release --no-fail-fast$' "$T/cmds")" 1
check cuda-require-device "$(grep -- "--features cuda --no-fail-fast" "$T/cmds" | grep -c "RCSR_REQUIRE_DEVICE=1")" 6
check cpu-no-require-device "$(grep -c "RCSR_REQUIRE_DEVICE" <<< "$(grep -v -- "--features cuda" "$T/cmds")")" 0
check bench-kill-after "$(grep -c 'timeout -k 30 5400 cargo build.*timeout -k 30 1800 ' "$T/cmds")" 1

# The first target times out: the others still run and the gate fails.
cat > "$T/rc-timeout" <<'EOF'
--lib 124
--bin rcsr 0
--test alpha 0
--test beta 0
--test gamma 0
--doc 0
EOF
run_gate timeout "$T/rc-timeout"
check timeout-exit-nonzero "$([ "$?" -ne 0 ] && echo yes || echo no)" yes
check timeout-all-ran "$(grep -c -- '--features cuda --no-fail-fast' "$T/cmds")" 6
check timeout-labelled "$(grep -c '^lib rc=124 .*| timeout$' "$T/gates/timeout-cuda-summary.log")" 1
check timeout-others-ok "$(grep -c 'rc=0' "$T/gates/timeout-cuda-summary.log")" 5
check timeout-line "$(grep -c '^cuda: targets=6 failed=1 timeouts=1$' "$T/out-timeout")" 1

# The 27/27 summary is the only bench verdict: a summary below 27/27 fails with every level at 9/9, no summary line fails.
RCSR_STUB_BENCH=$T/bench-low run_gate lowsum "$T/rc-ok"
check lowsum-exit-nonzero "$([ "$?" -ne 0 ] && echo yes || echo no)" yes
check lowsum-bench-line "$(grep -c "^bench: recovered 26/27" "$T/out-lowsum")" 1
RCSR_STUB_BENCH=$T/bench-nosummary run_gate nosum "$T/rc-ok"
check nosum-exit-nonzero "$([ "$?" -ne 0 ] && echo yes || echo no)" yes
check nosum-bench-line "$(grep -c "^bench: no recovered line" "$T/out-nosum")" 1

# A failing target fails the gate, the others still run.
cat > "$T/rc-fail" <<'EOF'
--lib 0
--bin rcsr 0
--test alpha 101
--test beta 0
--test gamma 0
--doc 0
EOF
run_gate fail "$T/rc-fail"
check fail-exit-nonzero "$([ "$?" -ne 0 ] && echo yes || echo no)" yes
check fail-all-ran "$(grep -c -- '--features cuda --no-fail-fast' "$T/cmds")" 6
check fail-line "$(grep -c '^cuda: targets=6 failed=1 timeouts=0$' "$T/out-fail")" 1

# An Xid rise after a target stops the remaining holds and exits 70.
cat > "$T/rc-xid" <<'EOF'
--lib 0
--bin rcsr 0
--test alpha 0 xid
--test beta 0
--test gamma 0
--doc 0
EOF
run_gate xid "$T/rc-xid"
check xid-exit "$?" 70
check xid-stopped "$(grep -c -- '--features cuda --no-fail-fast' "$T/cmds")" 1

# A build that fails runs no CUDA hold.
echo 'error: build failed' > "$T/build.json"
run_gate nobuild "$T/rc-ok"
check nobuild-exit-nonzero "$([ "$?" -ne 0 ] && echo yes || echo no)" yes
check nobuild-no-holds "$(grep -c -- '--features cuda --no-fail-fast' "$T/cmds")" 0

if [ "$fails" -ne 0 ]; then
  echo "test-gate: $fails failed"
  exit 1
fi
echo "test-gate: ok"
