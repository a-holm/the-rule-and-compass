#!/usr/bin/env bash
# Usage: test-runlib.sh
# Hermetic test of runlib.sh: steps, done-files, FORCE, list, protected steps, profiles, pinned binary, Xid, chunks.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RLIB=$HERE/../runlib.sh
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

# Fake lock and slot run the command directly and count their holds.
cat > "$T/lock.sh" <<EOF
#!/usr/bin/env bash
echo hold >> "$T/holds"
exec "\$@"
EOF
cat > "$T/slot.sh" <<EOF
#!/usr/bin/env bash
echo slot >> "$T/slots"
exec "\$@"
EOF
# Fake time prints a [time] line whose wall comes from the file $T/wall (default 1).
cat > "$T/time.sh" <<EOF
#!/usr/bin/env bash
shift 2
"\$@"
rc=\$?
w=\$(cat "$T/wall" 2>/dev/null || echo 1)
echo "[time] wall=\$w s user=0 s sys=0 s maxrss=1 KB rc=\$rc" >&2
exit \$rc
EOF
# Fake search: reports time_budget for the first \$(cat reasons_n) calls, then converged.
cat > "$T/search.sh" <<EOF
#!/usr/bin/env bash
n=\$(cat "$T/calls" 2>/dev/null || echo 0)
n=\$((n + 1)); echo \$n > "$T/calls"
echo "chunk_s=\$RL_CHUNK_S n=\$RL_CHUNK_N"
if [ \$n -le \$(cat "$T/tb") ]; then echo time_budget > "$T/stop_reason"; else echo converged > "$T/stop_reason"; fi
EOF
chmod +x "$T"/*.sh
echo 0 > "$T/xid"
cat > "$T/xid.sh" <<EOF
cat "$T/xid"
EOF

# Run a runbook snippet. Extra env goes first as args; the snippet defines steps then calls runlib_main.
rb() {
  local snippet=$1
  shift
  env RCSR_GATES="$T" RCSR_MACHINE_DIR="$T" RCSR_LOCK_SCRIPT="$T/lock.sh" RCSR_SLOT_SCRIPT="$T/slot.sh" \
    RCSR_XID_CMD="bash $T/xid.sh" RL_TIME="$T/time.sh" RUN="$T/run" "$@" \
    bash -c "set -uo pipefail; STEPS=(one two-b three); PROTECTED=(three); \
      step_one() { echo one >> '$T/order'; }; step_two_b() { echo two >> '$T/order'; }; \
      step_three() { echo three >> '$T/order'; }; \
      source '$RLIB'; $snippet" > "$T/out" 2>&1
}

# ordered run, done-files, protected step skipped without SEALED
rb 'runlib_main'; check run-rc $? 0
check order "$(tr '\n' ' ' < "$T/order")" "one two "
[ -f "$T/run/done/one.done" ] && [ -f "$T/run/done/two-b.done" ]; check done-files $? 0
[ ! -f "$T/run/done/three.done" ]; check protected-skipped $? 0
grep -q 'skip three (protected' "$T/run/run.log"; check protected-skip-logged $? 0

# second run skips; FORCE reruns
rm "$T/order"
rb 'runlib_main'; check rerun-rc $? 0
check rerun-skips "$(cat "$T/order" 2>/dev/null)" ""
rb 'runlib_main one' FORCE=1; check force-rc $? 0
check force "$(tr '\n' ' ' < "$T/order")" "one "

# list
rb 'runlib_main list'; check list-rc $? 0
check list-pending "$(grep -c 'three.pending' "$T/out")" 1
check list-done "$(grep -cE '^one	20[0-9-]+T[0-9:]+Z$' "$T/out")" 1

# unknown step, protected step
rb 'runlib_main nope'; check unknown-rc $? 1
grep -q "unknown step 'nope'" "$T/out"; check unknown-msg $? 0
rb 'runlib_main three'; check protected-rc $? 1
[ ! -f "$T/run/done/three.done" ]; check protected-nodone $? 0
rb 'runlib_main three' SEALED=1; check sealed-rc $? 0
[ -f "$T/run/done/three.done" ]; check sealed-done $? 0

# failing step stops the run without a done-file
rm -rf "$T/run"
rb 'step_one() { return 3; }; runlib_main'; check failstep-rc $? 1
[ ! -f "$T/run/done/one.done" ] && [ ! -f "$T/run/done/two-b.done" ]; check failstep-nodone $? 0

# profiles
rb 'runlib_main list' PROFILE=smoke; check profile-unknown $? 1
rb 'profile_smoke() { echo smoke-set; }; runlib_main list' PROFILE=smoke; check profile-known $? 0
grep -q smoke-set "$T/out"; check profile-called $? 0

# rl_gpu: [time] and [xid] lines, rc passthrough, one hold
rm -f "$T/holds" "$T/run/g.log"
rb 'rl_gpu g 30 -- echo hi; echo "rc=$?"'; check gpu-rc $? 0
grep -q '^hi$' "$T/run/g.log"; check gpu-output $? 0
grep -q '^\[time\] wall=1 s' "$T/run/g.log"; check gpu-time $? 0
grep -q '^\[xid\] before=0 after=0 rc=0$' "$T/run/g.log"; check gpu-xid $? 0
check gpu-holds "$(wc -l < "$T/holds")" 1
rb 'rl_gpu g2 30 -- bash -c "exit 4"; echo "rc=$?"'
grep -q '^rc=4$' "$T/out"; check gpu-passes-rc $? 0
grep -q '^\[xid\] before=0 after=0 rc=4$' "$T/run/g2.log"; check gpu-xid-rc $? 0

# refusal above LOCK_HOLD_MAX
rb 'rl_gpu big 900 -- true'; check refuse-rc $? 1
grep -q 'refused: rl_gpu big timeout 900s is above LOCK_HOLD_MAX=600s' "$T/out"; check refuse-msg $? 0
[ ! -f "$T/run/big.log" ]; check refuse-nohold $? 0

# rising Xid stops
rb 'rl_gpu x 30 -- bash -c "echo 5 > '"$T"'/xid"'; check xid-rc $? 1
grep -q 'Xid count rose 0 -> 5' "$T/out"; check xid-msg $? 0
echo 0 > "$T/xid"

# rl_cpu uses the slot, no Xid line
rm -f "$T/slots"
rb 'rl_cpu c 30 -- echo cpu'; check cpu-rc $? 0
check cpu-slot "$(wc -l < "$T/slots")" 1
grep -q xid "$T/run/c.log"; check cpu-noxid $? 1

# pinned binary
printf '#!/bin/sh\necho pinned\n' > "$T/bin"; chmod +x "$T/bin"
(cd "$T" && sha256sum bin | cut -d' ' -f1 > bin.sha256)
rb 'rl_gpu p 30 -- "$RL_BIN"' RL_BIN="$T/bin"; check pin-ok $? 0
echo '# changed' >> "$T/bin"
rb 'rl_gpu p 30 -- "$RL_BIN"' RL_BIN="$T/bin"; check pin-changed $? 1
grep -q 'changed after it was pinned' "$T/out"; check pin-msg $? 0

# rl_chunks: time_budget twice, then converged: exactly 3 holds
rm -f "$T/holds" "$T/calls" "$T/run/s.log" "$T/stop_reason"
echo 2 > "$T/tb"; echo 1 > "$T/wall"
rb 'rl_chunks s 100 1000 "cat '"$T"'/stop_reason" -- '"$T"'/search.sh'; check chunks-rc $? 0
check chunks-holds "$(wc -l < "$T/holds")" 3
check chunks-lines "$(grep -c '^\[chunk\] n=' "$T/run/s.log")" 3
grep -q '^\[chunk\] n=3 wall=1 stop=converged$' "$T/run/s.log"; check chunks-last $? 0
grep -q '^chunk_s=100 n=1$' "$T/run/s.log"; check chunks-env $? 0

# budget smaller than two chunks: the budget counts hold wall, the last chunk is trimmed
rm -f "$T/holds" "$T/calls" "$T/run/s.log"
echo 100 > "$T/tb"; echo 40 > "$T/wall"
rb 'rl_chunks s 60 100 "cat '"$T"'/stop_reason" -- '"$T"'/search.sh'; check budget-rc $? 0
check budget-holds "$(wc -l < "$T/holds")" 3
grep -q '^chunk_s=60 n=1$' "$T/run/s.log" && grep -q '^chunk_s=60 n=2$' "$T/run/s.log" && grep -q '^chunk_s=20 n=3$' "$T/run/s.log"
check budget-trim $? 0
rm -f "$T/holds" "$T/calls" "$T/run/s.log"
echo 60 > "$T/wall"
rb 'rl_chunks s 100 50 "cat '"$T"'/stop_reason" -- '"$T"'/search.sh'; check budget-lt-chunk $? 0
check budget-lt-holds "$(wc -l < "$T/holds")" 1
grep -q '^chunk_s=50 n=1$' "$T/run/s.log"; check budget-lt-trim $? 0

# grace: a stub past its budget but inside the grace survives; past budget + grace it is killed
rm -f "$T/holds" "$T/calls" "$T/run/gr.log"; echo 1 > "$T/wall"; echo 0 > "$T/tb"
rb 'rl_chunks gr 1 1000 "echo converged" -- sleep 2' RL_GRACE_S=4; check grace-inside-rc $? 0
grep -q "timeout 5s" "$T/run/run.log"; check grace-timeout-sum $? 0
rb 'rl_chunks gk 1 1000 "echo converged" -- sleep 10' RL_GRACE_S=1; check grace-kill-rc $? 1
grep -q "chunk 1 failed rc=124" "$T/out"; check grace-kill-msg $? 0
rb 'rl_chunks gz 500 1000 "echo converged" -- true'; check grace-refuse-rc $? 1
grep -q "refused: rl_chunks gz chunk 500s + grace 120s is above LOCK_HOLD_MAX=600s" "$T/out"; check grace-refuse-msg $? 0

# a stop-cmd that prints nothing ends the loop
rm -f "$T/holds" "$T/calls" "$T/run/s.log"; echo 1 > "$T/wall"
rb 'rl_chunks s 100 1000 "true" -- '"$T"'/search.sh'; check silent-rc $? 0
check silent-holds "$(wc -l < "$T/holds")" 1

# a failing hold stops the chunk loop
rm -f "$T/holds" "$T/run/f.log"
rb 'rl_chunks f 100 1000 "echo time_budget" -- false'; check chunk-fail-rc $? 1
check chunk-fail-holds "$(wc -l < "$T/holds")" 1

# no pgrep, pkill or python outside comments
check no-forbidden "$(grep -vE '^\s*#' "$RLIB" | grep -cE 'pgrep|pkill|python')" 0

[ "$fails" -eq 0 ] && echo "test-runlib: pass" && exit 0
echo "test-runlib: $fails failure(s)"
exit 1
