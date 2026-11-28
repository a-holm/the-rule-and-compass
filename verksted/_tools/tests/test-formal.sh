#!/usr/bin/env bash
# Usage: test-formal.sh
# Hermetic test of formal.sh with a stub build slot, stub lake, elan, leanchecker and cargo on a fixture worktree.
# Writes: a temp dir that it removes.
set -uo pipefail
export RCSR_CRATE=crate-x
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/lib/machine.sh"
. "$HERE/lib/fakes.sh"
FORMAL=$HERE/../formal.sh
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}
has() {
  if ! grep -qE -- "$3" "$2"; then
    echo "FAIL $1: '$3' not in $2"
    fails=$((fails + 1))
  fi
}

REL=leanprover/lean4:v9.9.9
mkdir -p "$T/gates" "$T/bin" "$T/elan"
printf '#!/usr/bin/env bash\necho "slot $*" >> "%s/calls"\nexec "$@"\n' "$T" > "$T/slot.sh"
chmod +x "$T/slot.sh"
cat > "$T/elan/elan" <<'STUB'
#!/usr/bin/env bash
echo "${FAKE_ELAN_LIST:-leanprover/lean4:v9.9.9 (default)}"
STUB
cat > "$T/elan/lake" <<'STUB'
#!/usr/bin/env bash
echo "lake $*" >> "$FAKE_DIR/calls"
case "$*" in
  "build --wfail")
    mkdir -p .lake/build/bin
    cp "$FAKE_DIR/checker" .lake/build/bin/rcsr_model_check
    exit "${FAKE_BUILD_RC:-0}" ;;
  "env lean --run Audit.lean")
    cat "${FAKE_AUDIT:-$FAKE_DIR/audit-ok}"
    exit "${FAKE_AUDIT_RC:-0}" ;;
  "env leanchecker "*)
    echo "leanchecker ${3}" >> "$FAKE_DIR/replays"
    [ "$3" = "${FAKE_REPLAY_CRASH:-@none@}" ] && exit 139
    exit 0 ;;
esac
echo "fake lake: unexpected $*" >&2
exit 1
STUB
cat > "$T/checker" <<'STUB'
#!/usr/bin/env bash
case "${FAKE_CHECK_MODE:-ok}" in
  empty) exit 0 ;;
  mismatch) for f in "$@"; do echo "$(basename "$f" .vec): 10 vectors, 1 mismatches"; done; exit 1 ;;
esac
for f in "$@"; do echo "$(basename "$f" .vec): 10 vectors, 0 mismatches"; done
STUB
cat > "$T/bin/cargo" <<'STUB'
#!/usr/bin/env bash
echo "cargo $*" >> "$FAKE_DIR/calls"
mkdir -p target/release
cp "$FAKE_DIR/vecbin" target/release/rcsr-formal-vectors
STUB
cat > "$T/vecbin" <<'STUB'
#!/usr/bin/env bash
echo "vecbin $*" >> "$FAKE_DIR/vecbin-calls"
printf 'rcsr-vec 1 cmp_error sig=w,w->w\nend 0\n' > "$2/cmp_error.vec"
STUB
chmod +x "$T/elan/elan" "$T/elan/lake" "$T/checker" "$T/bin/cargo" "$T/vecbin"
fake_timeout "$T/bin"

W=$T/wt
mkdir -p "$W/formal/core/RcsrCore" "$W/formal/vectors/golden" "$W/crate-x"
echo "$REL" > "$W/formal/core/lean-toolchain"
printf 'import RcsrCore.Prim\nimport RcsrCore.Alpha\n' > "$W/formal/core/RcsrCore.lean"
printf 'macro "rcsr_bv" : tactic => `(tactic| bv_decide)\n' > "$W/formal/core/RcsrCore/Prim.lean"
printf 'theorem alpha : True := trivial\n' > "$W/formal/core/RcsrCore/Alpha.lean"
printf 'Rcsr.alpha kernel\nRcsr.beta bv_decide\n' > "$W/formal/THEOREMS.txt"
printf 'rcsr-vec 1 cmp_error sig=w,w->w\nend 0\n' > "$W/formal/vectors/golden/cmp_error.vec"
touch "$W/crate-x/Cargo.toml"
cp "$W/formal/THEOREMS.txt" "$T/audit-ok"
printf 'Rcsr.alpha kernel\nRcsr.beta bv_decide\nRcsr.gamma kernel\n' > "$T/audit-extra"
git init -q "$W"
git -C "$W" add -A
git -C "$W" -c user.name=t -c user.email=t@t.invalid commit -q -m init
head=$(git -C "$W" rev-parse HEAD)
want_seed=$((16#${head:0:15}))

# run <extra env assignments...>: formal.sh on the fixture; stdout in $T/out, stderr in $T/err
run() {
  rm -rf "$W/formal/core/.lake" "$W/crate-x/target"
  : > "$T/calls"; : > "$T/replays"; : > "$T/vecbin-calls"; : > "$T/timeouts"; rm -f "$T"/gates/*
  env FAKE_DIR="$T" RCSR_MACHINE_DIR="$T/no-machine-dir" RCSR_GATES="$T/gates" RCSR_SLOT_SCRIPT="$T/slot.sh" RCSR_ELAN_BIN="$T/elan" FAKE_TIMEOUT_LOG="$T/timeouts" \
    PATH="$T/bin:$PATH" "$@" bash "$FORMAL" "$W" t1 > "$T/out" 2> "$T/err"
}
log=$T/gates/t1-formal.log

run; check ok "$?" 0
has ok-summary "$T/out" "^formal: theorems=2 bv_decide=1 vectors=20 mismatches=0 lean=$REL mathlib=none\$"
has ok-tested "$T/out" "^tested: $head\$"
check ok-no-failed "$(grep -c FAILED "$T/out")" 0
check ok-theorems-file "$(cat "$T/gates/t1-formal-theorems.txt")" "$(cat "$W/formal/THEOREMS.txt")"
check ok-vec-dir-removed "$(ls "$T/gates" | grep -c formal-vec)" 0

check ok-through-slot-lake "$(grep -c '^lake ' "$T/calls")" "$(grep -c '^slot .*lake ' "$T/calls")"
check ok-through-slot-cargo "$(grep -c '^cargo ' "$T/calls")" "$(grep -c '^slot .*cargo ' "$T/calls")"
check ok-slot-calls "$(grep -c '^slot ' "$T/calls")" 8
has ok-slot-lake "$T/calls" '^slot bash -c .*lake build --wfail'
has ok-slot-cargo "$T/calls" '^slot bash -c .*cargo build --release --bin rcsr-formal-vectors'

check ok-replay-per-module "$(sort "$T/replays" | tr '\n' ' ')" "leanchecker RcsrCore.Alpha leanchecker RcsrCore.Prim "
check ok-replay-calls "$(grep -c '^lake env leanchecker ' "$T/calls")" 2
has ok-seed "$T/vecbin-calls" " --count 100000 --seed $want_seed\$"

run RCSR_ELAN_BIN=; r=$?
check no-key "$r" 3
has no-key-msg "$T/err" ELAN_BIN
mkdir -p "$T/empty"
run RCSR_ELAN_BIN="$T/empty"; check no-lake "$?" 3
run RCSR_ELAN_BIN=/nonexistent; check nonexistent-dir "$?" 3
run FAKE_ELAN_LIST=leanprover/lean4:v1.0.0; r=$?
check no-release "$r" 3
has no-release-msg "$T/err" "elan toolchain install $REL"

printf 'theorem bad : True := by sorry\n' >> "$W/formal/core/RcsrCore/Alpha.lean"
run; r=$?
check refused-word "$r" 1
has refused-word-step "$T/out" '^source-rules: FAILED$'
has refused-word-log "$log" 'RcsrCore/Alpha.lean:2:theorem bad'
printf 'theorem alpha : True := trivial\n' > "$W/formal/core/RcsrCore/Alpha.lean"

printf 'theorem sat : True := by bv_decide\n' >> "$W/formal/core/RcsrCore/Alpha.lean"
run; r=$?
check sat-outside-macro "$r" 1
has sat-outside-macro-step "$T/out" '^source-rules: FAILED$'
printf 'theorem alpha : True := trivial\n' > "$W/formal/core/RcsrCore/Alpha.lean"

printf 'theorem g : True := trivial\n' > "$W/formal/core/RcsrCore/Gamma.lean"
run; r=$?
check not-imported "$r" 1
has not-imported-log "$log" 'Gamma.lean is not imported by RcsrCore.lean'
rm "$W/formal/core/RcsrCore/Gamma.lean"

mkdir -p "$W/formal/math"
echo leanprover/lean4:v1.0.0 > "$W/formal/math/lean-toolchain"
run; r=$?
check toolchains-differ "$r" 1
has toolchains-differ-step "$T/out" '^toolchain: FAILED$'
rm -r "$W/formal/math"

run FAKE_BUILD_RC=1; r=$?
check build-fails "$r" 1
has build-fails-step "$T/out" '^build: FAILED \(rc=1\)$'
has build-fails-later "$T/out" '^bulk: '

run FAKE_AUDIT_RC=1; r=$?
check audit-fails "$r" 1
has audit-fails-step "$T/out" '^audit: FAILED \(rc=1\)$'

run FAKE_AUDIT="$T/audit-extra"; r=$?
check inventory-differs "$r" 1
has inventory-differs-step "$T/out" '^inventory: FAILED$'
has inventory-differs-build "$T/out" '^build: ok$'
has inventory-differs-audit "$T/out" '^audit: ok$'

run FAKE_REPLAY_CRASH=RcsrCore.Alpha; r=$?
check replay-crash "$r" 1
has replay-crash-step "$T/out" '^replay-RcsrCore.Alpha: FAILED \(rc=139\)$'
check replay-crash-no-retry "$(grep -c 'leanchecker RcsrCore.Alpha' "$T/replays")" 1

run FAKE_CHECK_MODE=mismatch; r=$?
check golden-mismatch "$r" 1
has golden-mismatch-summary "$T/out" 'mismatches=2 '
has golden-mismatch-step "$T/out" '^golden: FAILED \(rc=1\)$'

run FAKE_CHECK_MODE=empty; r=$?
check empty-run "$r" 1
has empty-run-summary "$T/out" 'vectors=0 '

env FAKE_DIR="$T" RCSR_MACHINE_DIR="$T/no-machine-dir" RCSR_GATES="$T/gates" RCSR_SLOT_SCRIPT="$T/slot.sh" bash "$FORMAL" "$W" > /dev/null 2>&1; check usage-one "$?" 2
env FAKE_DIR="$T" RCSR_MACHINE_DIR="$T/no-machine-dir" RCSR_GATES="$T/gates" RCSR_SLOT_SCRIPT="$T/slot.sh" bash "$FORMAL" a b c > /dev/null 2>&1; check usage-three "$?" 2
bash "$FORMAL" --help > "$T/help" 2>&1; check help "$?" 0
has help-usage "$T/help" '^Usage: formal.sh <worktree> <tag>$'

[ "$fails" -eq 0 ] || { echo "$fails failed"; exit 1; }
echo "test-formal: ok"
