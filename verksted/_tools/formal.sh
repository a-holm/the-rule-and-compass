#!/usr/bin/env bash
# Usage: formal.sh <worktree> <tag>
# The formal step of the full gate on an integration tip: toolchain check, source rules, Lean build with warnings as
# errors, trust audit and theorem inventory, kernel replay, golden vectors, bulk vectors from the release build.
# Every Lean and cargo command runs through the build slot. No network. No skip mode: a missing toolchain is exit 3.
# The Lean tools come from the directory of machine key ELAN_BIN (override: RCSR_ELAN_BIN).
# Prints: one line per step `<step>: ok|FAILED`, then
#   formal: theorems=<n> bv_decide=<n> vectors=<n> mismatches=<n> lean=<release> mathlib=<revision or none>
#   tested: <sha>
# Writes: $D/<tag>-formal.log (every step under a `=== <step> ===` header), $D/<tag>-formal-theorems.txt (the audit output).
# Exit: 0 every step passed; 1 a step failed; 2 usage; 3 the toolchain or its release is missing.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac
[ $# -eq 2 ] || { usage >&2; exit 2; }
W=$1
T=$2
C=$(crate_dir "$W") || { echo "formal: no single engine crate under $W" >&2; exit 2; }
D=${RCSR_GATE_DIR:-$GATES}
[ -d "$D" ] || { echo "gate dir $D missing; link it to the log directory or set RCSR_GATES" >&2; exit 1; }
F=$W/formal
CORE=$F/core
log=$D/$T-formal.log
: > "$log"

ELAN=${RCSR_ELAN_BIN:-}
[ -n "$ELAN" ] || ELAN=$(machine_key ELAN_BIN) || { echo "formal: machine key ELAN_BIN is not set (the directory that holds lake and elan)" >&2; exit 3; }
[ -x "$ELAN/lake" ] && [ -x "$ELAN/elan" ] || { echo "formal: no lake or elan in $ELAN (machine key ELAN_BIN)" >&2; exit 3; }
release=$(tr -d '[:space:]' < "$CORE/lean-toolchain" 2> /dev/null)
[ -n "$release" ] || { echo "formal: $CORE/lean-toolchain is missing or empty" >&2; exit 1; }
"$ELAN/elan" toolchain list | grep -qF "$release" || { echo "formal: toolchain $release is not installed; run: elan toolchain install $release" >&2; exit 3; }

bad=0
# run <step> <command>: the command through the build slot, output appended to the log
run() {
  local name=$1 rc
  shift
  echo "=== $name ===" >> "$log"
  "$B" bash -c "$1" >> "$log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then echo "$name: ok"; else echo "$name: FAILED (rc=$rc)"; bad=1; fi
  return "$rc"
}
# note <step> <rc>: result of a check that ran in this shell
note() {
  if [ "$2" -eq 0 ]; then echo "$1: ok"; else echo "$1: FAILED"; bad=1; fi
}
lean="cd '$CORE' && PATH='$ELAN':\$PATH"

# 1. the packages share one toolchain release
rc=0
echo "=== toolchain ===" >> "$log"
if [ -f "$F/math/lean-toolchain" ]; then cmp "$CORE/lean-toolchain" "$F/math/lean-toolchain" >> "$log" 2>&1 || rc=1; fi
note toolchain "$rc"

# 2. source rules
rc=0
echo "=== source-rules ===" >> "$log"
if grep -rnwE 'sorry|axiom|native_decide|implemented_by|extern|unsafe|csimp|opaque' --include='*.lean' --exclude-dir=.lake "$F" >> "$log"; then rc=1; fi
if grep -rnw 'bv_decide' --include='*.lean' --exclude-dir=.lake "$F" | grep -v -e "^$CORE/Audit.lean:" -e "^$CORE/RcsrCore/Prim.lean:[0-9]*:macro \"rcsr_bv\"" >> "$log"; then rc=1; fi
modules=()
while IFS= read -r f; do
  m=${f#"$CORE"/}; m=${m%.lean}; m=${m//\//.}
  modules+=("$m")
  grep -qx "import $m" "$CORE/RcsrCore.lean" || { echo "$f is not imported by RcsrCore.lean" >> "$log"; rc=1; }
done < <(find "$CORE/RcsrCore" -name '*.lean' | LC_ALL=C sort)
note source-rules "$rc"

# 3. build, warnings are errors
run build "$lean timeout -k $KILL_AFTER 3000 lake build --wfail"

# 4. trust audit and theorem inventory
inv=$D/$T-formal-theorems.txt
run audit "$lean timeout -k $KILL_AFTER 1500 lake env lean --run Audit.lean > '$inv'"
rc=0
echo "=== inventory ===" >> "$log"
diff "$inv" "$F/THEOREMS.txt" >> "$log" 2>&1 || rc=1
note inventory "$rc"

# 5. kernel replay, one module per call
for m in "${modules[@]}"; do
  run "replay-$m" "$lean timeout -k $KILL_AFTER 900 lake env leanchecker $m"
done

# 6. golden vectors
check=$CORE/.lake/build/bin/rcsr_model_check
run golden "'$check' '$F'/vectors/golden/*.vec"

# 7. bulk vectors from the release build of this tree
vec=$(mktemp -d "$D/$T-formal-vec.XXXXXX")
trap 'rm -rf "$vec"' EXIT
seed=$((16#$(git -C "$W" rev-parse HEAD | cut -c1-15)))
run bulk-build "cd '$C' && CARGO_BUILD_JOBS=2 timeout -k $KILL_AFTER 5400 cargo build --release --bin rcsr-formal-vectors"
run bulk-write "'$C/target/release/rcsr-formal-vectors' --bulk '$vec' --count 100000 --seed $seed"
run bulk "'$check' '$vec'/*.vec"

theorems=$(grep -c . "$inv" 2> /dev/null)
sat=$(grep -c ' bv_decide$' "$inv" 2> /dev/null)
vectors=$(grep -aE '^[a-z_0-9]+: [0-9]+ vectors, [0-9]+ mismatches$' "$log" | awk '{ n += $2 } END { print n + 0 }')
mismatches=$(grep -aE '^[a-z_0-9]+: [0-9]+ vectors, [0-9]+ mismatches$' "$log" | awk '{ n += $4 } END { print n + 0 }')
echo "formal: theorems=${theorems:-0} bv_decide=${sat:-0} vectors=$vectors mismatches=$mismatches lean=$release mathlib=none"
echo "tested: $(git -C "$W" rev-parse HEAD)"
[ "$bad" -eq 0 ] && [ "$mismatches" -eq 0 ] && [ "${theorems:-0}" -gt 0 ] && [ "$vectors" -gt 0 ]
