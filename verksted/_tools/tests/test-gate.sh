#!/usr/bin/env bash
# Usage: test-gate.sh
# Hermetic test of gate.sh: both cargo test steps pass --no-fail-fast; a stub lock records the commands.
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
cat > "$T/lock.sh" <<'STUB'
#!/usr/bin/env bash
shift 2
echo "$1" >> "$RCSR_STUB_CMDS"
STUB
chmod +x "$T/lock.sh"
RCSR_STUB_CMDS=$T/cmds RCSR_LOCK_SCRIPT=$T/lock.sh RCSR_GATE_DIR=$T/gates bash "$GATE" "$T/wt" t > /dev/null 2>&1
check cpu-step "$(grep -c 'cargo test --release --no-fail-fast$' "$T/cmds")" 1
check cuda-step "$(grep -c 'cargo test --release --features cuda --no-fail-fast$' "$T/cmds")" 1
check test-steps "$(grep -c 'cargo test' "$T/cmds")" 2
if [ "$fails" -ne 0 ]; then
  echo "test-gate: $fails failed"
  exit 1
fi
echo "test-gate: ok"
