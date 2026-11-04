#!/usr/bin/env bash
# Usage: test-machine-sync.sh
# Hermetic test of machine-sync.sh (same, whitespace only, diff, refused install, install) and of the
# canonical build-slot.sh (two jobs, one slot, no interleaving).
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/lib/fakes.sh"
TOOLS=$HERE/..
T=$(mktemp -d)
cleanup() { [ -z "${HOLD_PID:-}" ] || release; rm -rf "$T"; }
trap cleanup EXIT
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}
sync() { RCSR_MACHINE_DIR=$T/ocl bash "$TOOLS/machine-sync.sh" "$@"; }
FILES="one-full-suite.sh desktop-busy.sh build-slot.sh"

mkdir -p "$T/oc"
ln -s oc "$T/ocl"
for f in $FILES; do cp "$TOOLS/machine/$f" "$T/oc/$f"; done

out=$(sync --check); rc=$?
check same-rc "$rc" 0
check same-out "$(echo "$out" | grep -c ': same$')" 3

sed -i 's/^/  /' "$T/oc/build-slot.sh"
out=$(sync); rc=$?
check ws-rc "$rc" 1
check ws-out "$(echo "$out" | grep -c 'build-slot.sh: whitespace only')" 1

echo '# change' >> "$T/oc/desktop-busy.sh"
out=$(sync); rc=$?
check diff-rc "$rc" 1
check diff-out "$(echo "$out" | grep -c '^desktop-busy.sh: differs')" 1
check diff-body "$(echo "$out" | grep -c '^-# change')" 1

sync --bogus > /dev/null 2>&1
check usage-rc "$?" 2

hold "exec 9> $(printf %q "$T/ocl/full-suite.lock") && flock 9"; check holder-ready $? 0
sync --install > /dev/null 2>&1
check refused-rc "$?" 1
check refused-untouched "$(grep -c '# change' "$T/oc/desktop-busy.sh")" 1
release
sync --install > /dev/null 2>&1
check install-rc "$?" 0
sync --check > /dev/null
check recheck-rc "$?" 0

# build-slot.sh: a held slot is refused once, then the job runs once.
if [ ! -x /mnt/lager/tmp/opencode/desktop-busy.sh ]; then
  echo "SKIP build-slot serialisation: live desktop-busy.sh missing"
else
  mkdir -p "$T/fake" "$T/slots"
  fake_slot_sleep "$T/fake" "$T/polls"
  printf '#!/bin/sh\necho "0, 0"\n' > "$T/fake/nvidia-smi"; chmod +x "$T/fake/nvidia-smi"
  exec 9> "$T/slots/slot1.lock"; flock 9
  PATH="$T/fake:$PATH" RCSR_BUILD_SLOT_DIR=$T/slots RCSR_TEST_PAUSE=$T/nopause RCSR_BUILD_SLOTS=1 \
    RCSR_BUILD_SLOTS_DISCORD=1 RCSR_BUILD_MIN_AVAIL_KB=0 RCSR_CPUS=0 RCSR_CPUS_DISCORD=0 \
    bash "$TOOLS/machine/build-slot.sh" bash -c 'echo ran' > "$T/slot.out" 2> "$T/slot.err"; rc=$?
  exec 9>&-
  check slot-rc "$rc" 0
  check slot-polls "$(cat "$T/polls")" 1
  check slot-ran "$(cat "$T/slot.out")" ran
  check slot-held "$(grep -c '^\[build-slot\] slot 1/1 held: bash -c echo ran$' "$T/slot.err")" 1
fi

if [ "$fails" -ne 0 ]; then
  echo "$fails failure(s)"
  exit 1
fi
echo "test-machine-sync: ok"
