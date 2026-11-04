#!/usr/bin/env bash
# Usage: test-machine-sync.sh
# Hermetic test of machine-sync.sh (same, whitespace only, diff, refused install, install) and of the
# canonical build-slot.sh (two jobs, one slot, no interleaving).
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOOLS=$HERE/..
T=$(mktemp -d)
bg=""
cleanup() { [ -z "$bg" ] || kill $bg 2> /dev/null; rm -rf "$T"; }
trap cleanup EXIT
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}
sync() { RCSR_MACHINE_DIR=$T/oc bash "$TOOLS/machine-sync.sh" "$@"; }
FILES="one-full-suite.sh desktop-busy.sh build-slot.sh"

mkdir -p "$T/oc"
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

( exec 9> "$T/oc/full-suite.lock"; flock 9; exec sleep 60 ) &
bg=$!
for _ in $(seq 1 50); do
  flock -n "$T/oc/full-suite.lock" true 2> /dev/null || break
  sleep 0.1
done
sync --install > /dev/null 2>&1
check refused-rc "$?" 1
check refused-untouched "$(grep -c '# change' "$T/oc/desktop-busy.sh")" 1
kill $bg 2> /dev/null
wait $bg 2> /dev/null
bg=""

sync --install > /dev/null 2>&1
check install-rc "$?" 0
sync --check > /dev/null
check recheck-rc "$?" 0

# build-slot.sh: one slot serialises two jobs.
if [ ! -x /mnt/lager/tmp/opencode/desktop-busy.sh ]; then
  echo "SKIP build-slot serialisation: live desktop-busy.sh missing"
else
  export RCSR_BUILD_SLOT_DIR=$T/slots RCSR_TEST_PAUSE=$T/nopause RCSR_BUILD_SLOTS=1 RCSR_BUILD_MIN_AVAIL_KB=0
  job() { bash "$TOOLS/machine/build-slot.sh" bash -c 'echo start; sleep 1; echo end' 2> /dev/null; }
  { job & job & wait; } > "$T/jobs.out"
  check slot-order "$(tr '\n' ' ' < "$T/jobs.out")" "start end start end "
fi

if [ "$fails" -ne 0 ]; then
  echo "$fails failure(s)"
  exit 1
fi
echo "test-machine-sync: ok"
