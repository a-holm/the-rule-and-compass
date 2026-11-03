#!/usr/bin/env bash
# Usage: machine-sync.sh [--check | --install]
# The canonical sources of the live machine scripts in $OC are verksted/_tools/machine/: one-full-suite.sh and
# desktop-busy.sh as byte copies, build-slot.sh with clean indentation.
# --check (default) compares each pair and prints, per file: `same`, `whitespace only` (diff -w is empty) or a unified diff.
# --install copies the canonical files over the live ones. Run it only when no pilot or A/B round uses them;
# it refuses while any process holds $LOCKFILE or a build slot.
# Writes: --check nothing; --install the files in $OC.
# Exit: 0 every pair same; 1 any difference or refused; 2 usage.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

FILES="one-full-suite.sh desktop-busy.sh build-slot.sh"
CANON=$VERKSTED/_tools/machine

mode=check
case "${1:-}" in
  "" | --check) ;;
  --install) mode=install ;;
  -h | --help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
[ $# -le 1 ] || { usage >&2; exit 2; }

holders() {
  find /proc/[0-9]*/fd -maxdepth 1 \( -lname "$LOCKFILE" -o -lname "$SLOTDIR/slot*.lock" \) -printf '%h\n' 2>/dev/null |
    cut -d/ -f3 | sort -un
}

if [ "$mode" = install ]; then
  busy=$(holders)
  if [ -n "$busy" ]; then
    echo "refused: lock or build slot held by pid $(echo $busy | tr ' ' ',')" >&2
    exit 1
  fi
  mkdir -p "$OC"
  rc=0
  for f in $FILES; do
    if cp -p "$CANON/$f" "$OC/$f"; then echo "installed $f"; else echo "failed $f" >&2; rc=1; fi
  done
  exit $rc
fi

rc=0
for f in $FILES; do
  live=$OC/$f
  if [ ! -e "$live" ]; then
    echo "$f: missing live file"
    rc=1
  elif cmp -s "$CANON/$f" "$live"; then
    echo "$f: same"
  elif diff -w -q "$live" "$CANON/$f" > /dev/null; then
    echo "$f: whitespace only"
    rc=1
  else
    echo "$f: differs"
    diff -u "$live" "$CANON/$f"
    rc=1
  fi
done
exit $rc
