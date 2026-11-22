#!/usr/bin/env bash
# Usage: test-crate-per-worktree.sh
# Hermetic test: with RCSR_ROOT holding a crate directory named differently from the worktree's, bin-pin.sh,
# sanitize-all.sh and gate.sh use the worktree's crate directory.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
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
git_q() { git -c user.name=t -c user.email=t@t.invalid "$@"; }
pkg() { mkdir -p "$1"; printf '[package]\nname = "rule-and-compass"\n' > "$1/Cargo.toml"; }
unset RCSR_CRATE

root=$T/root
wt=$T/wt
pkg "$root/old-name"
pkg "$wt/new-name"
export RCSR_ROOT=$root
git -C "$root" init -q
echo target > "$root/.gitignore"
git_q -C "$root" add . && git_q -C "$root" commit -q -m init
git -C "$wt" init -q
echo target > "$wt/.gitignore"
git_q -C "$wt" add .gitignore && git_q -C "$wt" commit -q -m init

rel=$wt/new-name/target/release
mkdir -p "$rel"
printf '#!/bin/sh\necho stub\n' > "$rel/rcsr"
chmod +x "$rel/rcsr"
bash "$HERE/../bin-pin.sh" "$wt" "$T/pin" > "$T/out" 2>&1
check bin-pin-rc "$?" "0"
check bin-pin-copy "$(cmp -s "$rel/rcsr" "$T/pin/rcsr" && echo same)" "same"

bash "$HERE/../sanitize-all.sh" "$wt" r1 --dry-run > "$T/out" 2>&1
check sanitize-all-plan "$(grep -c "plan $wt/new-name/tests/sanitizer.plan missing" "$T/out")" "1"

mkdir -p "$T/gates"
pkg "$wt/second"
RCSR_GATES=$T/gates bash "$HERE/../gate.sh" "$wt" t1 > "$T/out" 2>&1
check gate-two-crates-rc "$?" "2"
check gate-two-crates-msg "$(grep -c "no single engine crate under $wt" "$T/out")" "1"

[ "$fails" -eq 0 ] && echo "PASS test-crate-per-worktree.sh" && exit 0
exit 1
