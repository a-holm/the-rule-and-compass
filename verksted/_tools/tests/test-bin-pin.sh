#!/usr/bin/env bash
# Usage: test-bin-pin.sh
# Hermetic test of bin-pin.sh with a scratch git worktree and stub binaries.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PIN=$HERE/../bin-pin.sh
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

wt=$T/wt
rel=$wt/my-version/target/release
mkdir -p "$rel"
git -C "$wt" init -q
echo target > "$wt/.gitignore"
git_q -C "$wt" add .gitignore
git_q -C "$wt" commit -q -m init
head=$(git -C "$wt" rev-parse HEAD)
printf '#!/bin/sh\necho stub 1\n' > "$rel/rcsr"
printf '#!/bin/sh\necho stub 1\n' > "$rel/other"
chmod +x "$rel/rcsr" "$rel/other"

pin() { bash "$PIN" "$@" > "$T/out" 2>&1; }

# clean pin
pin "$wt" "$T/d1"; rc=$?
check pin-rc "$rc" "0"
check pin-copy "$(cmp -s "$rel/rcsr" "$T/d1/rcsr" && echo same)" "same"
check pin-exec "$([ -x "$T/d1/rcsr" ] && echo yes)" "yes"
check pin-sha "$(cat "$T/d1/rcsr.sha256")" "$(sha256sum "$rel/rcsr" | awk '{print $1}')  rcsr"
check pin-sha-verifies "$(cd "$T/d1" && sha256sum -c rcsr.sha256 > /dev/null 2>&1 && echo ok)" "ok"
check pin-rev "$(cat "$T/d1/rcsr.rev")" "$head"

# second identical pin
pin "$wt" "$T/d1"; rc=$?
check repin-rc "$rc" "0"

# a different existing pin
printf '#!/bin/sh\necho stub 2\n' > "$rel/rcsr"
pin "$wt" "$T/d1"; rc=$?
check different-rc "$rc" "1"
check different-kept "$("$T/d1/rcsr")" "stub 1"

# several names
pin "$wt" "$T/d2" other; rc=$?
check name-rc "$rc" "0"
check name-only "$(ls "$T/d2" | tr '\n' ' ')" "other other.rev other.sha256 "

# dirty tree
echo x > "$wt/dirty.txt"
pin "$wt" "$T/d3"; rc=$?
check dirty-rc "$rc" "0"
check dirty-rev "$(cat "$T/d3/rcsr.rev")" "$head-dirty"
rm "$wt/dirty.txt"

# a binary that fails under env -i
printf '#!/bin/sh\n[ -n "${NEED_BUILD_DIR:-}" ] || exit 127\necho ok\n' > "$rel/rcsr"
NEED_BUILD_DIR=1 bash "$PIN" "$wt" "$T/d4" > "$T/out" 2>&1; rc=$?
check envi-rc "$rc" "1"
check envi-no-pin "$([ -e "$T/d4/rcsr" ] && echo yes || echo no)" "no"
check envi-no-sha "$([ -e "$T/d4/rcsr.sha256" ] && echo yes || echo no)" "no"

# missing binary
pin "$wt" "$T/d5" nothere; rc=$?
check missing-rc "$rc" "1"

# RUNPATH into a build dir and NEEDED librcsrcuda
if command -v readelf > /dev/null && command -v cc > /dev/null; then
  printf 'int main(void){return 0;}\n' > "$T/m.c"
  if cc -o "$rel/rcsr" "$T/m.c" -Wl,--enable-new-dtags -Wl,-rpath,/x/build/y 2> /dev/null; then
    pin "$wt" "$T/d6"; rc=$?
    check runpath-rc "$rc" "1"
    check runpath-msg "$(grep -c RUNPATH "$T/out")" "1"
  else
    echo "skip runpath: cc failed"
  fi
  if cc -shared -fPIC -o "$T/librcsrcuda.so" "$T/m.c" 2> /dev/null &&
    cc -o "$rel/rcsr" "$T/m.c" -Wl,--no-as-needed "$T/librcsrcuda.so" 2> /dev/null; then
    pin "$wt" "$T/d7"; rc=$?
    check needed-rc "$rc" "1"
    check needed-msg "$(grep -c librcsrcuda "$T/out")" "1"
  else
    echo "skip needed: cc failed"
  fi
else
  echo "skip elf checks: readelf or cc missing"
fi

# never pin from default-ab-run
mkdir -p "$T/default-ab-run"
pin "$T/default-ab-run" "$T/d8"; rc=$?
check default-ab-run "$rc" "1"

# usage
pin; rc=$?
check usage-rc "$rc" "2"
pin "$wt"; rc=$?
check usage-one-arg "$rc" "2"
pin -h; rc=$?
check help-rc "$rc" "0"

if [ "$fails" -ne 0 ]; then
  echo "test-bin-pin: $fails failed"
  exit 1
fi
echo "test-bin-pin: ok"
