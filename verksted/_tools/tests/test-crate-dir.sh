#!/usr/bin/env bash
# Usage: test-crate-dir.sh
# Hermetic test of crate-dir.sh: one match, none, two, and a [lib] or [[bin]] entry alone.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CD=$HERE/../crate-dir.sh
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fails=0
check() {
  local name=$1 got=$2 want=$3
  if [ "$got" != "$want" ]; then
    echo "FAIL $name: got '$got' want '$want'"
    fails=$((fails + 1))
  else
    echo "ok case $name"
  fi
}
pkg() { mkdir -p "$1"; printf '[package]\nname = "rule-and-compass"\nversion = "0.1.0"\n' > "$1/Cargo.toml"; }
run() { env -u RCSR_CRATE "$CD" "$1" 2> "$T/err"; echo "rc=$?"; }

pkg "$T/one/pkg"
mkdir -p "$T/one/other/target"
check one-match "$(run "$T/one")" "$T/one/pkg
rc=0"

mkdir -p "$T/none/a"
printf '[package]\nname = "other"\n' > "$T/none/a/Cargo.toml"
check none "$(run "$T/none")" "rc=1"

pkg "$T/two/a"
pkg "$T/two/b"
check two "$(run "$T/two")" "rc=1"

mkdir -p "$T/libonly/a"
printf '[package]\nname = "other"\n[lib]\nname = "rule-and-compass"\n[[bin]]\nname = "rule-and-compass"\n' > "$T/libonly/a/Cargo.toml"
check lib-bin-only "$(run "$T/libonly")" "rc=1"

[ "$fails" -eq 0 ] && echo "PASS test-crate-dir.sh (4 cases)" && exit 0
exit 1
